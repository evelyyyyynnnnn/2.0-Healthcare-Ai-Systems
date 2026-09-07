"""Synthetic arterial blood pressure and photoplethysmogram waveforms.

These are NOT patient data and are not a substitute for it. They exist so the
pipeline has a signal with known ground truth to be tested against: every
artifact below is injected at a known location, which is the only way to
measure whether artifact rejection works rather than assert that it does.

The generator models the features the pipeline actually keys on -- cardiac
periodicity, respiratory modulation, dicrotic notch, baseline wander -- and the
four artifact classes that dominate real ICU records.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

FS_DEFAULT = 125.0  # Hz, the MIMIC-IV waveform sampling rate


@dataclass
class SynthSegment:
    signal: np.ndarray
    fs: float
    artifacts: list = field(default_factory=list)   # (start_idx, end_idx, kind)
    heart_rate_bpm: float = 75.0
    label: str = ""

    def duration_s(self) -> float:
        return len(self.signal) / self.fs

    def artifact_mask(self) -> np.ndarray:
        m = np.zeros(len(self.signal), dtype=bool)
        for a, b, _ in self.artifacts:
            m[a:b] = True
        return m


def _cardiac_cycle(n: int, notch: float = 0.35) -> np.ndarray:
    """One ABP-like beat: sharp upstroke, dicrotic notch, exponential decay."""
    t = np.linspace(0, 1, n, endpoint=False)
    systole = np.exp(-((t - 0.18) ** 2) / (2 * 0.055 ** 2))
    dicrotic = notch * np.exp(-((t - 0.42) ** 2) / (2 * 0.045 ** 2))
    runoff = 0.35 * np.exp(-3.0 * np.clip(t - 0.42, 0, None))
    beat = systole + dicrotic + runoff
    return beat / beat.max()


def make_segment(duration_s: float = 60.0, fs: float = FS_DEFAULT,
                 hr_bpm: float = 75.0, map_mmhg: float = 85.0,
                 pulse_pressure: float = 45.0, resp_rate: float = 15.0,
                 noise: float = 0.012, rng: np.random.Generator | None = None,
                 artifacts: tuple = ()) -> SynthSegment:
    """Build a clean segment, then inject the requested artifacts."""
    rng = rng or np.random.default_rng(0)
    n = int(duration_s * fs)
    samples_per_beat = int(round(60.0 / hr_bpm * fs))

    sig = np.zeros(n)
    i = 0
    while i < n:
        # Beat-to-beat variability: real hearts are not metronomes, and a
        # pipeline tuned on perfectly periodic input fails on real data.
        jitter = int(rng.normal(0, samples_per_beat * 0.025))
        length = max(8, samples_per_beat + jitter)
        beat = _cardiac_cycle(length)
        end = min(n, i + length)
        sig[i:end] += beat[:end - i]
        i += length

    t = np.arange(n) / fs
    resp = 0.06 * np.sin(2 * np.pi * resp_rate / 60.0 * t)          # modulation
    wander = 0.03 * np.sin(2 * np.pi * 0.05 * t)                     # baseline drift
    sig = sig * (1.0 + resp) + wander
    sig = map_mmhg - pulse_pressure * 0.35 + pulse_pressure * sig
    sig += rng.normal(0, noise * pulse_pressure, n)

    seg = SynthSegment(signal=sig, fs=fs, heart_rate_bpm=hr_bpm)
    for kind in artifacts:
        _inject(seg, kind, rng)
    return seg


def _inject(seg: SynthSegment, kind: str, rng: np.random.Generator) -> None:
    n = len(seg.signal)
    fs = seg.fs
    if kind == "flatline":
        # Line disconnection or a stopcock turned to the transducer.
        dur = int(rng.uniform(2, 6) * fs)
        a = rng.integers(0, max(1, n - dur))
        seg.signal[a:a + dur] = seg.signal[a]
        seg.artifacts.append((int(a), int(a + dur), "flatline"))
    elif kind == "saturation":
        # Amplifier rail: values clipped at a ceiling.
        dur = int(rng.uniform(1.5, 4) * fs)
        a = rng.integers(0, max(1, n - dur))
        seg.signal[a:a + dur] = np.clip(seg.signal[a:a + dur] * 2.6, None, 220.0)
        seg.artifacts.append((int(a), int(a + dur), "saturation"))
    elif kind == "motion":
        # Patient movement or a flush: large low-frequency excursion.
        dur = int(rng.uniform(2, 5) * fs)
        a = rng.integers(0, max(1, n - dur))
        t = np.arange(dur) / fs
        seg.signal[a:a + dur] += 40 * np.sin(2 * np.pi * 0.7 * t) * np.hanning(dur)
        seg.artifacts.append((int(a), int(a + dur), "motion"))
    elif kind == "spike":
        # Electrocautery or a transient: isolated extreme samples.
        for _ in range(int(rng.integers(3, 8))):
            a = int(rng.integers(0, n - 3))
            seg.signal[a:a + 2] += rng.choice([-1, 1]) * rng.uniform(60, 120)
            seg.artifacts.append((a, a + 2, "spike"))


def make_cohort(n_segments: int = 60, duration_s: float = 60.0,
                fs: float = FS_DEFAULT, artifact_rate: float = 0.45,
                seed: int = 7) -> list:
    """A cohort with a known artifact prevalence."""
    rng = np.random.default_rng(seed)
    kinds = ("flatline", "saturation", "motion", "spike")
    out = []
    for i in range(n_segments):
        arts = ()
        if rng.random() < artifact_rate:
            k = rng.integers(1, 3)
            arts = tuple(rng.choice(kinds, size=int(k), replace=False))
        out.append(make_segment(
            duration_s=duration_s, fs=fs,
            hr_bpm=float(rng.uniform(55, 110)),
            map_mmhg=float(rng.uniform(62, 105)),
            pulse_pressure=float(rng.uniform(30, 60)),
            resp_rate=float(rng.uniform(10, 22)),
            rng=rng, artifacts=arts))
    return out
