"""Signal quality assessment and artifact rejection.

The pipeline's job is to decide, per window, whether a segment of waveform is
trustworthy enough to derive features from. Getting that wrong in either
direction is costly: admitting artifact produces confident garbage downstream,
and rejecting good signal throws away the data an early-warning model needs.

Every rule below is a named, inspectable check with a stated threshold rather
than a learned score, because a clinician asking "why was this window dropped"
deserves an answer better than "the model said so".
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# Physiological plausibility bounds for arterial blood pressure (mmHg).
ABP_MIN, ABP_MAX = 20.0, 220.0


@dataclass(frozen=True)
class ChannelLimits:
    """Plausibility limits for one kind of waveform.

    These were originally constants, which was fine for as long as the only
    input was an arterial line. Running the pipeline on a real
    photoplethysmogram exposed the assumption: PLETH is recorded in arbitrary
    units centred near zero, so every window fell outside 20-220 "mmHg" and the
    gate rejected the entire recording. The signal was fine; the units were an
    assumption nobody had written down.

    delta_max None means the step limit is taken from the window's own
    peak-to-peak range, which is the only sensible rule for a channel whose
    amplitude scale is arbitrary.
    """
    lo: float
    hi: float
    delta_max: float | None
    units: str = "mmHg"
    delta_frac: float = 0.6

    def step_limit(self, x) -> float:
        if self.delta_max is not None:
            return self.delta_max
        ptp = float(np.max(x) - np.min(x)) if len(x) else 0.0
        return max(self.delta_frac * ptp, 1e-9)


ABP_LIMITS = ChannelLimits(ABP_MIN, ABP_MAX, 45.0, units="mmHg")
# A plethysmogram carries no pressure units, so range limits are meaningless
# and the step limit is relative to the window's own amplitude.
PPG_LIMITS = ChannelLimits(float("-inf"), float("inf"), None, units="arbitrary")


@dataclass
class WindowQuality:
    start: int
    end: int
    sqi: float
    flat_frac: float
    out_of_range_frac: float
    spike_count: int
    delta_max: float
    accepted: bool
    reasons: list          # (code, human-readable detail)

    def codes(self) -> list:
        return [c for c, _ in self.reasons]

    def as_dict(self) -> dict:
        d = asdict(self)
        d["reasons"] = [{"code": c, "detail": m} for c, m in self.reasons]
        d["sqi"] = round(self.sqi, 4)
        d["flat_frac"] = round(self.flat_frac, 4)
        d["out_of_range_frac"] = round(self.out_of_range_frac, 4)
        d["delta_max"] = round(self.delta_max, 3)
        return d


def flat_fraction(x: np.ndarray, eps: float = 1e-6) -> float:
    """Share of samples where the signal does not move.

    Catches disconnection and a closed stopcock, both of which produce a
    perfectly constant trace that every amplitude-based check would pass.
    """
    if len(x) < 2:
        return 0.0
    return float(np.mean(np.abs(np.diff(x)) < eps))


def out_of_range_fraction(x: np.ndarray, lo: float = ABP_MIN,
                          hi: float = ABP_MAX) -> float:
    return float(np.mean((x < lo) | (x > hi)))


def spike_count(x: np.ndarray, k: float = 6.0) -> int:
    """Count narrow transients: a large jump immediately reversed.

    Two earlier versions of this got it wrong, and both failures are worth
    keeping in mind because they look like detector problems and are not.

    Scoring |diff| against its own median flags every systolic upstroke, since
    an arterial upstroke genuinely is the largest sample-to-sample change in the
    window. That rejected 99% of clean windows.

    Subtracting a 5-sample running median does not fix it either: the median
    lags a fast upstroke, so each beat leaves ten samples of large residual and
    a clean 10-second window scores ~200 "spikes".

    What actually separates the two is direction. A pressure upstroke is
    monotone -- consecutive differences share a sign for the whole rise. A spike
    goes up and comes straight back down, so its consecutive differences have
    opposite signs. Requiring a sign reversal AND a magnitude above the robust
    scale of the window isolates transients without touching the pulse.
    """
    if len(x) < 4:
        return 0
    d = np.diff(x)
    mag = np.abs(d)
    med = np.median(mag)
    mad = np.median(np.abs(mag - med))
    scale = 1.4826 * mad if mad > 1e-9 else max(med, 1e-9)
    thresh = med + k * scale

    reversal = d[:-1] * d[1:] < 0                     # direction flips
    both_large = np.minimum(mag[:-1], mag[1:]) > thresh
    return int(np.sum(reversal & both_large))


def pulsatility_sqi(x: np.ndarray, fs: float) -> float:
    """Fraction of spectral power in the cardiac band (0.7-3.5 Hz).

    A physiological arterial waveform concentrates power at the heart rate and
    its harmonics. Motion artifact and drift push power below the band;
    high-frequency noise pushes it above. One number, both failure modes.
    """
    if len(x) < int(fs * 2):
        return 0.0
    x = x - np.mean(x)
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / fs)
    total = float(np.sum(spec[freqs > 0.05]))
    if total <= 0:
        return 0.0
    band = float(np.sum(spec[(freqs >= 0.7) & (freqs <= 3.5)]))
    return band / total


def assess_window(x: np.ndarray, fs: float, start: int,
                  sqi_min: float = 0.35, flat_max: float = 0.10,
                  oor_max: float = 0.02, spike_max: int = 3,
                  delta_max_mmhg: float | None = None,
                  limits: ChannelLimits = ABP_LIMITS) -> WindowQuality:
    """Assess one window against the limits for its channel.

    `delta_max_mmhg` still overrides the step limit, so the threshold sweep in
    the demo works unchanged; when it is None the limit comes from `limits`.
    """
    reasons = []
    sqi = pulsatility_sqi(x, fs)
    flat = flat_fraction(x)
    oor = out_of_range_fraction(x, lo=limits.lo, hi=limits.hi)
    spikes = spike_count(x)
    dmax = float(np.max(np.abs(np.diff(x)))) if len(x) > 1 else 0.0
    step_limit = (delta_max_mmhg if delta_max_mmhg is not None
                  else limits.step_limit(x))

    if sqi < sqi_min:
        reasons.append(("low_pulsatility",
                        f"SQI {sqi:.2f} < {sqi_min}"))
    if flat > flat_max:
        reasons.append(("flatline",
                        f"flat for {flat:.0%} of the window"))
    if oor > oor_max:
        reasons.append(("out_of_range",
                        f"{oor:.1%} outside {limits.lo:.0f}-{limits.hi:.0f} "
                        f"{limits.units}"))
    if spikes > spike_max:
        reasons.append(("spikes", f"{spikes} narrow transients"))
    if dmax > step_limit:
        reasons.append(("step_change",
                        f"max jump {dmax:.3g} {limits.units} between samples "
                        f"(limit {step_limit:.3g})"))

    return WindowQuality(
        start=start, end=start + len(x), sqi=sqi, flat_frac=flat,
        out_of_range_frac=oor, spike_count=spikes, delta_max=dmax,
        accepted=not reasons, reasons=reasons)


def window_signal(sig: np.ndarray, fs: float, window_s: float = 10.0,
                  stride_s: float = 10.0):
    """Yield (start_index, window) pairs."""
    w = int(window_s * fs)
    s = int(stride_s * fs)
    for a in range(0, max(0, len(sig) - w + 1), s):
        yield a, sig[a:a + w]


def assess_segment(sig: np.ndarray, fs: float, window_s: float = 10.0,
                   **kw) -> list:
    return [assess_window(win, fs, a, **kw)
            for a, win in window_signal(sig, fs, window_s, window_s)]
