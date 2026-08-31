"""Feature extraction from accepted windows.

Only windows that pass quality assessment reach this module. The features are
the ones an ICU early-warning model actually consumes: beat-level haemodynamics
plus variability, which is where deterioration shows up before the mean does.
"""

from __future__ import annotations

import numpy as np

FEATURE_NAMES = (
    "map_mmhg", "sbp_mmhg", "dbp_mmhg", "pulse_pressure",
    "heart_rate_bpm", "hr_variability_ms", "sbp_slope_mmhg_per_min",
    "pulse_pressure_variation", "sqi",
)


def find_beats(x: np.ndarray, fs: float) -> np.ndarray:
    """Systolic peak indices.

    Two gates, and both are needed. The refractory period alone is not enough:
    the dicrotic notch sits roughly 0.4 of a cardiac cycle after the systolic
    peak, so at 55 bpm it falls 460 ms later -- outside any refractory window
    short enough to permit a 140 bpm rhythm. Detecting on slope alone counted
    it as a beat and reported 143 bpm for a 55 bpm trace.

    The notch is distinguishable by height rather than timing: it reaches about
    a third of the pulse amplitude, never the top. Requiring a candidate to
    clear 60% of the diastolic-to-systolic range rejects it at every heart rate,
    which is why the amplitude gate does the real work here and the refractory
    period only suppresses duplicate detections on the same upstroke.
    """
    if len(x) < int(fs):
        return np.array([], dtype=int)

    dbp = float(np.percentile(x, 3))
    sbp = float(np.percentile(x, 97))
    if sbp - dbp < 1e-6:
        return np.array([], dtype=int)
    height_gate = dbp + 0.60 * (sbp - dbp)

    d = np.diff(x)
    thr = np.percentile(d, 97)
    if thr <= 0:
        return np.array([], dtype=int)
    candidates = np.where(d > thr * 0.5)[0]
    if len(candidates) == 0:
        return np.array([], dtype=int)

    refractory = int(0.3 * fs)          # 200 bpm ceiling
    peaks: list = []
    last = -refractory
    for c in candidates:
        if c - last < refractory:
            continue
        lo, hi = c, min(len(x), c + int(0.25 * fs))
        idx = int(lo + np.argmax(x[lo:hi]))
        if x[idx] < height_gate:
            continue                     # dicrotic notch or a noise blip
        peaks.append(idx)
        last = c
    return np.array(peaks, dtype=int)


def extract(x: np.ndarray, fs: float, sqi: float = 1.0) -> dict:
    peaks = find_beats(x, fs)
    sbp = float(np.percentile(x, 97))
    dbp = float(np.percentile(x, 3))
    mean_p = float(np.mean(x))

    if len(peaks) >= 3:
        rr = np.diff(peaks) / fs
        hr = float(60.0 / np.mean(rr))
        hrv = float(np.std(rr) * 1000.0)
        # Beat-wise systolic values -> pulse-pressure variation, the classic
        # fluid-responsiveness signal.
        sys_vals = x[peaks]
        pp_var = float((np.max(sys_vals) - np.min(sys_vals)) /
                       np.mean(sys_vals) * 100.0) if np.mean(sys_vals) else 0.0
    else:
        hr, hrv, pp_var = float("nan"), float("nan"), float("nan")

    t = np.arange(len(x)) / fs
    slope = float(np.polyfit(t, x, 1)[0] * 60.0) if len(x) > 2 else 0.0

    return {
        "map_mmhg": mean_p,
        "sbp_mmhg": sbp,
        "dbp_mmhg": dbp,
        "pulse_pressure": sbp - dbp,
        "heart_rate_bpm": hr,
        "hr_variability_ms": hrv,
        "sbp_slope_mmhg_per_min": slope,
        "pulse_pressure_variation": pp_var,
        "sqi": sqi,
    }
