"""End-to-end: waveform in, derived feature dataset out.

Also the place where the pipeline is held accountable. Because the synthetic
cohort carries artifact locations as ground truth, artifact rejection can be
scored rather than asserted -- which is the difference between a pipeline that
works and one that runs.
"""

from __future__ import annotations

import csv
import pathlib

import numpy as np

from .features import FEATURE_NAMES, extract
from .quality import assess_window, window_signal

ROOT = pathlib.Path(__file__).resolve().parent.parent


def record_path(p) -> str:
    """How a written file is named in the result.

    The result file is cited as evidence, so it names the output relative to
    the project. The absolute path of whichever machine produced the run is
    not evidence of anything, is wrong on every other machine, and discloses
    the author's home directory to anyone who opens the published page.
    """
    p = pathlib.Path(p)
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return p.name


def process_segment(sig: np.ndarray, fs: float, window_s: float = 10.0,
                    **qkw) -> tuple:
    """Return (rows, qualities) for one segment."""
    rows, quals = [], []
    for a, win in window_signal(sig, fs, window_s, window_s):
        q = assess_window(win, fs, a, **qkw)
        quals.append(q)
        if q.accepted:
            rows.append(extract(win, fs, sqi=q.sqi))
    return rows, quals


# Artifacts split into two kinds, and they need different contamination rules.
# A flatline or motion excursion corrupts a window in proportion to how much of
# it they cover. A spike is two samples wide -- 0.16% of a 10-second window at
# 125 Hz -- but it still destroys any beat-detection done on that window.
#
# Scoring both with one fractional-overlap rule was wrong, and wrong in a way
# that looked like a detector failure: the spike rule was correctly rejecting
# spiked windows, while the ground truth said those windows were clean, so
# precision read 0.07 for a detector that was working.
WIDE_ARTIFACTS = frozenset({"flatline", "saturation", "motion"})
NARROW_ARTIFACTS = frozenset({"spike"})


def _is_contaminated(seg, a: int, b: int, overlap_frac: float) -> bool:
    wide_cover = 0
    for s, e, kind in seg.artifacts:
        lo, hi = max(a, s), min(b, e)
        if hi <= lo:
            continue
        if kind in NARROW_ARTIFACTS:
            return True                      # presence is enough
        wide_cover += hi - lo
    return (wide_cover / max(1, b - a)) >= overlap_frac


def score_rejection(segments, fs: float, window_s: float = 10.0,
                    overlap_frac: float = 0.25, **qkw) -> dict:
    """Score artifact rejection against the injected ground truth.

    Wide artifacts count when they cover at least `overlap_frac` of the window;
    narrow ones count whenever they touch it. Both thresholds are judgement
    calls and both are stated rather than buried.
    """
    tp = fp = tn = fn = 0
    for seg in segments:
        for a, win in window_signal(seg.signal, fs, window_s, window_s):
            contaminated = _is_contaminated(seg, a, a + len(win), overlap_frac)
            q = assess_window(win, fs, a, **qkw)
            rejected = not q.accepted
            if contaminated and rejected:
                tp += 1
            elif contaminated and not rejected:
                fn += 1
            elif not contaminated and rejected:
                fp += 1
            else:
                tn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": round(prec, 4), "recall": round(rec, 4),
        "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0.0,
        "overlap_threshold": overlap_frac,
    }


def build_dataset(segments, fs: float, window_s: float = 10.0,
                  out_path: str | pathlib.Path | None = None, **qkw) -> dict:
    all_rows, n_windows, n_accepted = [], 0, 0
    total_samples = 0
    for sid, seg in enumerate(segments):
        rows, quals = process_segment(seg.signal, fs, window_s, **qkw)
        total_samples += len(seg.signal)
        n_windows += len(quals)
        n_accepted += sum(1 for q in quals if q.accepted)
        for r in rows:
            all_rows.append({"segment_id": sid, **r})

    if out_path:
        p = pathlib.Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf8") as fh:
            w = csv.DictWriter(fh, fieldnames=["segment_id", *FEATURE_NAMES])
            w.writeheader()
            for r in all_rows:
                w.writerow(r)

    hours = total_samples / fs / 3600.0
    return {
        "n_segments": len(segments),
        "waveform_hours": round(hours, 4),
        "n_windows": n_windows,
        "n_accepted": n_accepted,
        "acceptance_rate": round(n_accepted / n_windows, 4) if n_windows else 0.0,
        "n_feature_rows": len(all_rows),
        "features": list(FEATURE_NAMES),
        "output": record_path(out_path) if out_path else None,
    }


def reject_reason_histogram(segments, fs: float, window_s: float = 10.0,
                            **qkw) -> dict:
    """Which rule is doing the work? A rule that never fires is dead weight."""
    counts: dict = {}
    for seg in segments:
        for a, win in window_signal(seg.signal, fs, window_s, window_s):
            q = assess_window(win, fs, a, **qkw)
            for code in q.codes():
                counts[code] = counts.get(code, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
