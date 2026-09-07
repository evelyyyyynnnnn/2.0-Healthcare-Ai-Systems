"""Discrimination, calibration and alarm-burden metrics.

Discrimination alone is not enough for a clinical alerting system. A model can
rank patients correctly and still emit probabilities that mean nothing, and a
nurse acting on "82% risk" needs that number to be true 82% of the time. Hence
calibration is reported beside AUROC, not instead of it.
"""

from __future__ import annotations

import numpy as np


def auroc(y: np.ndarray, s: np.ndarray) -> float:
    """Rank-based AUROC, ties handled by average rank."""
    pos, neg = int(y.sum()), int((1 - y).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # average ranks within ties
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inv, ranks)
    ranks = (sums / counts)[inv]
    return float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def auprc(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision. The metric that matters when events are rare."""
    order = np.argsort(-s, kind="mergesort")
    y = y[order]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    total = y.sum()
    return float((prec * y).sum() / total) if total else float("nan")


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def reliability(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list:
    """Observed frequency vs predicted probability, per bin."""
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if not m.any():
            continue
        out.append({"bin_lo": round(lo, 3), "bin_hi": round(hi, 3),
                    "n": int(m.sum()),
                    "predicted": round(float(p[m].mean()), 4),
                    "observed": round(float(y[m].mean()), 4)})
    return out


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    """Expected calibration error: mean |predicted - observed|, n-weighted."""
    rows = reliability(y, p, bins)
    n = len(y)
    return float(sum(r["n"] / n * abs(r["predicted"] - r["observed"]) for r in rows))


def threshold_at_sensitivity(y: np.ndarray, s: np.ndarray,
                             target: float = 0.80) -> float:
    """Lowest threshold reaching the target sensitivity.

    Alerting systems are specified by required sensitivity, not by a
    convenient cut-off, so every comparison here is made at matched
    sensitivity. Comparing false-alarm rates at different sensitivities is
    the most common way to make an alerting result look better than it is.
    """
    pos = s[y == 1]
    if len(pos) == 0:
        return float("nan")
    return float(np.quantile(pos, 1.0 - target))


def alarm_stats(y: np.ndarray, s: np.ndarray, thr: float) -> dict:
    pred = s >= thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "threshold": round(float(thr), 5),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else 0.0,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0.0,
        "ppv": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
        "alerts": tp + fp,
        "false_alerts": fp,
        "false_alert_rate": round(fp / (fp + tn), 4) if (fp + tn) else 0.0,
        "false_alerts_per_100_obs": round(100.0 * fp / len(y), 3),
    }
