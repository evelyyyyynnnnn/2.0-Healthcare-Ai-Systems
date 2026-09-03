"""Alarm-burden comparison at matched sensitivity.

Any alerting system can cut false alarms by catching fewer events. Comparing
two systems at whatever operating point each happens to sit on is the most
common way an alerting result is made to look better than it is, so this module
only compares at a sensitivity you have to name.
"""

from __future__ import annotations

import numpy as np


def threshold_at_sensitivity(y, scores, target: float = 0.80) -> float:
    y = np.asarray(y)
    s = np.asarray(scores, float)
    pos = s[y == 1]
    if len(pos) == 0:
        return float("nan")
    return float(np.quantile(pos, 1.0 - target))


def alarm_stats(y, scores, threshold: float) -> dict:
    y = np.asarray(y)
    s = np.asarray(scores, float)
    pred = s >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "sensitivity": tp / (tp + fn) if (tp + fn) else 0.0,
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "ppv": tp / (tp + fp) if (tp + fp) else 0.0,
        "false_alerts": fp,
        "false_alerts_per_100": 100.0 * fp / len(y) if len(y) else 0.0,
    }


def compare_at_matched_sensitivity(y, baseline_scores, model_scores,
                                   target: float = 0.80) -> dict:
    """Set both systems to the same sensitivity, then count false alerts."""
    b = alarm_stats(y, baseline_scores,
                    threshold_at_sensitivity(y, baseline_scores, target))
    m = alarm_stats(y, model_scores,
                    threshold_at_sensitivity(y, model_scores, target))
    denom = b["false_alerts"] or 1
    return {
        "target_sensitivity": target,
        "baseline": b,
        "model": m,
        "false_alert_reduction_pct": 100.0 * (b["false_alerts"] - m["false_alerts"]) / denom,
        "sensitivity_gap": m["sensitivity"] - b["sensitivity"],
    }
