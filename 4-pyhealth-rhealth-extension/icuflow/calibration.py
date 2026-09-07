"""Calibration: making a score mean what it says.

A model can rank patients correctly and still emit numbers that are not
probabilities. A clinician acting on "82% risk" needs that to be right 82% of
the time, so calibration is reported beside discrimination, never instead of it.
"""

from __future__ import annotations

import numpy as np


class PlattCalibrator:
    """Logistic (Platt) scaling. Parametric, data-efficient, assumes sigmoid."""

    def __init__(self, max_iter: int = 200, lr: float = 0.1):
        self.a, self.b = 1.0, 0.0
        self.max_iter, self.lr = max_iter, lr

    def fit(self, scores, y):
        s = np.asarray(scores, float)
        y = np.asarray(y, float)
        s = (s - s.mean()) / (s.std() + 1e-12)
        self._mu, self._sd = float(np.mean(scores)), float(np.std(scores) + 1e-12)
        for _ in range(self.max_iter):
            p = 1 / (1 + np.exp(-(self.a * s + self.b)))
            ga = float(np.mean((p - y) * s))
            gb = float(np.mean(p - y))
            self.a -= self.lr * ga
            self.b -= self.lr * gb
        return self

    def transform(self, scores):
        s = (np.asarray(scores, float) - self._mu) / self._sd
        return 1 / (1 + np.exp(-(self.a * s + self.b)))


class IsotonicCalibrator:
    """Isotonic regression via pool-adjacent-violators.

    Non-parametric, so it fixes the S-shaped miscalibration boosted trees
    reliably produce. Costs more data than Platt, which is the trade.
    """

    def __init__(self):
        self.x_ = None
        self.y_ = None

    def fit(self, scores, y):
        s = np.asarray(scores, float)
        t = np.asarray(y, float)
        order = np.argsort(s, kind="mergesort")
        s, t = s[order], t[order]

        # PAVA: merge adjacent blocks until the fit is non-decreasing.
        vals = list(t)
        weights = [1.0] * len(t)
        i = 0
        while i < len(vals) - 1:
            if vals[i] <= vals[i + 1] + 1e-12:
                i += 1
                continue
            w = weights[i] + weights[i + 1]
            v = (vals[i] * weights[i] + vals[i + 1] * weights[i + 1]) / w
            vals[i:i + 2] = [v]
            weights[i:i + 2] = [w]
            if i > 0:
                i -= 1
        fitted, k = [], 0
        for v, w in zip(vals, weights):
            fitted.extend([v] * int(round(w)))
        self.x_ = s
        self.y_ = np.asarray(fitted[:len(s)], float)
        return self

    def transform(self, scores):
        if self.x_ is None:
            raise RuntimeError("fit before transform")
        return np.interp(np.asarray(scores, float), self.x_, self.y_)


def reliability_table(y, p, bins: int = 10) -> list:
    y = np.asarray(y, float)
    p = np.asarray(p, float)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if m.any():
            out.append({"bin_lo": float(lo), "bin_hi": float(hi),
                        "n": int(m.sum()),
                        "predicted": float(p[m].mean()),
                        "observed": float(y[m].mean())})
    return out


def ece(y, p, bins: int = 10) -> float:
    rows = reliability_table(y, p, bins)
    n = len(np.asarray(y))
    return float(sum(r["n"] / n * abs(r["predicted"] - r["observed"]) for r in rows))
