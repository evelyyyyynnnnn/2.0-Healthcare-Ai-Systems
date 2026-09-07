"""Models, calibration and uncertainty.

Three things a clinical early-warning model needs beyond a score:

  calibration  the number has to mean what it says
  uncertainty  the model has to be able to say it does not know
  a baseline   the comparison has to be against what the ward does today
"""

from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def logistic() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, C=1.0)),
    ])


def boosted() -> HistGradientBoostingClassifier:
    """Histogram-based boosting.

    The exact-split GradientBoostingClassifier took 35 s per fit on this cohort,
    which is 20 minutes once it is wrapped in a calibration CV and an 8-member
    ensemble. The histogram implementation is the same model family at a
    fraction of the cost, and the pipeline stays re-runnable.
    """
    return HistGradientBoostingClassifier(
        max_iter=150, max_depth=4, learning_rate=0.06,
        early_stopping=False, random_state=0)


def calibrated(base, method: str = "isotonic", cv: int = 3):
    """Wrap a model so its outputs are probabilities, not just scores.

    Isotonic by default: it is non-parametric and fixes the S-shaped
    miscalibration that boosted trees reliably produce. It needs more data than
    Platt scaling, which is the trade being made.
    """
    return CalibratedClassifierCV(base, method=method, cv=cv)


class SingleThresholdAlarm:
    """What the ward does today: alarm when one vital crosses a fixed limit.

    This is the baseline any early-warning claim has to beat. Comparing a model
    against nothing, or against chance, is how alerting systems get adopted and
    then ignored.
    """

    name = "single-threshold alarm"

    def __init__(self, feature_index: int, invert: bool = True):
        self.i = feature_index
        self.invert = invert

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        v = X[:, self.i].astype(float)
        v = -v if self.invert else v
        lo, hi = np.percentile(v, 1), np.percentile(v, 99)
        s = np.clip((v - lo) / (hi - lo + 1e-12), 0, 1)
        return np.column_stack([1 - s, s])


class EnsembleUncertainty:
    """Bagged ensemble whose spread is the model's uncertainty estimate.

    Deliberately not a confidence interval. The spread across bootstrap members
    captures uncertainty from finite training data; it says nothing about
    whether the feature set is adequate or the cohort representative. Reporting
    it as if it covered those would be worse than reporting nothing.
    """

    def __init__(self, factory, n_members: int = 8, seed: int = 0):
        self.factory = factory
        self.n_members = n_members
        self.seed = seed
        self.members: list = []

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        self.members = []
        for _ in range(self.n_members):
            idx = rng.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:      # a bootstrap can lose a class
                continue
            m = self.factory()
            m.fit(X[idx], y[idx])
            self.members.append(m)
        return self

    def predict_proba(self, X):
        P = np.stack([m.predict_proba(X)[:, 1] for m in self.members])
        mean = P.mean(axis=0)
        return np.column_stack([1 - mean, mean])

    def uncertainty(self, X) -> np.ndarray:
        P = np.stack([m.predict_proba(X)[:, 1] for m in self.members])
        return P.std(axis=0)


def selective_prediction(y, p, unc, keep_frac: float) -> dict:
    """Accuracy when the model is allowed to abstain on its least certain cases.

    The clinically useful form of an uncertainty estimate: if the model defers
    the most uncertain 20% to a human, does it get better on the rest? If it
    does not, the uncertainty estimate is decoration.
    """
    from .metrics import auroc, brier
    k = max(1, int(len(p) * keep_frac))
    keep = np.argsort(unc)[:k]
    yk, pk = y[keep], p[keep]
    if len(np.unique(yk)) < 2:
        return {"keep_frac": keep_frac, "n": int(k), "auroc": float("nan"),
                "brier": round(brier(yk, pk), 5)}
    return {"keep_frac": keep_frac, "n": int(k),
            "auroc": round(auroc(yk, pk), 4),
            "brier": round(brier(yk, pk), 5),
            "event_rate": round(float(yk.mean()), 4)}
