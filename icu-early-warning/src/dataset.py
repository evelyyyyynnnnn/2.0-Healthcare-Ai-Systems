"""Turn patient time series into a supervised prediction-horizon dataset.

The label is the question the clinician actually asks: will this patient
deteriorate in the next H hours? Not "is this patient deteriorating now",
which is a monitor's job and needs no model.

Splitting is by patient, never by observation. Two windows from the same
patient share a baseline and a latent trajectory, so a random row split leaks
that patient across train and test and inflates every metric on this page.
"""

from __future__ import annotations

import numpy as np

from .cohort import EVENTS, VITALS, Patient

# Features are current value, short trend and short volatility per vital.
# Trend matters because deterioration is a direction before it is a level.
LAGS = (1, 2, 4)


def _feature_columns(p: Patient) -> tuple:
    """All feature columns for one patient, vectorised.

    The obvious per-row dict comprehension is correct and takes 68 seconds on a
    400-patient cohort, which is long enough that nobody re-runs the pipeline
    while iterating. Building each column once with array operations gives the
    same matrix in well under a second.
    """
    n = p.n_steps()
    names: list = ["age"]
    cols: list = [np.full(n, p.age)]
    for v in VITALS:
        s = p.vitals[v]
        names.append(v)
        cols.append(s)
        for L in LAGS:
            idx = np.maximum(0, np.arange(n) - L)
            names.append(f"{v}_delta{L}")
            cols.append(s - s[idx])
        # trailing 7-sample std, computed from cumulative moments
        w = 7
        c1 = np.concatenate([[0.0], np.cumsum(s)])
        c2 = np.concatenate([[0.0], np.cumsum(s * s)])
        lo = np.maximum(0, np.arange(n) - w + 1)
        hi = np.arange(n) + 1
        cnt = (hi - lo).astype(float)
        m1 = (c1[hi] - c1[lo]) / cnt
        m2 = (c2[hi] - c2[lo]) / cnt
        names.append(f"{v}_std6")
        cols.append(np.sqrt(np.maximum(0.0, m2 - m1 * m1)))
    return names, np.column_stack(cols)


def _feature_names() -> tuple:
    dummy = Patient(0, np.zeros(8), {v: np.zeros(8) for v in VITALS},
                    {e: np.zeros(8, bool) for e in EVENTS})
    return tuple(_feature_columns(dummy)[0])


FEATURE_NAMES = _feature_names()


def build(patients: list, event: str = "hypotension",
          horizon_h: float = 4.0, step_h: float = 0.5,
          min_history: int = 4):
    """Return X, y, groups (patient id per row), times."""
    horizon_steps = int(horizon_h / step_h)
    Xs, ys, gs, ts = [], [], [], []
    for p in patients:
        ev = p.events[event]
        n = p.n_steps()
        hi = n - horizon_steps
        if hi <= min_history:
            continue
        _, M = _feature_columns(p)
        idx = np.arange(min_history, hi)
        idx = idx[~ev[idx]]          # not already deteriorating
        if len(idx) == 0:
            continue
        # future[i] = does the event occur in the next `horizon_steps` steps?
        fut = np.array([ev[i + 1:i + 1 + horizon_steps].any() for i in idx])
        Xs.append(M[idx])
        ys.append(fut.astype(int))
        gs.append(np.full(len(idx), p.pid))
        ts.append(p.times[idx])
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(gs), np.concatenate(ts))


def split_by_patient(groups: np.ndarray, test_frac: float = 0.3,
                     seed: int = 3):
    """Grouped split. Never split rows -- see the module docstring."""
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * test_frac))
    test_ids = set(uniq[:n_test].tolist())
    test_mask = np.array([g in test_ids for g in groups])
    return ~test_mask, test_mask
