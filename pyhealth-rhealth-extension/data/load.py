"""Build (X, y, subject) from real MIMIC-IV stays.

This is the project's central claim put to a real test. The synthetic
experiment constructs a subject effect that provably cannot generalise, so
row-splitting is guaranteed to inflate the score -- which proves the tool
detects leakage, but not that leakage exists in practice.

Real ICU records answer the harder question. Each patient has a genuine
physiological baseline: their own resting heart rate, their own blood pressure.
A model split by row sees rows from a patient it has already learned the
baseline of. Whether that inflates AUROC on real data is a measurement, and
this is the input for it.

One property matters more than any other here: rows must be grouped by SUBJECT,
not by stay. A patient readmitted to the ICU appears as two stay_ids and one
person, and splitting on stay_id would put the same patient's baseline on both
sides -- the exact leak the project exists to prevent.
"""
from __future__ import annotations

import pathlib

import numpy as np

from .mimicvitals import EVENT_DEFINITIONS, events_from_vitals, stay_series

ROOT = pathlib.Path(__file__).resolve().parent

VITALS = ("map_mmhg", "heart_rate_bpm", "spo2_pct", "resp_rate", "temp_c",
          "pulse_pressure")
LAGS = (1, 2, 4)
HORIZON_STEPS = 8          # 4 hours on a 30-minute grid
MIN_HISTORY = 4


def _features(times, vitals, age):
    n = len(times)
    names = ["age"]
    cols = [np.full(n, age)]
    for v in VITALS:
        s = vitals[v]
        names.append(v)
        cols.append(s)
        for L in LAGS:
            idx = np.maximum(0, np.arange(n) - L)
            names.append(f"{v}_delta{L}")
            cols.append(s - s[idx])
    return names, np.column_stack(cols)


def load_xy(root=ROOT, event: str = "hypoxemia", min_hours: float = 12.0):
    """Return (X, y, subjects, stays, feature_names, provenance)."""
    stays, prov = stay_series(root, min_hours=min_hours)

    Xs, ys, subs, sids = [], [], [], []
    names = None
    for s in stays:
        ev = events_from_vitals(s["vitals"])[event]
        n = len(s["times"])
        hi = n - HORIZON_STEPS
        if hi <= MIN_HISTORY:
            continue
        names, M = _features(s["times"], s["vitals"], s["age"])
        idx = np.arange(MIN_HISTORY, hi)
        idx = idx[~ev[idx]]                      # not already in the event
        if not len(idx):
            continue
        fut = np.array([ev[i + 1:i + 1 + HORIZON_STEPS].any() for i in idx])
        Xs.append(M[idx])
        ys.append(fut.astype(int))
        # Grouping key is the PERSON, not the stay: a readmitted patient must
        # not appear on both sides of a split.
        subs.append(np.full(len(idx), s["subject_id"], dtype=object))
        sids.append(np.full(len(idx), s["stay_id"]))

    if not Xs:
        from .datakit import FetchError
        raise FetchError(
            f"no usable rows for '{event}' in the demo cohort; with ~100 stays "
            f"an event can be too rare to build a prediction task from")

    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    subjects = np.concatenate(subs)
    stay_ids = np.concatenate(sids)

    prov = dict(prov)
    prov.update({
        "event": event,
        "event_definition": EVENT_DEFINITIONS[event],
        "horizon_hours": HORIZON_STEPS * prov["grid_step_hours"],
        "n_rows": int(len(y)),
        "n_subjects": int(len(np.unique(subjects))),
        "n_stays_used": int(len(np.unique(stay_ids))),
        "positive_rate": round(float(y.mean()), 5),
        "grouping_key": "subject_id (the person), not stay_id",
    })
    return X, y, subjects, stay_ids, names, prov
