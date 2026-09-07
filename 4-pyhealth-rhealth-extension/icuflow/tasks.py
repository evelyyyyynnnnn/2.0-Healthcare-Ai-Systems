"""Task construction in the shape PyHealth and RHealth pipelines expect.

Both frameworks model a task as: take a patient record, emit samples. These
helpers cover the two steps every ICU horizon task needs and that every project
rewrites -- windowing a series, and labelling a prediction horizon without
leaking the present into the future.
"""

from __future__ import annotations

import numpy as np


def sliding_windows(n: int, window: int, stride: int = 1):
    """Yield (start, end) index pairs. End is exclusive."""
    if window <= 0 or stride <= 0:
        raise ValueError("window and stride must be positive")
    for a in range(0, max(0, n - window + 1), stride):
        yield a, a + window


def horizon_label(events, index: int, horizon_steps: int,
                  exclude_active: bool = True):
    """Does an event occur in the horizon strictly after `index`?

    Returns None when the row should be dropped, which happens when the event
    is already active. Predicting an event in progress is not early warning,
    and including those rows inflates every metric that follows.
    """
    ev = np.asarray(events, bool)
    if exclude_active and ev[index]:
        return None
    future = ev[index + 1: index + 1 + horizon_steps]
    if len(future) < horizon_steps:
        return None                     # not enough follow-up to label honestly
    return int(future.any())


def build_horizon_task(subjects, series_by_subject, events_by_subject,
                       featurise, horizon_steps: int, min_history: int = 4):
    """Assemble X, y, subject arrays for a horizon-prediction task."""
    X, y, subj = [], [], []
    for s in subjects:
        ev = np.asarray(events_by_subject[s], bool)
        n = len(ev)
        for i in range(min_history, n):
            lab = horizon_label(ev, i, horizon_steps)
            if lab is None:
                continue
            X.append(featurise(series_by_subject[s], i))
            y.append(lab)
            subj.append(s)
    return np.asarray(X, float), np.asarray(y, int), np.asarray(subj)
