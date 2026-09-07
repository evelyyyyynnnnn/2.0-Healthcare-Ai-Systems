"""Subject-grouped splitting.

The most common defect in clinical prediction code, and the one that most
inflates published numbers: splitting rows instead of patients. Two windows
from one admission share a baseline, a trajectory and often the label, so a
random row split lets the model memorise the patient.

`assert_no_subject_leak` exists so that this can be a test in a pipeline rather
than a convention someone remembers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class SubjectLeakError(AssertionError):
    """Raised when the same subject appears in two folds."""


@dataclass
class GroupSplit:
    train: np.ndarray          # boolean mask
    test: np.ndarray

    def sizes(self) -> tuple:
        return int(self.train.sum()), int(self.test.sum())


def split_by_subject(subjects, test_frac: float = 0.3,
                     seed: int = 0) -> GroupSplit:
    subjects = np.asarray(subjects)
    uniq = np.unique(subjects)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_ids = set(uniq[order[:n_test]].tolist())
    test = np.fromiter((s in test_ids for s in subjects), bool, len(subjects))
    return GroupSplit(train=~test, test=test)


def group_kfold(subjects, n_splits: int = 5, seed: int = 0) -> list:
    """K folds that never split a subject across them."""
    subjects = np.asarray(subjects)
    uniq = np.unique(subjects)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    folds = np.array_split(uniq[order], n_splits)
    out = []
    for f in folds:
        ids = set(f.tolist())
        test = np.fromiter((s in ids for s in subjects), bool, len(subjects))
        out.append(GroupSplit(train=~test, test=test))
    return out


def assert_no_subject_leak(subjects, train_mask, test_mask) -> None:
    subjects = np.asarray(subjects)
    shared = set(subjects[train_mask].tolist()) & set(subjects[test_mask].tolist())
    if shared:
        raise SubjectLeakError(
            f"{len(shared)} subject(s) appear in both train and test, "
            f"e.g. {sorted(shared)[:5]}")
