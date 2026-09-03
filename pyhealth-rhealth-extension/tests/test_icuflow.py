import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from icuflow import (IsotonicCalibrator, PlattCalibrator, alarm_stats,
                     assert_no_subject_leak, compare_at_matched_sensitivity,
                     ece, group_kfold, horizon_label, reliability_table,
                     sliding_windows, split_by_subject)
from icuflow.splits import SubjectLeakError


# --- splits --------------------------------------------------------------

def test_split_never_shares_a_subject():
    subj = np.repeat(np.arange(50), 10)
    sp = split_by_subject(subj, 0.3, seed=0)
    assert not (set(subj[sp.train].tolist()) & set(subj[sp.test].tolist()))


def test_split_covers_every_row():
    subj = np.repeat(np.arange(30), 7)
    sp = split_by_subject(subj, 0.25, seed=1)
    assert (sp.train | sp.test).all()
    assert not (sp.train & sp.test).any()


def test_split_is_deterministic_for_a_seed():
    subj = np.repeat(np.arange(40), 5)
    a = split_by_subject(subj, 0.3, seed=7)
    b = split_by_subject(subj, 0.3, seed=7)
    assert np.array_equal(a.test, b.test)


def test_group_kfold_partitions_subjects_exactly_once():
    subj = np.repeat(np.arange(40), 6)
    folds = group_kfold(subj, n_splits=4, seed=2)
    assert len(folds) == 4
    seen = np.zeros(len(subj), int)
    for f in folds:
        seen += f.test.astype(int)
        assert not (set(subj[f.train].tolist()) & set(subj[f.test].tolist()))
    assert (seen == 1).all(), "every row must be tested exactly once"


def test_leak_guard_raises():
    subj = np.repeat(np.arange(10), 4)
    with pytest.raises(SubjectLeakError):
        assert_no_subject_leak(subj, np.ones(len(subj), bool),
                               np.ones(len(subj), bool))


def test_leak_guard_passes_a_clean_split():
    subj = np.repeat(np.arange(10), 4)
    sp = split_by_subject(subj, 0.3, seed=0)
    assert_no_subject_leak(subj, sp.train, sp.test)     # must not raise


# --- calibration ---------------------------------------------------------

def test_isotonic_is_monotone():
    """Calibration must move probabilities without reordering patients."""
    rng = np.random.default_rng(0)
    s = rng.uniform(0, 1, 2000)
    y = (rng.uniform(size=2000) < s).astype(int)
    cal = IsotonicCalibrator().fit(s, y)
    grid = np.linspace(0, 1, 200)
    out = cal.transform(grid)
    assert np.all(np.diff(out) >= -1e-9)


def test_isotonic_improves_a_distorted_score():
    rng = np.random.default_rng(1)
    p = rng.uniform(0.02, 0.98, 6000)
    y = (rng.uniform(size=6000) < p).astype(int)
    bad = p ** 3 / (p ** 3 + (1 - p) ** 3)
    cal = IsotonicCalibrator().fit(bad[:3000], y[:3000])
    assert ece(y[3000:], cal.transform(bad[3000:])) < ece(y[3000:], bad[3000:])


def test_platt_improves_a_distorted_score():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.95, 6000)
    y = (rng.uniform(size=6000) < p).astype(int)
    bad = p ** 3 / (p ** 3 + (1 - p) ** 3)
    cal = PlattCalibrator().fit(bad[:3000], y[:3000])
    assert ece(y[3000:], cal.transform(bad[3000:])) < ece(y[3000:], bad[3000:])


def test_ece_near_zero_for_a_calibrated_forecast():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 50000)
    y = (rng.uniform(size=50000) < p).astype(int)
    assert ece(y, p) < 0.02


def test_reliability_bins_sum_to_the_sample():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 1, 1000)
    y = (rng.uniform(size=1000) < p).astype(int)
    assert sum(r["n"] for r in reliability_table(y, p)) == 1000


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError):
        IsotonicCalibrator().transform(np.array([0.5]))


# --- alarms --------------------------------------------------------------

def test_matched_comparison_equalises_sensitivity():
    rng = np.random.default_rng(5)
    y = (rng.uniform(size=5000) < 0.15).astype(int)
    base = rng.uniform(size=5000) + y * 0.2
    model = rng.uniform(size=5000) + y * 0.7
    cmp = compare_at_matched_sensitivity(y, base, model, 0.80)
    assert abs(cmp["sensitivity_gap"]) < 0.05


def test_better_model_cuts_false_alerts_at_matched_sensitivity():
    rng = np.random.default_rng(6)
    y = (rng.uniform(size=5000) < 0.15).astype(int)
    base = rng.uniform(size=5000) + y * 0.15
    model = rng.uniform(size=5000) + y * 0.9
    cmp = compare_at_matched_sensitivity(y, base, model, 0.80)
    assert cmp["false_alert_reduction_pct"] > 0


def test_alarm_counts_partition_the_sample():
    y = np.array([0, 1, 1, 0, 1])
    s = np.array([0.1, 0.9, 0.4, 0.6, 0.8])
    a = alarm_stats(y, s, 0.5)
    assert a["tp"] + a["fp"] + a["fn"] + a["tn"] == len(y)


# --- tasks ---------------------------------------------------------------

def test_horizon_label_drops_active_events():
    assert horizon_label(np.array([0, 1, 0, 0, 0], bool), 1, 2) is None


def test_horizon_label_drops_short_followup():
    assert horizon_label(np.array([0, 0, 0], bool), 2, 3) is None


def test_horizon_label_sees_only_the_future():
    ev = np.array([1, 0, 0, 0, 1, 0], bool)
    assert horizon_label(ev, 1, 2) == 0        # events at 0 and 4, neither in 2..3
    assert horizon_label(ev, 2, 2) == 1        # event at 4 is inside 3..4


def test_sliding_windows_are_in_range():
    n = 25
    for a, b in sliding_windows(n, 10, 5):
        assert 0 <= a < b <= n


def test_sliding_windows_rejects_bad_arguments():
    with pytest.raises(ValueError):
        list(sliding_windows(10, 0, 1))
