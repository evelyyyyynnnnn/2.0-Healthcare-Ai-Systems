import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.cohort import make_cohort, cohort_stats, EVENTS, VITALS
from src.dataset import build, split_by_patient, FEATURE_NAMES
from src.metrics import (alarm_stats, auprc, auroc, brier, ece, reliability,
                         threshold_at_sensitivity)
from src.models import EnsembleUncertainty, SingleThresholdAlarm, boosted, logistic


@pytest.fixture(scope="module")
def cohort():
    return make_cohort(n_patients=120, hours=48.0, step_h=0.5, seed=11)


# --- metric correctness on known inputs ----------------------------------

def test_auroc_of_a_perfect_ranking_is_one():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0


def test_auroc_of_a_reversed_ranking_is_zero():
    y = np.array([0, 0, 1, 1])
    assert auroc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auroc_handles_ties_as_one_half():
    y = np.array([0, 1])
    assert auroc(y, np.array([0.5, 0.5])) == 0.5


def test_brier_is_zero_for_perfect_probabilities():
    assert brier(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0


def test_ece_is_zero_for_a_perfectly_calibrated_forecast():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 40000)
    y = (rng.uniform(0, 1, 40000) < p).astype(int)
    assert ece(y, p) < 0.02


def test_threshold_at_sensitivity_achieves_the_target():
    rng = np.random.default_rng(1)
    y = (rng.uniform(size=4000) < 0.2).astype(int)
    s = rng.uniform(size=4000) + y * 0.4
    thr = threshold_at_sensitivity(y, s, 0.80)
    assert alarm_stats(y, s, thr)["sensitivity"] >= 0.78


# --- cohort realism ------------------------------------------------------

def test_event_rate_is_clinically_plausible(cohort):
    """A 38% event rate makes AUPRC look excellent and is nothing like a ward."""
    s = cohort_stats(cohort)
    for e in EVENTS:
        assert 0.002 < s[f"{e}_rate"] < 0.06, f"{e} at {s[f'{e}_rate']:.2%} per step"


def test_vitals_stay_physiological(cohort):
    for p in cohort[:20]:
        assert np.all(p.vitals["spo2_pct"] <= 100)
        assert np.all(p.vitals["spo2_pct"] >= 70)
        assert np.all(p.vitals["map_mmhg"] > 30)


def test_every_declared_vital_is_present(cohort):
    assert set(cohort[0].vitals) == set(VITALS)


# --- dataset -------------------------------------------------------------

def test_feature_matrix_matches_declared_names(cohort):
    X, y, g, t = build(cohort, event="hypotension", horizon_h=4.0)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert len(y) == len(g) == len(t) == len(X)


def test_rows_during_an_active_event_are_excluded(cohort):
    """Predicting an event already in progress is not early warning."""
    X, y, g, _ = build(cohort, event="hypotension", horizon_h=4.0)
    assert 0.0 < y.mean() < 0.5


def test_split_is_by_patient_not_by_row(cohort):
    _, _, g, _ = build(cohort, event="hypotension", horizon_h=4.0)
    tr, te = split_by_patient(g, test_frac=0.3, seed=3)
    train_ids, test_ids = set(g[tr].tolist()), set(g[te].tolist())
    assert not (train_ids & test_ids), "a patient appears in both splits"
    assert train_ids and test_ids


def test_no_nan_or_inf_in_features(cohort):
    X, _, _, _ = build(cohort, event="hypoxemia", horizon_h=4.0)
    assert np.isfinite(X).all()


# --- models --------------------------------------------------------------

def test_model_beats_the_single_threshold_alarm(cohort):
    X, y, g, _ = build(cohort, event="hypotension", horizon_h=4.0)
    tr, te = split_by_patient(g, 0.3, 3)
    base = SingleThresholdAlarm(FEATURE_NAMES.index("map_mmhg")).fit(X[tr], y[tr])
    m = logistic().fit(X[tr], y[tr])
    a_base = auroc(y[te], base.predict_proba(X[te])[:, 1])
    a_model = auroc(y[te], m.predict_proba(X[te])[:, 1])
    assert a_model > a_base, f"model {a_model:.3f} did not beat alarm {a_base:.3f}"


def test_model_beats_chance(cohort):
    X, y, g, _ = build(cohort, event="hypoxemia", horizon_h=4.0)
    tr, te = split_by_patient(g, 0.3, 3)
    m = logistic().fit(X[tr], y[tr])
    assert auroc(y[te], m.predict_proba(X[te])[:, 1]) > 0.60


def test_performance_stays_below_the_synthetic_ceiling(cohort):
    """Events are driven by a latent the features only partly observe.

    An AUROC near 1.0 would mean the generator leaks the label into the
    features, which would make every other number on the site meaningless.
    """
    X, y, g, _ = build(cohort, event="hypotension", horizon_h=4.0)
    tr, te = split_by_patient(g, 0.3, 3)
    m = boosted().fit(X[tr], y[tr])
    a = auroc(y[te], m.predict_proba(X[te])[:, 1])
    assert a < 0.90, f"AUROC {a:.3f} suggests label leakage in the generator"


def test_ensemble_reports_nonzero_uncertainty(cohort):
    X, y, g, _ = build(cohort, event="hypotension", horizon_h=4.0)
    tr, te = split_by_patient(g, 0.3, 3)
    ens = EnsembleUncertainty(boosted, n_members=4, seed=1).fit(X[tr], y[tr])
    u = ens.uncertainty(X[te])
    assert u.shape == (te.sum(),)
    assert u.mean() > 0


def test_alarm_stats_are_internally_consistent():
    y = np.array([0, 0, 1, 1, 1])
    s = np.array([0.1, 0.6, 0.2, 0.7, 0.9])
    a = alarm_stats(y, s, 0.5)
    assert a["tp"] + a["fp"] + a["fn"] + a["tn"] == len(y)
    assert a["alerts"] == a["tp"] + a["fp"]
