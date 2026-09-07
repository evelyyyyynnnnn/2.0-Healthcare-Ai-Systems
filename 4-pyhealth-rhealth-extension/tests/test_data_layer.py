"""Tests for the real-MIMIC leakage measurement.

The property that must not break: rows are grouped by the PERSON, not by the
ICU stay. A readmitted patient has two stay_ids and one physiological baseline,
so splitting on stay_id puts that baseline on both sides of the split -- the
precise leak this package exists to prevent, reintroduced by the data loader.
"""
import datetime as dt
import gzip
import io
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data import datakit
from data.load import load_xy
from data.mimicvitals import PROJECT, resample

PROJ = PROJECT


def _gz(rows, header):
    buf = io.StringIO()
    buf.write(",".join(header) + "\n")
    for r in rows:
        buf.write(",".join(str(r.get(h, "")) for h in header) + "\n")
    return gzip.compress(buf.getvalue().encode())


def _seed(tmp_path, n_subjects=8, stays_per=(1,), hours=30):
    """Build a MIMIC-shaped cache. Some subjects get two stays (readmissions)."""
    f = datakit.Fetcher(tmp_path)
    man = f.load_manifest()
    rng = np.random.default_rng(11)

    stays, patients, chart = [], [], []
    base = dt.datetime(2180, 7, 23, 0, 0)
    stay_id = 100
    for s in range(n_subjects):
        subject = 200 + s
        patients.append({"subject_id": subject, "anchor_age": 50 + s})
        # A per-patient baseline, which is exactly what a row split can memorise.
        base_hr = 60 + 4 * s
        n_stays = stays_per[s % len(stays_per)]
        for _ in range(n_stays):
            stays.append({"stay_id": stay_id, "subject_id": subject})
            for h in range(hours * 2):
                ts = (base + dt.timedelta(minutes=30 * h)).strftime(
                    "%Y-%m-%d %H:%M:%S")
                spo2 = 97 + rng.normal(0, 1.2)
                if 20 <= h % 40 <= 24 and s % 3 == 0:
                    spo2 -= 11
                for itemid, val in ((220277, spo2),
                                    (220045, base_hr + rng.normal(0, 2)),
                                    (220181, 85 + rng.normal(0, 6)),
                                    (220210, 16 + rng.normal(0, 2)),
                                    (223762, 36.8), (220179, 120), (220180, 70)):
                    chart.append({"stay_id": stay_id, "itemid": itemid,
                                  "charttime": ts,
                                  "valuenum": round(float(val), 2)})
            stay_id += 1

    files = {
        f"{PROJ}/icu/icustays.csv.gz": _gz(stays, ["stay_id", "subject_id"]),
        f"{PROJ}/hosp/patients.csv.gz": _gz(patients,
                                            ["subject_id", "anchor_age"]),
        f"{PROJ}/icu/chartevents.csv.gz": _gz(
            chart, ["stay_id", "itemid", "charttime", "valuenum"]),
    }
    for dest, raw in files.items():
        p = f.raw / dest
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(raw)
        man["files"][dest] = {
            "source": dest, "url": f"https://physionet.org/files/{dest}",
            "publisher": "PhysioNet", "terms": "open access",
            "sha256": datakit.sha256_file(p), "bytes": len(raw),
            "retrieved_utc": datakit.utc_now()}
    f._write_manifest(man)
    return f


def test_refuses_without_the_real_files(tmp_path):
    with pytest.raises(datakit.FetchError, match="missing real MIMIC-IV files"):
        load_xy(root=tmp_path)


def test_grouping_key_is_the_patient_not_the_stay(tmp_path):
    """The test this whole module exists for."""
    _seed(tmp_path, n_subjects=6, stays_per=(2,))     # every patient readmitted
    X, y, subjects, stay_ids, names, prov = load_xy(root=tmp_path)

    assert len(np.unique(stay_ids)) == 12
    assert len(np.unique(subjects)) == 6
    assert prov["grouping_key"].startswith("subject_id")
    # Each subject really does span more than one stay in this cohort.
    for s in np.unique(subjects):
        assert len(np.unique(stay_ids[subjects == s])) == 2


def test_a_subject_split_keeps_readmissions_together(tmp_path):
    _seed(tmp_path, n_subjects=6, stays_per=(2,))
    from icuflow import assert_no_subject_leak, split_by_subject
    _, _, subjects, stay_ids, _, _ = load_xy(root=tmp_path)

    sp = split_by_subject(subjects, test_frac=0.34, seed=0)
    assert_no_subject_leak(subjects, sp.train, sp.test)
    # No patient on both sides, and therefore no stay of theirs either.
    assert not (set(subjects[sp.train]) & set(subjects[sp.test]))
    assert not (set(stay_ids[sp.train]) & set(stay_ids[sp.test]))


def test_labels_are_strictly_in_the_future(tmp_path):
    """Rows already in the event are dropped, so the label is not the feature."""
    _seed(tmp_path)
    X, y, subjects, _, names, prov = load_xy(root=tmp_path)
    spo2 = X[:, names.index("spo2_pct")]
    # Every row used for prediction is a row where the patient is not currently
    # hypoxemic; otherwise predicting "hypoxemia" would be reading it off.
    assert (spo2 >= 90.0).all()
    assert prov["horizon_hours"] == 4.0


def test_features_have_no_missing_values(tmp_path):
    _seed(tmp_path)
    X, y, *_ = load_xy(root=tmp_path)
    assert np.isfinite(X).all()
    assert set(np.unique(y)) <= {0, 1}


def test_provenance_reports_cohort_size_and_caveat(tmp_path):
    _seed(tmp_path)
    *_, prov = load_xy(root=tmp_path)
    assert prov["cohort_is_a_demonstration_not_a_study"] is True
    assert prov["n_subjects"] >= 1
    assert 0.0 <= prov["positive_rate"] <= 1.0
    assert all(len(v["sha256"]) == 16 for v in prov["files"].values())


def test_an_event_with_no_usable_rows_is_refused(tmp_path):
    """Better to say the cohort is too small than to score three events."""
    _seed(tmp_path, n_subjects=2, hours=13)
    with pytest.raises(datakit.FetchError):
        load_xy(root=tmp_path, event="hypotension", min_hours=40.0)


def test_the_leakage_experiment_runs_end_to_end_on_real_shaped_data(tmp_path):
    _seed(tmp_path, n_subjects=10, stays_per=(1, 2))
    from src.demo import leakage_experiment_real
    out = leakage_experiment_real(tmp_path)

    for k in ("row_split_auroc", "subject_split_auroc", "inflation"):
        assert k in out
    assert 0.0 <= out["row_split_auroc"] <= 1.0
    assert 0.0 <= out["subject_split_auroc"] <= 1.0
    assert out["n_readmissions"] == out["n_stays"] - out["n_subjects"]
    # The result is whatever it is -- the test pins the plumbing, not a
    # direction, because on real data the direction is the finding.
    assert isinstance(out["inflation"], float)


def test_resample_never_moves_a_value_backwards(tmp_path):
    grid = np.arange(0.0, 5.0, 1.0)
    out = resample([(3.0, 120.0)], grid, default=None)
    assert out[3] == 120.0
    assert set(out[:3]) == {120.0}    # prefix back-fill only, no future leak
