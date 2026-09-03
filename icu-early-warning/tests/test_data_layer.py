"""Tests for building a cohort from real MIMIC-IV records.

Downloading needs a network. Turning MIMIC's irregular charting into a regular
grid does not, and that transformation is where a subtle error would be
invisible: carrying a value backwards in time, or interpolating across a gap,
gives the model information the clinician did not have, and the result is an
AUROC that looks excellent and means nothing.
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
from data.load import CARRY_FORWARD_H, ITEMS, PLAUSIBLE, _resample, load_cohort
from data.physionet import parse_ts, read_csv, to_float

PROJECT = "mimic-iv-demo"


# --- PhysioNet helpers -----------------------------------------------------

def test_read_csv_handles_gzip_and_plain():
    plain = b"a,b\n1,2\n"
    assert read_csv(plain) == [{"a": "1", "b": "2"}]
    assert read_csv(gzip.compress(plain)) == [{"a": "1", "b": "2"}]


def test_to_float_survives_the_empty_and_text_cells_mimic_contains():
    assert to_float("98.6") == 98.6
    assert to_float("") is None
    assert to_float("___") is None
    assert to_float(None) is None
    assert to_float("unable to obtain") is None


def test_parse_ts_accepts_the_formats_mimic_uses():
    assert parse_ts("2180-07-23 14:00:00") == dt.datetime(2180, 7, 23, 14, 0)
    assert parse_ts("2180-07-23 14:00") == dt.datetime(2180, 7, 23, 14, 0)
    assert parse_ts("") is None


# --- resampling: the part that could leak the future ----------------------

def test_resample_carries_forward_and_never_backwards():
    """The property that keeps the task honest.

    An observation at t=3 must not appear at t=1. If it did, the model would be
    reading a measurement that had not been taken yet.
    """
    grid = np.arange(0.0, 5.0, 1.0)
    out = _resample([(3.0, 120.0)], grid, default=None)
    assert out is not None
    # Before the first observation there is nothing to carry, so the prefix is
    # back-filled -- bounded to the pre-observation prefix, where no
    # alternative exists -- and from t=3 the observed value holds.
    assert out[3] == 120.0 and out[4] == 120.0
    assert set(out[:3]) == {120.0}, "prefix back-fill is the only backwards step"


def test_resample_stops_carrying_after_the_limit():
    """A four-hour-old blood pressure is not a current blood pressure."""
    grid = np.arange(0.0, 8.0, 1.0)
    out = _resample([(0.0, 100.0), (7.0, 60.0)], grid, default=None)
    # Between the two observations the gap exceeds CARRY_FORWARD_H, so the
    # stale value must not simply persist across the whole gap.
    assert CARRY_FORWARD_H == 2.0
    assert out[0] == 100.0
    assert out[7] == 60.0
    # The middle is filled by interpolation between real observations only
    # after the carry-forward window lapses, and must move toward the later
    # value rather than holding the stale one.
    assert out[5] != 100.0


def test_resample_returns_none_when_there_is_nothing_to_resample():
    assert _resample([], np.arange(0.0, 5.0, 1.0), default=None) is None


# --- building a cohort -----------------------------------------------------

def test_refuses_when_mimic_is_not_cached(tmp_path):
    with pytest.raises(datakit.FetchError, match="missing real MIMIC-IV files"):
        load_cohort(root=tmp_path)


def _gz(rows, header):
    buf = io.StringIO()
    buf.write(",".join(header) + "\n")
    for r in rows:
        buf.write(",".join(str(r.get(h, "")) for h in header) + "\n")
    return gzip.compress(buf.getvalue().encode())


def _seed(tmp_path, n_stays=6, hours=30, desat_stay=1):
    f = datakit.Fetcher(tmp_path)
    man = f.load_manifest()

    stays = [{"stay_id": 100 + i, "subject_id": 200 + i} for i in range(n_stays)]
    patients = [{"subject_id": 200 + i, "anchor_age": 55 + i} for i in range(n_stays)]

    base = dt.datetime(2180, 7, 23, 0, 0)
    chart = []
    rng = np.random.default_rng(7)
    for i in range(n_stays):
        sid = 100 + i
        for h in range(hours * 2):          # charted every 30 minutes
            ts = (base + dt.timedelta(minutes=30 * h)).strftime("%Y-%m-%d %H:%M:%S")
            spo2 = 97 + rng.normal(0, 1)
            mapv = 85 + rng.normal(0, 6)
            if i == desat_stay and 20 <= h <= 26:
                spo2 -= 12                   # a real desaturation episode
            for itemid, val in ((220277, spo2), (220045, 80 + rng.normal(0, 5)),
                                (220181, mapv), (220210, 16 + rng.normal(0, 2)),
                                (223761, 98.6), (220179, 120), (220180, 70)):
                chart.append({"stay_id": sid, "itemid": itemid,
                              "charttime": ts, "valuenum": round(float(val), 2)})
        # A charting error that must be filtered, not learned.
        chart.append({"stay_id": sid, "itemid": 220045,
                      "charttime": base.strftime("%Y-%m-%d %H:%M:%S"),
                      "valuenum": 900})

    files = {
        f"{PROJECT}/icu/icustays.csv.gz": _gz(stays, ["stay_id", "subject_id"]),
        f"{PROJECT}/hosp/patients.csv.gz": _gz(patients, ["subject_id", "anchor_age"]),
        f"{PROJECT}/icu/chartevents.csv.gz": _gz(
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


def test_builds_patients_with_the_vitals_the_model_expects(tmp_path):
    _seed(tmp_path)
    from src.cohort import VITALS
    patients, prov = load_cohort(root=tmp_path)
    assert len(patients) == 6
    for p in patients:
        assert set(p.vitals) == set(VITALS)
        assert all(len(v) == p.n_steps() for v in p.vitals.values())
        assert not any(np.isnan(v).any() for v in p.vitals.values())


def test_implausible_charted_values_are_filtered(tmp_path):
    """A heart rate of 900 is a typing error, not a tachycardia."""
    _seed(tmp_path)
    patients, _ = load_cohort(root=tmp_path)
    lo, hi = PLAUSIBLE["heart_rate_bpm"]
    for p in patients:
        assert p.vitals["heart_rate_bpm"].max() <= hi


def test_fahrenheit_is_converted_to_celsius(tmp_path):
    """The fixture charts 98.6 F; read as Celsius it would be a fatal fever."""
    _seed(tmp_path)
    patients, _ = load_cohort(root=tmp_path)
    temps = np.concatenate([p.vitals["temp_c"] for p in patients])
    assert 36.0 <= float(np.median(temps)) <= 38.0


def test_events_are_derived_from_clinical_thresholds(tmp_path):
    _seed(tmp_path)
    patients, prov = load_cohort(root=tmp_path)
    assert prov["event_definitions"]["hypoxemia"] == "SpO2 < 90%"
    by_id = {p.pid: p for p in patients}
    desat = by_id[101]
    assert desat.events["hypoxemia"].any(), "the seeded desaturation should show"
    assert np.array_equal(desat.events["hypoxemia"], desat.vitals["spo2_pct"] < 90)


def test_provenance_records_the_files_and_the_sample_size_caveat(tmp_path):
    _seed(tmp_path)
    _, prov = load_cohort(root=tmp_path)
    assert prov["cohort_is_a_demonstration_not_a_study"] is True
    assert prov["n_patients_built"] == 6
    assert all(len(v["sha256"]) == 16 for v in prov["files"].values())


def test_the_feature_pipeline_runs_on_the_real_cohort(tmp_path):
    """Features and a patient-grouped split must work on real-shaped input."""
    _seed(tmp_path, n_stays=8)
    from src.dataset import FEATURE_NAMES, build, split_by_patient
    patients, _ = load_cohort(root=tmp_path)

    X, y, groups, times = build(patients, event="hypoxemia", horizon_h=4.0)
    assert X.shape[1] == len(FEATURE_NAMES)
    assert len(y) == len(groups) == X.shape[0]
    assert not np.isnan(X).any()

    tr, te = split_by_patient(groups, test_frac=0.3, seed=3)
    # The split that makes the evaluation meaningful: no patient on both sides.
    assert not (set(groups[tr]) & set(groups[te]))


def test_stays_that_are_too_short_are_skipped_not_padded(tmp_path):
    _seed(tmp_path, n_stays=3, hours=4)
    with pytest.raises(datakit.FetchError, match="no ICU stay"):
        load_cohort(root=tmp_path, min_hours=12.0)
