import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.synth import make_segment, make_cohort, FS_DEFAULT
from src.quality import (assess_window, flat_fraction, out_of_range_fraction,
                         pulsatility_sqi, spike_count, window_signal)
from src.features import extract, find_beats
from src.pipeline import build_dataset, score_rejection, reject_reason_histogram
from src.loaders import describe_sources

FS = FS_DEFAULT


def clean_window(hr=75.0):
    return make_segment(duration_s=10.0, fs=FS, hr_bpm=hr,
                        rng=np.random.default_rng(1)).signal


# --- the detector must not fire on good signal ---------------------------

def test_spike_detector_ignores_systolic_upstrokes():
    """Regression, and the bug that broke two earlier versions.

    An arterial upstroke is the largest sample-to-sample change in a clean
    window. A detector keyed on magnitude alone rejected 99% of good data.
    """
    for hr in (50.0, 75.0, 110.0, 140.0):
        assert spike_count(clean_window(hr)) == 0, f"false spikes at {hr} bpm"


def test_spike_detector_catches_an_injected_transient():
    x = clean_window().copy()
    x[600] += 90.0
    assert spike_count(x) >= 1


def test_clean_window_is_accepted():
    q = assess_window(clean_window(), FS, 0)
    assert q.accepted, q.reasons


def test_flatline_is_caught():
    x = clean_window().copy()
    x[200:900] = x[200]
    q = assess_window(x, FS, 0)
    assert not q.accepted
    assert "flatline" in q.codes()


def test_out_of_range_is_caught():
    x = clean_window().copy()
    x[100:400] = 400.0
    q = assess_window(x, FS, 0)
    assert not q.accepted
    assert "out_of_range" in q.codes()


def test_flat_fraction_is_one_for_a_constant_trace():
    assert flat_fraction(np.full(100, 80.0)) == 1.0


def test_pulsatility_sqi_is_higher_for_a_pulse_than_for_noise():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 10, int(FS * 10))
    assert pulsatility_sqi(clean_window(), FS) > pulsatility_sqi(noise, FS)


# --- features ------------------------------------------------------------

def test_beat_detection_recovers_the_heart_rate():
    for hr in (55.0, 75.0, 100.0):
        seg = make_segment(duration_s=30.0, fs=FS, hr_bpm=hr,
                           rng=np.random.default_rng(3))
        peaks = find_beats(seg.signal, FS)
        est = 60.0 / np.mean(np.diff(peaks) / FS)
        assert abs(est - hr) / hr < 0.10, f"{hr} bpm -> {est:.1f}"


def test_extracted_pressures_are_ordered():
    f = extract(clean_window(), FS)
    assert f["dbp_mmhg"] < f["map_mmhg"] < f["sbp_mmhg"]
    assert f["pulse_pressure"] > 0


def test_extract_returns_every_declared_feature():
    from src.features import FEATURE_NAMES
    assert set(extract(clean_window(), FS)) == set(FEATURE_NAMES)


# --- pipeline ------------------------------------------------------------

def test_windowing_is_non_overlapping_and_complete():
    sig = np.zeros(int(FS * 60))
    wins = list(window_signal(sig, FS, 10.0, 10.0))
    assert len(wins) == 6
    assert all(len(w) == int(FS * 10) for _, w in wins)


def test_dataset_reports_waveform_hours(tmp_path):
    cohort = make_cohort(n_segments=6, duration_s=60.0, fs=FS, artifact_rate=0.0, seed=1)
    ds = build_dataset(cohort, FS, out_path=tmp_path / "d.csv")
    assert ds["waveform_hours"] == pytest.approx(6 * 60 / 3600, rel=1e-6)
    assert (tmp_path / "d.csv").exists()


def test_clean_cohort_is_almost_entirely_accepted():
    cohort = make_cohort(n_segments=20, duration_s=60.0, fs=FS,
                         artifact_rate=0.0, seed=2)
    ds = build_dataset(cohort, FS)
    assert ds["acceptance_rate"] > 0.95, (
        f"rejecting clean data at {1 - ds['acceptance_rate']:.1%}")


def test_rejection_beats_chance_on_a_contaminated_cohort():
    cohort = make_cohort(n_segments=40, duration_s=120.0, fs=FS,
                         artifact_rate=0.5, seed=5)
    s = score_rejection(cohort, FS)
    assert s["recall"] > 0.7, f"missing artifacts: recall {s['recall']}"
    assert s["precision"] > 0.5, f"over-rejecting: precision {s['precision']}"
    assert s["specificity"] > 0.9


def test_narrow_artifacts_count_as_contamination():
    """A two-sample spike is 0.16% of a window but still ruins beat detection.

    Scoring it with the same fractional-overlap rule used for flatlines made a
    working detector read as precision 0.07.
    """
    from src.pipeline import _is_contaminated
    from src.synth import SynthSegment
    seg = SynthSegment(signal=np.zeros(1250), fs=FS,
                       artifacts=[(600, 602, "spike")])
    assert _is_contaminated(seg, 0, 1250, 0.25)
    seg2 = SynthSegment(signal=np.zeros(1250), fs=FS,
                        artifacts=[(600, 610, "motion")])
    assert not _is_contaminated(seg2, 0, 1250, 0.25)


def test_reason_histogram_uses_stable_codes():
    cohort = make_cohort(n_segments=10, duration_s=60.0, fs=FS,
                         artifact_rate=1.0, seed=4)
    hist = reject_reason_histogram(cohort, FS)
    assert hist, "no rule fired on a fully contaminated cohort"
    assert all(" " not in k for k in hist), f"unstable keys: {list(hist)}"


def test_real_sources_are_declared_unbundled():
    """Guards the honesty of the site: no real cohort ships here."""
    for s in describe_sources():
        if s["name"] != "Synthetic cohort":
            assert s["bundled"] is False
