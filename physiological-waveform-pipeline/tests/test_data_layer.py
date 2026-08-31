"""Tests for the real-recording path.

The interesting property here is that the real path can FAIL in a way the
synthetic path cannot. On synthetic segments the pipeline is checked against
artifacts the generator placed. On BIDMC it is checked against a bedside
monitor's own pulse rate -- an independent device with its own algorithm -- so
a disagreement is evidence about the pipeline, not about the fixture.
"""
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data import datakit
from data.load import (FS, load_records, read_numerics, read_signals,
                       reference_rate, to_segments)

# BIDMC headers carry a leading space and a bracketed unit.
SIG_HEADER = "Time [s], II, AVR, PLETH, RESP\n"
NUM_HEADER = "Time [s], HR, PULSE, RESP, SpO2\n"


def _signals_csv(n=125 * 30, hr_bpm=72.0, fs=FS):
    t = np.arange(n) / fs
    # A plausible pulse: sharp upstroke, dicrotic notch, slow decay.
    phase = 2 * np.pi * (hr_bpm / 60.0) * t
    pleth = (1.0 * np.sin(phase) + 0.35 * np.sin(2 * phase + 0.9)
             + 0.10 * np.sin(3 * phase) + 2.5)
    rows = [SIG_HEADER]
    for i in range(n):
        rows.append(f"{t[i]:.3f},{0.1:.4f},{0.2:.4f},"
                    f"{pleth[i]:.5f},{0.3:.4f}\n")
    return "".join(rows).encode()


def _numerics_csv(seconds=30, hr=71.0, pulse=72.0):
    rows = [NUM_HEADER]
    for s in range(seconds):
        # The monitor writes '-' when a value is momentarily unavailable.
        p = "-" if s == 3 else f"{pulse:.1f}"
        rows.append(f"{s},{hr:.1f},{p},18.0,98.0\n")
    return "".join(rows).encode()


def test_read_signals_selects_the_named_channel_despite_header_padding():
    sig = read_signals(_signals_csv(), "pleth")
    assert len(sig) == 125 * 30
    # PLETH here is centred near 2.5; picking the wrong column would give ~0.1.
    assert 2.0 < float(np.mean(sig)) < 3.0


def test_read_signals_rejects_an_unknown_channel():
    with pytest.raises(ValueError, match="not in"):
        read_signals(_signals_csv(), "abp")


def test_read_signals_refuses_a_recording_shorter_than_one_window():
    with pytest.raises(ValueError, match="at least 10 seconds"):
        read_signals(_signals_csv(n=100), "pleth")


def test_read_numerics_turns_the_monitor_dash_into_nan_not_zero():
    """A '-' read as 0 would drag the reference rate down and look like a bug
    in beat detection."""
    num = read_numerics(_numerics_csv())
    pulse = [v for k, v in num.items() if k.startswith("pulse")][0]
    assert np.isnan(pulse[3])
    assert np.nanmedian(pulse) == pytest.approx(72.0)


def test_reference_rate_prefers_pulse_over_ecg_heart_rate():
    """PULSE comes from the same plethysmogram the pipeline reads, so it is
    the like-for-like comparison; HR comes from the ECG."""
    arr, channel = reference_rate(read_numerics(_numerics_csv(hr=71.0, pulse=72.0)))
    assert channel.startswith("pulse")
    assert np.nanmedian(arr) == pytest.approx(72.0)


def test_reference_rate_falls_back_to_hr_when_pulse_is_absent():
    raw = ("Time [s], HR, RESP\n" + "".join(f"{s},70.0,18.0\n" for s in range(30))
           ).encode()
    arr, channel = reference_rate(read_numerics(raw))
    assert channel.startswith("hr")


# --- end to end ------------------------------------------------------------

def test_refuses_when_nothing_is_cached(tmp_path):
    with pytest.raises(datakit.FetchError, match="no real waveform recordings"):
        load_records(root=tmp_path)


def _seed(tmp_path, records=(("01", 72.0), ("05", 96.0), ("11", 54.0))):
    f = datakit.Fetcher(tmp_path)
    man = f.load_manifest()
    (f.raw / "bidmc" / "bidmc_csv").mkdir(parents=True, exist_ok=True)
    for rid, hr in records:
        for suffix, raw in (("Signals", _signals_csv(hr_bpm=hr)),
                            ("Numerics", _numerics_csv(hr=hr - 1, pulse=hr))):
            dest = f"bidmc/bidmc_csv/bidmc_{rid}_{suffix}.csv"
            (f.raw / dest).write_bytes(raw)
            man["files"][dest] = {
                "source": dest, "url": f"https://physionet.org/files/{dest}",
                "publisher": "PhysioNet", "terms": "open access",
                "sha256": datakit.sha256_file(f.raw / dest), "bytes": len(raw),
                "retrieved_utc": datakit.utc_now()}
    f._write_manifest(man)
    return f


def test_load_records_pairs_signals_with_their_numerics(tmp_path):
    _seed(tmp_path)
    records, prov = load_records(root=tmp_path)
    assert len(records) == 3
    assert all(r["reference_rate"] is not None for r in records)
    assert prov["n_records"] == 3
    assert all(p["has_monitor_reference"] for p in prov["records"]
               if p["status"] == "ok")


def test_provenance_states_both_limits_of_this_dataset(tmp_path):
    _seed(tmp_path)
    _, prov = load_records(root=tmp_path)
    assert prov["artifact_locations_annotated"] is False
    assert "no answer key" in prov["artifact_scoring_withheld_because"]
    assert prov["signal_is_ppg_not_arterial_line"] is True


def test_a_recording_without_numerics_still_loads(tmp_path):
    f = _seed(tmp_path, records=(("01", 72.0),))
    (f.raw / "bidmc/bidmc_csv/bidmc_01_Numerics.csv").unlink()
    records, prov = load_records(root=tmp_path)
    assert len(records) == 1
    assert records[0]["reference_rate"] is None


def test_beat_detection_agrees_with_the_monitor_on_real_shaped_signals(tmp_path):
    """The external check: derived rate against an independently reported one."""
    _seed(tmp_path)
    from src.pipeline import process_segment
    from src.quality import PPG_LIMITS
    records, _ = load_records(root=tmp_path)
    segments = to_segments(records)

    for rec, seg in zip(records, segments):
        rows, quals = process_segment(seg.signal, seg.fs, limits=PPG_LIMITS)
        assert quals, "windowing produced nothing"
        hrs = [r["heart_rate_bpm"] for r in rows
               if r["heart_rate_bpm"] == r["heart_rate_bpm"]]
        assert hrs, f"no beats detected in record {rec['record']}"
        derived = float(np.median(hrs))
        ref = float(np.nanmedian(rec["reference_rate"]))
        # Counting the dicrotic notch as a beat would put this near 2x.
        assert abs(derived - ref) < 5.0, (rec["record"], derived, ref)


def test_segments_carry_no_fabricated_artifact_labels(tmp_path):
    """Real recordings have no annotations, and must not acquire empty ones
    that could be mistaken for 'no artifacts present'."""
    _seed(tmp_path)
    records, _ = load_records(root=tmp_path)
    segments = to_segments(records)
    assert all(s.artifacts == [] for s in segments)
    assert all(s.label.startswith("bidmc_") for s in segments)


def test_arterial_limits_reject_a_plethysmogram_entirely(tmp_path):
    """The bug real data exposed, kept as a regression test.

    The quality gate's plausibility band was written in mmHg for an arterial
    line. A photoplethysmogram is recorded in arbitrary units near zero, so
    every sample fell outside 20-220 and the gate discarded whole recordings --
    reporting a signal-quality failure for a signal that was fine, because the
    units were an assumption nobody had written down.
    """
    _seed(tmp_path, records=(("01", 72.0),))
    from src.pipeline import process_segment
    from src.quality import ABP_LIMITS, PPG_LIMITS
    records, _ = load_records(root=tmp_path)
    sig, fs = records[0]["signal"], records[0]["fs"]

    _, with_abp = process_segment(sig, fs, limits=ABP_LIMITS)
    _, with_ppg = process_segment(sig, fs, limits=PPG_LIMITS)

    assert not any(q.accepted for q in with_abp), \
        "arterial limits should reject this plethysmogram"
    assert "out_of_range" in with_abp[0].codes()
    assert any(q.accepted for q in with_ppg), \
        "the same signal must pass under limits appropriate to its channel"


def test_ppg_step_limit_scales_with_the_windows_own_amplitude(tmp_path):
    """PLETH has no fixed amplitude scale, so a fixed step limit in mmHg would
    pass one recording and reject another identical one recorded at a
    different gain."""
    from src.quality import PPG_LIMITS
    small = np.array([0.0, 0.1, 0.2, 0.1, 0.0])
    large = small * 1000.0
    assert PPG_LIMITS.step_limit(large) == pytest.approx(
        1000.0 * PPG_LIMITS.step_limit(small))
