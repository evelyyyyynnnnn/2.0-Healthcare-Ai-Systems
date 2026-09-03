"""Tests for parsing real consultation transcripts.

The parsing decisions here change what the instrument measures. A continuation
line dropped instead of attached removes the longest utterances -- exactly the
ones carrying the language this project scores -- and the result would be a
score distribution that looks plausible and is built on truncated speech.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from data import datakit
from data.load import load_transcripts, parse_dialogue

DIALOGUE = """Doctor: What brings you in today?
Patient: My knee has been hurting for about three weeks now.
It gets worse when I climb stairs, and it woke me up twice last week.
Doctor: That sounds difficult. Does anything make it better?
Patient: Ice helps a little.
Guest_family: She has been limping since the fall."""


def test_parse_dialogue_splits_on_speaker_prefixes():
    turns = parse_dialogue(DIALOGUE)
    assert turns[0] == ("doctor", "What brings you in today?")
    assert turns[2][0] == "doctor"


def test_continuation_lines_attach_to_the_speaker_still_talking():
    """Dropping them would delete the longest utterances in the corpus."""
    turns = parse_dialogue(DIALOGUE)
    patient_first = turns[1][1]
    assert "three weeks" in patient_first
    assert "climb stairs" in patient_first, \
        "the unprefixed continuation line was dropped"
    assert "woke me up twice" in patient_first


def test_family_and_clinician_guests_are_assigned_a_side():
    turns = parse_dialogue(DIALOGUE)
    assert turns[-1][0] == "patient", "a family member speaks on the patient side"
    assert parse_dialogue("Guest_clinician: I reviewed the scan.")[0][0] == "doctor"


def test_nurse_counts_as_clinician():
    assert parse_dialogue("Nurse: I'll take your blood pressure.")[0][0] == "doctor"


def test_empty_and_prefixless_text_yields_nothing():
    assert parse_dialogue("") == []
    assert parse_dialogue("no speaker prefix anywhere here") == []


def test_parse_is_case_insensitive_and_tolerates_spacing():
    assert parse_dialogue("DOCTOR:  Hello there.")[0] == ("doctor", "Hello there.")


# --- end to end ------------------------------------------------------------

def test_refuses_when_nothing_is_cached(tmp_path):
    with pytest.raises(datakit.FetchError, match="no real consultation"):
        load_transcripts(root=tmp_path)


def _seed(tmp_path, n=6):
    import csv
    import io
    f = datakit.Fetcher(tmp_path)
    man = f.load_manifest()
    (f.raw / "mts-dialog").mkdir(parents=True, exist_ok=True)

    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["ID", "section_header", "section_text", "dialogue"])
    for i in range(n):
        w.writerow([i, "CC", "Knee pain.", DIALOGUE])
    # A row too short to be a consultation, and one that will not parse.
    w.writerow([n, "CC", "x", "Doctor: Hello."])
    w.writerow([n + 1, "CC", "x", "no speaker prefixes at all"])
    raw = buf.getvalue().encode()

    dest = "mts-dialog/MTS-Dialog-TrainingSet.csv"
    (f.raw / dest).write_bytes(raw)
    man["files"][dest] = {
        "source": dest, "url": "https://raw.githubusercontent.com/x/y/z.csv",
        "publisher": "MTS-Dialog", "terms": "public research dataset",
        "sha256": datakit.sha256_file(f.raw / dest), "bytes": len(raw),
        "retrieved_utc": datakit.utc_now()}
    f._write_manifest(man)
    return f


def test_transcripts_carry_no_rating_and_say_why(tmp_path):
    """A rating of 0.0 would be scored as 'least empathic' rather than
    'unknown', which is the difference between missing data and a low score."""
    _seed(tmp_path)
    transcripts, prov = load_transcripts(root=tmp_path)
    assert transcripts
    assert all(t.rating is None for t in transcripts)
    assert prov["empathy_ratings_available"] is False
    assert "no such ratings exist" in prov["validity_withheld_because"]


def test_short_and_unparseable_rows_are_counted_not_silently_dropped(tmp_path):
    _seed(tmp_path, n=6)
    transcripts, prov = load_transcripts(root=tmp_path)
    assert len(transcripts) == 6
    assert prov["skipped"]["too_short"] == 1
    assert prov["skipped"]["unparsed"] == 1


def test_provenance_records_the_file_hash(tmp_path):
    _seed(tmp_path)
    _, prov = load_transcripts(root=tmp_path)
    ok = [f for f in prov["files"] if f["status"] == "ok"]
    assert ok and len(ok[0]["sha256"]) == 16


def test_the_scorer_runs_on_real_transcripts(tmp_path):
    _seed(tmp_path)
    from src.scoring import explain, features
    transcripts, _ = load_transcripts(root=tmp_path)
    tr = transcripts[0]

    f = features(tr)
    assert f["doctor_turns"] >= 1 and f["patient_turns"] >= 1
    assert 0.0 <= f["doctor_token_share"] <= 1.0
    assert f["lexicon_score"] >= 0.0
    # "That sounds difficult" is an acknowledgement cue and should fire.
    assert explain(tr), "no cue fired on a transcript containing one"


def test_limit_caps_the_number_parsed(tmp_path):
    _seed(tmp_path, n=20)
    transcripts, _ = load_transcripts(root=tmp_path, limit=5)
    assert len(transcripts) == 5
