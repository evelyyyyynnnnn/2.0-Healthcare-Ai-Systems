"""Parse MTS-Dialog consultations into Transcript objects.

The dialogue column is plain text with speaker prefixes:

    Doctor: What brings you in today?
    Patient: My knee has been hurting.

Speakers appear as Doctor/Patient, sometimes Guest_family or Guest_clinician,
and a turn can run over several lines. Everything that is not clearly the
clinician is treated as the patient side, and turns that carry no speaker
prefix are attached to the previous speaker rather than dropped -- dropping
them would quietly delete the longest utterances, which are exactly the ones
carrying the language this project measures.
"""
from __future__ import annotations

import csv
import io
import pathlib
import re

from .datakit import Fetcher, FetchError

ROOT = pathlib.Path(__file__).resolve().parent

SPEAKER = re.compile(r"^\s*(doctor|physician|patient|guest[_ ]?family|"
                     r"guest[_ ]?clinician|nurse|caregiver)\s*[:：]\s*",
                     re.IGNORECASE)
CLINICIAN = {"doctor", "physician", "nurse", "guest_clinician", "guestclinician"}


def _role(raw: str) -> str:
    key = raw.strip().lower().replace(" ", "_")
    return "doctor" if key in CLINICIAN else "patient"


def parse_dialogue(text: str) -> list:
    """Return [(speaker, utterance)] from one MTS-Dialog dialogue cell."""
    turns = []
    for line in (text or "").replace("\r", "\n").split("\n"):
        if not line.strip():
            continue
        m = SPEAKER.match(line)
        if m:
            turns.append([_role(m.group(1)), line[m.end():].strip()])
        elif turns:
            # A continuation line belongs to the speaker who is still talking.
            turns[-1][1] = (turns[-1][1] + " " + line.strip()).strip()
    return [(s, t) for s, t in turns if t]


def load_transcripts(root=ROOT, min_turns: int = 4, limit: int = 0):
    """Return (transcripts, provenance). Refuses when nothing real is cached."""
    from src.scoring import Transcript, Turn

    f = Fetcher(root)
    man = f.load_manifest()
    files = sorted(k for k in man["files"] if k.startswith("mts-dialog/"))
    if not files:
        raise FetchError(
            "no real consultation transcripts cached. Run `python -m data.fetch` "
            "in a networked environment first; this project will not present "
            "authored dialogue as recorded clinical speech.")

    transcripts, prov, skipped = [], [], {"too_short": 0, "unparsed": 0}
    for dest in files:
        raw = (f.raw / dest).read_bytes().decode("utf-8", errors="replace")
        rows = list(csv.DictReader(io.StringIO(raw)))
        col = next((c for c in (rows[0] if rows else {}) if "dialog" in c.lower()),
                   None)
        if col is None:
            prov.append({"file": dest, "status": "no dialogue column"})
            continue
        n_ok = 0
        for i, r in enumerate(rows):
            pairs = parse_dialogue(r.get(col, ""))
            if not pairs:
                skipped["unparsed"] += 1
                continue
            if len(pairs) < min_turns:
                skipped["too_short"] += 1
                continue
            if not any(s == "doctor" for s, _ in pairs):
                skipped["unparsed"] += 1
                continue
            transcripts.append(Transcript(
                tid=f"MTS-{pathlib.Path(dest).stem[-5:]}-{r.get('ID', i)}",
                language="en",
                rating=None,          # MTS-Dialog carries no empathy rating
                turns=[Turn(s, t) for s, t in pairs],
                note="real consultation excerpt from MTS-Dialog; no "
                     "clinician-assigned empathy rating exists for it",
            ))
            n_ok += 1
            if limit and len(transcripts) >= limit:
                break
        rec = man["files"][dest]
        prov.append({"file": dest, "status": "ok", "rows": len(rows),
                     "transcripts": n_ok, "sha256": rec["sha256"][:16],
                     "url": rec["url"], "retrieved_utc": rec.get("retrieved_utc")})
        if limit and len(transcripts) >= limit:
            break

    if not transcripts:
        raise FetchError("no usable consultation transcripts parsed from the cache")

    return transcripts, {
        "dataset": "MTS-Dialog (public research dataset)",
        "n_transcripts": len(transcripts),
        "skipped": skipped,
        "empathy_ratings_available": False,
        "validity_withheld_because":
            "MTS-Dialog is annotated for clinical note sections, not for "
            "communication quality. Correlation with clinician-assigned empathy "
            "ratings cannot be computed because no such ratings exist for these "
            "conversations.",
        "files": prov,
    }
