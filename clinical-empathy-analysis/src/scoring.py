"""Turning a transcript into features and an empathy score."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .lexicon import CATEGORIES, LEXICON, POSITIVE_CATEGORIES, all_cues

_CJK = re.compile(r"[一-鿿]")


@dataclass
class Turn:
    speaker: str          # "doctor" or "patient"
    text: str


@dataclass
class Transcript:
    tid: str
    turns: list
    language: str = "en"
    rating: float | None = None       # clinician-assigned empathy rating, 0-4
    note: str = ""

    def doctor_text(self) -> str:
        return " ".join(t.text for t in self.turns if t.speaker == "doctor")

    def patient_text(self) -> str:
        return " ".join(t.text for t in self.turns if t.speaker == "patient")


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def count_cues(text: str) -> dict:
    """Cue hits per category, with a per-category weighted total."""
    t = _norm(text)
    hits = {c: [] for c in CATEGORIES}
    for cue, (cat, w) in all_cues().items():
        n = t.count(cue)
        if n:
            hits[cat].append((cue, n, w))
    return hits


def tokens(text: str) -> list:
    """Word tokens for Latin script, characters for CJK.

    Length normalisation matters here: a long consultation accumulates cue hits
    simply by being long, so an unnormalised count measures duration.
    """
    t = _norm(text)
    if _CJK.search(t):
        return _CJK.findall(t) + re.findall(r"[a-z]+", t)
    return re.findall(r"[a-z']+", t)


def features(tr: Transcript) -> dict:
    doc = tr.doctor_text()
    pat = tr.patient_text()
    hits = count_cues(doc)
    n_tok = max(1, len(tokens(doc)))

    f: dict = {}
    for cat in CATEGORIES:
        raw = sum(n * w for _, n, w in hits[cat])
        f[f"{cat}_weighted"] = raw
        f[f"{cat}_per_100tok"] = 100.0 * raw / n_tok
    f["lexicon_score"] = sum(f[f"{c}_per_100tok"] for c in CATEGORIES)

    doc_turns = [t for t in tr.turns if t.speaker == "doctor"]
    pat_turns = [t for t in tr.turns if t.speaker == "patient"]
    doc_tok = len(tokens(doc))
    pat_tok = len(tokens(pat))

    # Structural features. Empathic consultations differ in shape, not only in
    # vocabulary: the doctor talks proportionally less and asks more.
    f["doctor_turns"] = len(doc_turns)
    f["patient_turns"] = len(pat_turns)
    f["doctor_token_share"] = doc_tok / max(1, doc_tok + pat_tok)
    f["mean_doctor_turn_len"] = doc_tok / max(1, len(doc_turns))
    f["question_rate"] = doc.count("?") / max(1, len(doc_turns))
    f["open_question_rate"] = sum(
        doc.lower().count(w) for w in ("how ", "what ", "tell me", "怎么", "什么")
    ) / max(1, len(doc_turns))
    f["patient_token_share"] = pat_tok / max(1, doc_tok + pat_tok)
    return f


FEATURE_NAMES = tuple(features(Transcript("x", [Turn("doctor", "hi"),
                                                Turn("patient", "hello")])).keys())


def lexicon_score(tr: Transcript) -> float:
    return features(tr)["lexicon_score"]


def explain(tr: Transcript) -> list:
    """Which cues fired, so a score can be argued with."""
    out = []
    for cat, items in count_cues(tr.doctor_text()).items():
        for cue, n, w in items:
            out.append({"category": cat, "cue": cue, "count": n,
                        "weight": w, "contribution": round(n * w, 3)})
    return sorted(out, key=lambda r: -abs(r["contribution"]))
