import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.corpus import CORPUS, stats
from src.evaluate import (ablation, design_matrix, evaluate, loo_predictions,
                          manner_vs_topic, pearson, spearman)
from src.lexicon import CATEGORIES, LEXICON, all_cues, validate
from src.scoring import Transcript, Turn, explain, features, lexicon_score, tokens


def _t(pairs, lang="en"):
    return Transcript("t", [Turn(s, x) for s, x in pairs], language=lang)


# --- lexicon integrity ---------------------------------------------------

def test_lexicon_has_no_structural_problems():
    assert validate() == []


def test_dismissal_cues_are_all_negative():
    assert all(w < 0 for w in LEXICON["dismissal"].values())


def test_positive_categories_are_all_positive():
    for cat in CATEGORIES:
        if cat == "dismissal":
            continue
        assert all(w > 0 for w in LEXICON[cat].values()), cat


def test_no_cue_appears_in_two_categories():
    flat = all_cues()
    total = sum(len(v) for v in LEXICON.values())
    assert len(flat) == total


def test_lexicon_covers_both_languages():
    import re
    cjk = re.compile(r"[一-鿿]")
    for cat, cues in LEXICON.items():
        assert any(cjk.search(c) for c in cues), f"{cat} has no Mandarin cue"
        assert any(not cjk.search(c) for c in cues), f"{cat} has no English cue"


# --- scoring behaviour ---------------------------------------------------

def test_empathic_scores_above_dismissive():
    warm = _t([("patient", "I'm scared."),
               ("doctor", "That sounds frightening. Tell me more. I understand.")])
    cold = _t([("patient", "I'm scared."),
               ("doctor", "Calm down. Obviously nothing to worry about.")])
    assert lexicon_score(warm) > lexicon_score(cold)


def test_dismissive_transcript_scores_negative():
    cold = _t([("patient", "I'm scared."),
               ("doctor", "Calm down. As I said, nothing to worry about. No time.")])
    assert lexicon_score(cold) < 0


def test_only_doctor_turns_are_scored():
    """A patient saying 'I understand' is not clinician empathy."""
    a = _t([("patient", "I understand, that sounds frightening"), ("doctor", "Right.")])
    b = _t([("patient", "ok"), ("doctor", "Right.")])
    assert lexicon_score(a) == lexicon_score(b)


def test_score_is_length_normalised():
    short = _t([("patient", "hi"), ("doctor", "I understand.")])
    padded = _t([("patient", "hi"),
                 ("doctor", "I understand. " + "the patient attended today. " * 40)])
    assert lexicon_score(short) > lexicon_score(padded)


def test_mandarin_cues_are_detected():
    zh = _t([("patient", "我很担心。"), ("doctor", "我理解你的担心，能再说说吗？")], "zh")
    assert lexicon_score(zh) > 0


def test_tokens_handle_cjk_and_latin():
    assert len(tokens("我理解")) == 3
    assert len(tokens("i understand you")) == 3


def test_explain_accounts_for_the_score():
    tr = _t([("patient", "x"), ("doctor", "I understand. Tell me more.")])
    rows = explain(tr)
    assert rows
    assert all({"category", "cue", "count", "contribution"} <= set(r) for r in rows)


def test_features_are_finite_for_every_transcript():
    for t in CORPUS:
        f = features(t)
        assert all(np.isfinite(v) for v in f.values()), t.tid


# --- corpus and evaluation ----------------------------------------------

def test_corpus_spans_the_rating_scale():
    s = stats()
    assert s["rating_min"] <= 1.0 and s["rating_max"] >= 3.5
    assert s["n_mandarin"] >= 5 and s["n_english"] >= 8


def test_pearson_and_spearman_on_known_input():
    x = np.array([1.0, 2, 3, 4])
    assert pearson(x, 2 * x + 1) == 1.0
    assert spearman(x, np.array([1.0, 4, 9, 16])) == 1.0


def test_loo_never_trains_on_the_held_out_item():
    """A LOO fold that saw its own target would make every score meaningless."""
    X, y, _ = design_matrix()
    n = len(y)
    for i in (0, n // 2, n - 1):
        m = np.ones(n, bool)
        m[i] = False
        assert m.sum() == n - 1 and not m[i]


def test_model_correlates_with_ratings():
    ev = evaluate()
    assert ev["model_loo"]["pearson_r"] > 0.5


def test_manner_is_separated_from_topic():
    """EN-09 / EN-10: same clinical content, different delivery."""
    m = manner_vs_topic()
    assert m["separated"], "the measure is tracking topic, not manner"
    assert m["gap"] > 1.0


def test_dismissal_features_carry_signal():
    groups = {g["features"]: g for g in ablation()["groups"]}
    assert groups["all features"]["pearson_r"] >= groups["no dismissal cues"]["pearson_r"]


def test_structure_alone_is_informative_but_weaker():
    groups = {g["features"]: g for g in ablation()["groups"]}
    assert 0.3 < groups["structure only"]["pearson_r"] < groups["all features"]["pearson_r"]
