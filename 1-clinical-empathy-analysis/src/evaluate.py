"""Evaluation, with leave-one-out because the corpus is tiny.

Sixteen transcripts cannot support a train/test split, so performance is
estimated by leave-one-out cross-validation and reported with that stated
plainly. LOO on sixteen items has wide error bars, and the site says so rather
than quoting a correlation as though it were established.
"""

from __future__ import annotations

import numpy as np

from .corpus import CORPUS
from .scoring import FEATURE_NAMES, features


def design_matrix(transcripts=None):
    ts = transcripts if transcripts is not None else CORPUS
    X = np.array([[features(t)[k] for k in FEATURE_NAMES] for t in ts], float)
    y = np.array([t.rating for t in ts], float)
    lang = np.array([t.language for t in ts])
    return X, y, lang


def pearson(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def spearman(a, b) -> float:
    def rank(v):
        order = np.argsort(np.asarray(v, float), kind="mergesort")
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        return r
    return pearson(rank(a), rank(b))


def loo_predictions(X, y, ridge: float = 1.0):
    """Leave-one-out ridge regression, fitted from scratch each fold."""
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        Xt, yt = X[m], y[m]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-9
        Z = (Xt - mu) / sd
        A = Z.T @ Z + ridge * np.eye(Z.shape[1])
        w = np.linalg.solve(A, Z.T @ (yt - yt.mean()))
        preds[i] = ((X[i] - mu) / sd) @ w + yt.mean()
    return preds


def evaluate(ridge: float = 1.0) -> dict:
    X, y, lang = design_matrix()
    lex = X[:, FEATURE_NAMES.index("lexicon_score")]
    preds = loo_predictions(X, y, ridge)

    def block(name, pred, mask=None):
        m = np.ones(len(y), bool) if mask is None else mask
        return {
            "name": name, "n": int(m.sum()),
            "pearson_r": round(pearson(pred[m], y[m]), 4),
            "spearman_rho": round(spearman(pred[m], y[m]), 4),
            "mae": round(float(np.mean(np.abs(pred[m] - y[m]))), 4),
        }

    return {
        "n": int(len(y)),
        "ridge": ridge,
        "lexicon_only": block("lexicon score alone", lex),
        "model_loo": block("ridge over all features (LOO)", preds),
        "by_language": [
            block("English", preds, lang == "en"),
            block("Mandarin", preds, lang == "zh"),
        ],
        "predictions": [
            {"tid": t.tid, "language": t.language, "rating": t.rating,
             "lexicon_score": round(float(lex[i]), 3),
             "predicted": round(float(preds[i]), 3),
             "error": round(float(preds[i] - y[i]), 3)}
            for i, t in enumerate(CORPUS)
        ],
    }


def manner_vs_topic() -> dict:
    """EN-09 and EN-10 share clinical content and differ only in manner.

    A measure that scores them the same is tracking topic, not empathy. This is
    the single most diagnostic pair in the corpus.
    """
    by_id = {t.tid: t for t in CORPUS}
    a, b = by_id["EN-09"], by_id["EN-10"]
    fa, fb = features(a), features(b)
    return {
        "pair": ["EN-09", "EN-10"],
        "ratings": [a.rating, b.rating],
        "lexicon_scores": [round(fa["lexicon_score"], 3),
                           round(fb["lexicon_score"], 3)],
        "separated": fb["lexicon_score"] > fa["lexicon_score"],
        "gap": round(fb["lexicon_score"] - fa["lexicon_score"], 3),
    }


def ablation() -> dict:
    """Which feature families carry the signal?"""
    X, y, _ = design_matrix()
    groups = {
        "all features": list(range(len(FEATURE_NAMES))),
        "lexicon only": [i for i, n in enumerate(FEATURE_NAMES)
                         if n.endswith("_per_100tok") or n == "lexicon_score"],
        "structure only": [i for i, n in enumerate(FEATURE_NAMES)
                           if not n.endswith("_per_100tok")
                           and not n.endswith("_weighted") and n != "lexicon_score"],
        "no dismissal cues": [i for i, n in enumerate(FEATURE_NAMES)
                              if not n.startswith("dismissal")],
    }
    out = []
    for name, idx in groups.items():
        p = loo_predictions(X[:, idx], y)
        out.append({"features": name, "n_features": len(idx),
                    "pearson_r": round(pearson(p, y), 4),
                    "mae": round(float(np.mean(np.abs(p - y))), 4)})
    return {"groups": out}
