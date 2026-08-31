"""Score the corpus, evaluate, and rebuild the site."""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import datetime, timezone

from .corpus import CORPUS, stats
from .evaluate import ablation, evaluate, manner_vs_topic
from .lexicon import CATEGORIES, LEXICON, validate
from .scoring import explain, features

ROOT = pathlib.Path(__file__).resolve().parent.parent


def run() -> dict:
    problems = validate()
    ev = evaluate()
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_synthetic": True,
        "data_source": "16 authored bilingual transcripts with analyst empathy ratings",
        "corpus": stats(),
        "lexicon": {
            "categories": list(CATEGORIES),
            "n_cues": sum(len(v) for v in LEXICON.values()),
            "per_category": {k: len(v) for k, v in LEXICON.items()},
            "validation_problems": problems,
        },
        "evaluation": ev,
        "manner_vs_topic": manner_vs_topic(),
        "ablation": ablation(),
        "worked_example": {
            "tid": CORPUS[0].tid,
            "rating": CORPUS[0].rating,
            "cues": explain(CORPUS[0])[:10],
            "lexicon_score": round(features(CORPUS[0])["lexicon_score"], 3),
        },
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "latest.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf8")
    return results


def main() -> int:
    r = run()
    c, ev = r["corpus"], r["evaluation"]
    print(f"corpus: {c['n_transcripts']} transcripts "
          f"({c['n_english']} EN / {c['n_mandarin']} ZH), {c['n_turns']} turns")
    print(f"ratings {c['rating_min']}-{c['rating_max']}, "
          f"mean {c['rating_mean']} sd {c['rating_sd']}")
    print(f"lexicon: {r['lexicon']['n_cues']} cues across "
          f"{len(r['lexicon']['categories'])} categories, "
          f"{len(r['lexicon']['validation_problems'])} structural problems")
    print(f"\nagreement with analyst ratings (n={ev['n']}, leave-one-out):")
    for k in ("lexicon_only", "model_loo"):
        b = ev[k]
        print(f"  {b['name']:<32} r={b['pearson_r']:+.3f} "
              f"rho={b['spearman_rho']:+.3f}  MAE={b['mae']:.3f}")
    for b in ev["by_language"]:
        print(f"  {b['name']:<32} r={b['pearson_r']:+.3f} n={b['n']}")
    m = r["manner_vs_topic"]
    print(f"\nmanner vs topic ({'/'.join(m['pair'])}): "
          f"ratings {m['ratings']}, lexicon {m['lexicon_scores']}, "
          f"separated={m['separated']} (gap {m['gap']:+.2f})")
    print("\nablation:")
    for g in r["ablation"]["groups"]:
        print(f"  {g['features']:<20} ({g['n_features']:>2} feats) "
              f"r={g['pearson_r']:+.3f}  MAE={g['mae']:.3f}")
    try:
        from .site import build_site
        build_site(r)
        print("\nwebsite/ rebuilt from this run")
    except Exception as exc:
        print(f"\n(site not rebuilt: {exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
