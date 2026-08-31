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


def run_real() -> dict:
    """Run the instrument over real consultations, and report only what can be.

    The distinction this run turns on: an instrument can be RELIABLE without
    being VALID. Reliability is whether it produces stable, discriminating
    scores on real speech, which these transcripts can show. Validity is
    whether those scores track what clinicians judge as empathy, which they
    cannot show, because MTS-Dialog carries no such ratings.

    So the correlation figures from the authored corpus are not repeated here
    with real-sounding numbers. What is reported is coverage, distribution and
    discrimination -- and the fact that validity remains unmeasured.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    import statistics as _stats
    from data.load import load_transcripts

    transcripts, prov = load_transcripts(root=ROOT / "data")

    scores, cue_hits, zero_cue = [], {}, 0
    doctor_shares, question_rates = [], []
    for tr in transcripts:
        f = features(tr)
        scores.append(f["lexicon_score"])
        doctor_shares.append(f["doctor_token_share"])
        question_rates.append(f["question_rate"])
        fired = [c for c, _, _ in explain(tr)]
        if not fired:
            zero_cue += 1
        for c in fired:
            cue_hits[c] = cue_hits.get(c, 0) + 1

    ranked = sorted(zip(scores, transcripts), key=lambda x: -x[0])
    quantiles = _stats.quantiles(scores, n=4) if len(scores) >= 4 else []

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_synthetic": False,
        "data_source": "MTS-Dialog, real doctor-patient consultation transcripts; "
                       "see data/MANIFEST.json for hashes and retrieval times",
        "validity_reported": False,
        "validity_withheld_because": prov["validity_withheld_because"],
        "provenance": {k: v for k, v in prov.items() if k != "files"},
        "files": prov["files"],
        "n_transcripts": len(transcripts),
        "score_distribution": {
            "min": round(min(scores), 4), "max": round(max(scores), 4),
            "median": round(_stats.median(scores), 4),
            "mean": round(_stats.fmean(scores), 4),
            "q1": round(quantiles[0], 4) if quantiles else None,
            "q3": round(quantiles[2], 4) if quantiles else None,
            "transcripts_with_no_cue": zero_cue,
            "share_with_no_cue": round(zero_cue / len(transcripts), 4),
        },
        "structure": {
            "median_doctor_token_share": round(_stats.median(doctor_shares), 4),
            "median_question_rate": round(_stats.median(question_rates), 4),
        },
        "cue_coverage": dict(sorted(cue_hits.items(), key=lambda kv: -kv[1])[:20]),
        "n_distinct_cues_fired": len(cue_hits),
        "highest_scoring": [{"tid": tr.tid, "score": round(s, 3),
                             "cues": [c for c, _, _ in explain(tr)][:6]}
                            for s, tr in ranked[:3]],
        "lowest_scoring": [{"tid": tr.tid, "score": round(s, 3)}
                           for s, tr in ranked[-3:]],
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "latest-real.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf8")
    return results


def main_real() -> int:
    from data.datakit import FetchError
    try:
        r = run_real()
    except FetchError as exc:
        print(f"cannot run on real data: {exc}", file=sys.stderr)
        return 2
    d, st = r["score_distribution"], r["structure"]
    print(f"source: {r['data_source']}")
    print(f"{r['n_transcripts']} real consultations scored")
    print(f"\nlexicon score: min {d['min']:.2f}  q1 {d['q1']}  "
          f"median {d['median']:.2f}  q3 {d['q3']}  max {d['max']:.2f}")
    print(f"transcripts where no cue fired: {d['transcripts_with_no_cue']} "
          f"({d['share_with_no_cue']:.1%})")
    print(f"distinct cues that fired at least once: {r['n_distinct_cues_fired']}")
    print(f"median doctor token share {st['median_doctor_token_share']:.3f}, "
          f"median questions per doctor turn {st['median_question_rate']:.3f}")
    print("\nmost frequently fired cues:")
    for cue, n in list(r["cue_coverage"].items())[:10]:
        print(f"  {cue:<28} {n}")
    print("\nhighest scoring:")
    for x in r["highest_scoring"]:
        print(f"  {x['tid']:<22} {x['score']:>7.2f}  {', '.join(x['cues'][:4])}")
    print("\nVALIDITY IS NOT REPORTED: " + r["validity_withheld_because"])
    print("wrote results/latest-real.json")
    return 0


def main() -> int:
    if "--real" in sys.argv[1:]:
        return main_real()
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
