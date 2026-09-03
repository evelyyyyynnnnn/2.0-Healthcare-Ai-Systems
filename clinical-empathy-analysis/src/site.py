"""Builds website/ from the last demo run."""

from __future__ import annotations

import pathlib

from . import sitekit as sk

ROOT = pathlib.Path(__file__).resolve().parent.parent

META = {
    "name": "Clinical Empathy Analysis",
    "slug": "clinical-empathy-analysis",
    "repo": "2.0-Healthcare-Ai-Systems",
    "pillar": "Healthcare Safety",
    "tagline": "Measuring empathic behaviour in bilingual consultation transcripts by "
               "fusing a theory-grounded weighted lexicon with structural features of "
               "the conversation.",
    "tags": [("EN + 中文", ""), ("73 weighted cues", ""),
             ("leave-one-out", ""), ("authored corpus", "demo"),
             ("circular validation", "warn")],
    "banner": "The correlation on this page is CIRCULAR and must not be read as "
              "validity. The same person wrote the transcripts and the lexicon, so the "
              "transcripts contain the cues the lexicon looks for. What is demonstrated "
              "is that the pipeline is internally consistent and that the evaluation "
              "protocol is sound — not that the measure tracks clinical empathy.",
}


def build_site(results: dict) -> pathlib.Path:
    c, lx, ev = results["corpus"], results["lexicon"], results["evaluation"]
    mt, ab = results["manner_vs_topic"], results["ablation"]
    we = results["worked_example"]

    metrics = sk.metric_grid([
        ("Transcripts", c["n_transcripts"],
         f"{c['n_english']} EN / {c['n_mandarin']} 中文, {c['n_turns']} turns"),
        ("Lexicon cues", lx["n_cues"], f"{len(lx['categories'])} categories"),
        ("LOO correlation", f"{ev['model_loo']['pearson_r']:+.3f}",
         "circular — see the banner"),
        ("Mean abs. error", f"{ev['model_loo']['mae']:.2f}", "on a 0–4 rating scale"),
    ])

    cat_tbl = sk.table(
        ["Category", "Cues", "Behaviour it indexes"],
        [["acknowledgement", lx["per_category"]["acknowledgement"],
          "Naming or reflecting the patient's emotion back to them"],
         ["validation", lx["per_category"]["validation"],
          "Explicit emotional support; normalising the experience"],
         ["exploration", lx["per_category"]["exploration"],
          "Inviting the patient to say more — the strongest behavioural signal"],
         ["partnership", lx["per_category"]["partnership"],
          "Shared decision-making; checking understanding"],
         ["warmth", lx["per_category"]["warmth"],
          "Weak individually, real in aggregate"],
         ["dismissal", lx["per_category"]["dismissal"],
          "Negative weights. Without these, a curt consultation scores as neutral"]],
        numeric_cols=(1,))

    eval_tbl = sk.table(
        ["Predictor", "n", "Pearson r", "Spearman ρ", "MAE"],
        [[ev["lexicon_only"]["name"], ev["lexicon_only"]["n"],
          f"{ev['lexicon_only']['pearson_r']:+.3f}",
          f"{ev['lexicon_only']['spearman_rho']:+.3f}",
          f"{ev['lexicon_only']['mae']:.3f}"],
         [ev["model_loo"]["name"], ev["model_loo"]["n"],
          f"{ev['model_loo']['pearson_r']:+.3f}",
          f"{ev['model_loo']['spearman_rho']:+.3f}",
          f"{ev['model_loo']['mae']:.3f}"]]
        + [[b["name"] + " subset", b["n"], f"{b['pearson_r']:+.3f}",
            f"{b['spearman_rho']:+.3f}", f"{b['mae']:.3f}"]
           for b in ev["by_language"]],
        numeric_cols=(1, 2, 3, 4))

    scatter = sk.line_chart(
        [("ideal", [(0.0, 0.0), (4.0, 4.0)]),
         ("predicted", sorted((p["rating"], p["predicted"])
                              for p in ev["predictions"]))],
        xlabel="analyst rating (0–4)", ylabel="predicted")

    ab_chart = sk.bar_chart(
        [(g["features"], g["pearson_r"]) for g in ab["groups"]], fmt="{:.3f}")

    pred_tbl = sk.table(
        ["ID", "Lang", "Rating", "Lexicon score", "Predicted", "Error"],
        [[p["tid"], p["language"], p["rating"], f"{p['lexicon_score']:.2f}",
          f"{p['predicted']:.2f}", f"{p['error']:+.2f}"]
         for p in ev["predictions"]],
        numeric_cols=(2, 3, 4, 5))

    cue_tbl = sk.table(
        ["Cue", "Category", "Count", "Weight", "Contribution"],
        [[r["cue"], r["category"], r["count"], r["weight"], r["contribution"]]
         for r in we["cues"]],
        numeric_cols=(2, 3, 4))

    body = f"""
<section>
  <h2>What is being measured</h2>
  <div class="stack">
    <p>Not "nice words". Clinical empathy research measures specific behaviours, and the
    lexicon is organised by the behaviour each cue signals, following the structure of
    established consultation-coding schemes — the Empathic Communication Coding System
    and the socio-emotional categories of the Roter Interaction Analysis System.</p>
    <p>Scores are length-normalised per 100 tokens. A long consultation accumulates cue
    hits simply by being long, so an unnormalised count measures duration.</p>
  </div>
</section>

<section>
  <h2>This run</h2>
  <div class="stack-lg">
    {metrics}
    <p class="mono" style="color:var(--muted);font-size:12.5px">
      generated {sk.esc(results['generated_at'])} &middot;
      ratings {c['rating_min']}–{c['rating_max']}, mean {c['rating_mean']}
      (sd {c['rating_sd']}) &middot;
      {len(lx['validation_problems'])} lexicon structural problems
    </p>
  </div>
</section>

<section>
  <h2>The lexicon</h2>
  <div class="stack-lg">
    {cat_tbl}
    <div class="note">
      <h3>Why dismissal carries negative weight</h3>
      <p>Empathy is not only the presence of warmth. A lexicon with no negative
      category scores a curt, blame-shifting consultation as merely neutral, which is
      exactly the case a quality measure needs to catch. Removing the dismissal cues
      drops the correlation from {ab['groups'][0]['pearson_r']:+.3f} to
      {[g for g in ab['groups'] if g['features'] == 'no dismissal cues'][0]['pearson_r']:+.3f}.</p>
    </div>
  </div>
</section>

<section>
  <h2>Agreement with the ratings</h2>
  <div class="stack-lg">
    {eval_tbl}
    {scatter}
    <div class="note warn" style="background:var(--warn-bg);border-color:transparent">
      <h3>This correlation is circular</h3>
      <p>One person wrote the transcripts, the ratings <em>and</em> the lexicon. The
      transcripts therefore contain the cues the lexicon looks for, and the ratings
      reflect the same intuitions that set the weights. A correlation of
      {ev['model_loo']['pearson_r']:+.3f} under these conditions measures internal
      consistency, not validity.</p>
      <p>What the number does establish is narrower and still worth having: the
      pipeline runs end to end, the evaluation protocol is honest (leave-one-out, no
      fold reuse), and the features separate the cases they were designed to separate.
      Validity requires a collected corpus with ratings from clinicians who did not
      build the measure — and inter-rater agreement reported alongside.</p>
    </div>
    {pred_tbl}
  </div>
</section>

<section>
  <h2>Manner or topic?</h2>
  <div class="stack-lg">
    <p>EN-09 and EN-10 carry identical clinical content — a patient worried about test
    results, a doctor explaining a routine follow-up — and differ only in delivery. A
    measure that scores them the same is tracking subject matter, not empathy.</p>
    {sk.table(["Transcript", "Analyst rating", "Lexicon score"],
              [[mt["pair"][0], mt["ratings"][0], f"{mt['lexicon_scores'][0]:.2f}"],
               [mt["pair"][1], mt["ratings"][1], f"{mt['lexicon_scores'][1]:.2f}"]],
              numeric_cols=(1, 2))}
    <p>Separated by {mt['gap']:+.2f} points in the intended direction. This is the most
    diagnostic pair in the corpus and the one worth reproducing first on real data.</p>
  </div>
</section>

<section>
  <h2>Which features carry it</h2>
  <div class="stack-lg">
    {ab_chart}
    {sk.table(["Feature set", "Features", "Pearson r", "MAE"],
              [[g["features"], g["n_features"], f"{g['pearson_r']:+.3f}",
                f"{g['mae']:.3f}"] for g in ab["groups"]],
              numeric_cols=(1, 2, 3))}
    <p>Structural features alone — who talks, how much, how many open questions —
    reach {[g for g in ab['groups'] if g['features'] == 'structure only'][0]['pearson_r']:+.3f}
    without reading a single word of content. That is a useful result in its own right,
    because structural features survive translation and transcription error far better
    than lexical ones.</p>
  </div>
</section>

<section>
  <h2>A scored transcript</h2>
  <div class="stack-lg">
    <p>{sk.esc(we['tid'])}, rated {we['rating']}, lexicon score
    {we['lexicon_score']:.2f}. Every score decomposes into the cues that produced it,
    so a clinician can disagree with a specific line rather than with a number.</p>
    {cue_tbl}
  </div>
</section>

<section>
  <h2>Reproduce it</h2>
  <div class="stack">
    <pre>cd clinical-empathy-analysis
pip install -r requirements.txt
python -m pytest tests/ -q
python -m src.demo</pre>
  </div>
</section>

<section>
  <h2>What this does not establish</h2>
  <div class="stack">
    <ul class="tight">
      <li><strong>No validity evidence.</strong> The correlation is circular, as set out
      above. Nothing here shows the measure tracks clinical empathy.</li>
      <li>Sixteen authored transcripts is not a corpus. Leave-one-out on sixteen items
      has error bars wide enough to contain most values you might care about.</li>
      <li>No inter-rater agreement, because there is one rater.</li>
      <li>No patient outcome is attached to any transcript, so nothing here speaks to
      whether the measured behaviour helps anyone.</li>
      <li>The Mandarin cues are a smaller set than the English ones and were written by
      the same person; the near-identical cross-language correlation reflects that
      symmetry in authoring, not demonstrated cross-lingual validity.</li>
    </ul>
    <p>The predecessor project trained on data synthesised at runtime and evaluated on
    the same generator. This replaces that with a held-out protocol and an explicit
    statement of what is still missing. The remaining gap is a real corpus.</p>
  </div>
</section>
"""
    return sk.build(ROOT, META, body, results)
