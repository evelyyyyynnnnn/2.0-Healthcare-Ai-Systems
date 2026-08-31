"""Builds website/ from the last demo run."""

from __future__ import annotations

import pathlib

from . import sitekit as sk

ROOT = pathlib.Path(__file__).resolve().parent.parent

META = {
    "name": "icuflow",
    "slug": "pyhealth-rhealth-extension",
    "repo": "2.0-Healthcare-Ai-Systems",
    "pillar": "Healthcare Safety",
    "tagline": "An installable package for the three parts of a clinical prediction "
               "pipeline that are easiest to get wrong: subject-grouped splits, "
               "calibration, and alarm comparison at matched sensitivity.",
    "tags": [("pip-installable", ""), ("PyHealth-compatible shape", ""),
             ("CLI", ""), ("not yet published", "warn")],
    "banner": "The package is real and installable from this repository, but it is "
              "NOT published to PyPI, so it has no download statistics and no users. "
              "The experiments below run on synthetic subjects designed to exhibit the "
              "specific failure each module prevents.",
}


def build_site(results: dict) -> pathlib.Path:
    lk, al, g = results["leakage"], results["alarms"], results["guards"]
    cases = results["calibration"]["cases"]

    metrics = sk.metric_grid([
        ("Leakage inflation", f"+{lk['inflation_pct']:.1f}%",
         "AUROC from splitting rows not subjects"),
        ("ECE reduction", f"{(1 - cases[1]['isotonic_ece'] / cases[1]['raw_ece']) * 100:.0f}%",
         "on an overconfident score"),
        ("False alerts", f"−{al['reduction_pct']:.1f}%",
         f"at matched {al['target_sensitivity']:.0%} sensitivity"),
        ("Public API", results["package"]["exports"], "functions and classes"),
    ])

    leak_tbl = sk.table(
        ["Split strategy", "Test AUROC", "What it means"],
        [["Random row split", f"{lk['row_split_auroc']:.4f}",
          "The same subjects appear in train and test"],
         ["Subject-grouped split", f"{lk['subject_split_auroc']:.4f}",
          "Every test subject is unseen"],
         ["Difference", f"+{lk['inflation']:.4f}",
          f"{lk['inflation_pct']:.1f}% of the honest score, invented by the split"]],
        numeric_cols=(1,))

    leak_chart = sk.bar_chart(
        [("random row split", lk["row_split_auroc"]),
         ("subject-grouped split", lk["subject_split_auroc"])], fmt="{:.4f}")

    cal_tbl = sk.table(
        ["Score", "Raw ECE", "Isotonic", "Platt", "AUROC before → after"],
        [[c["case"], f"{c['raw_ece']:.5f}", f"{c['isotonic_ece']:.5f}",
          f"{c['platt_ece']:.5f}", f"{c['raw_auroc']:.3f} → {c['isotonic_auroc']:.3f}"]
         for c in cases],
        numeric_cols=(1, 2, 3))

    alarm_tbl = sk.table(
        ["", "Ward baseline", "Model", "Change"],
        [["Sensitivity", f"{al['baseline_sensitivity']:.3f}",
          f"{al['model_sensitivity']:.3f}", "matched by construction"],
         ["False alerts", f"{al['baseline_false_alerts']:,}",
          f"{al['model_false_alerts']:,}", f"{al['reduction_pct']:.1f}% fewer"]],
        numeric_cols=(1, 2))

    guard_tbl = sk.table(
        ["Guard", "Fired correctly"],
        [["Subject appearing in both splits raises SubjectLeakError",
          "yes" if g["leak_detected"] else "NO"],
         ["Row with an event already active is dropped",
          "yes" if g["active_event_row_dropped"] else "NO"],
         ["Row without full follow-up is dropped",
          "yes" if g["insufficient_followup_dropped"] else "NO"]])

    body = f"""
<section>
  <h2>Why these three</h2>
  <div class="stack">
    <p>They are the parts of a clinical prediction pipeline that get
    re-implemented in every project and get quietly wrong in most of them. The
    functions take and return plain arrays, so they drop into an existing PyHealth
    or RHealth task without adopting another framework.</p>
  </div>
</section>

<section>
  <h2>This run</h2>
  <div class="stack-lg">
    {metrics}
    <p class="mono" style="color:var(--muted);font-size:12.5px">
      generated {sk.esc(results['generated_at'])} &middot;
      icuflow {sk.esc(results['package']['version'])} &middot;
      {lk['n_subjects']} subjects, {lk['n_rows']:,} rows
    </p>
  </div>
</section>

<section>
  <h2>Subject leakage, measured</h2>
  <div class="stack-lg">
    {leak_chart}
    {leak_tbl}
    <div class="note">
      <h3>The first version of this experiment showed no leakage at all</h3>
      <p>Giving each subject an offset that shifted both a feature and the label
      produced identical scores under both splits. That is correct behaviour: the
      offset-to-label relationship is real and <em>generalises</em>, so a model
      learning it does just as well on unseen subjects. Row-splitting looked harmless.</p>
      <p>Leakage needs a subject effect that does not generalise. Here each subject
      carries a random label propensity unrelated to anything predictive, plus a
      near-noise-free fingerprint feature that identifies them. A model trained on rows
      memorises fingerprint → propensity and scores well on held-out rows from subjects
      it has already seen; on held-out subjects the fingerprints are new and the
      memorised mapping is worth nothing.</p>
      <p>That is the structure of a patient baseline in ICU data, and it is why
      row-splitting inflates published clinical results.</p>
    </div>
  </div>
</section>

<section>
  <h2>Calibration, including when not to use it</h2>
  <div class="stack-lg">
    {cal_tbl}
    <div class="note">
      <h3>Both rows are here on purpose</h3>
      <p>Boosted trees trained on log loss come out close to calibrated already, and on
      those, calibration is a no-op that adds variance &mdash; ECE goes from
      {cases[0]['raw_ece']:.3f} to {cases[0]['isotonic_ece']:.3f}, i.e. slightly
      <em>worse</em>. The module earns its place on scores that are not probabilities:
      a margin, a rank, an overconfident output. On the distorted score, ECE drops from
      {cases[1]['raw_ece']:.3f} to {cases[1]['isotonic_ece']:.3f}.</p>
      <p>AUROC is unchanged in both rows, which is the check that the calibrators are
      monotone: calibration must move probabilities without reordering patients.
      Publishing only the flattering row would misrepresent when to reach for this.</p>
    </div>
  </div>
</section>

<section>
  <h2>Alarm comparison</h2>
  <div class="stack-lg">
    {alarm_tbl}
    <p>Both systems are driven to the same sensitivity before false alerts are counted.
    Any alerting system can cut false alarms by catching fewer events, so a comparison
    made at each system's own convenient operating point is not a comparison.</p>
  </div>
</section>

<section>
  <h2>Guards</h2>
  <div class="stack-lg">
    {guard_tbl}
    <p>These are assertions a pipeline can call, not conventions someone has to
    remember. <code>assert_no_subject_leak</code> is meant to sit in a test.</p>
  </div>
</section>

<section>
  <h2>Install and use</h2>
  <div class="stack">
    <pre>pip install -e .            # from this directory

python -c "
import numpy as np, icuflow
sp = icuflow.split_by_subject(subject_ids, test_frac=0.3, seed=0)
icuflow.assert_no_subject_leak(subject_ids, sp.train, sp.test)
"

icuflow check-split predictions.csv --subject-col subject_id
icuflow score predictions.csv --label-col y --score-col p --baseline-col news2</pre>
  </div>
</section>

<section>
  <h2>What this does not establish</h2>
  <div class="stack">
    <ul class="tight">
      <li><strong>Not published.</strong> The package is installable from this
      repository and is not on PyPI. It has no downloads, no registry statistics
      and no users, and nothing here should be described as adopted.</li>
      <li>Compatibility with PyHealth and RHealth is at the level of data shapes:
      plain arrays in, plain arrays out. It has not been tested against either
      framework's task API.</li>
      <li>Every experiment above runs on synthetic subjects constructed to exhibit
      the specific failure each module prevents. That demonstrates the module works
      on the problem it was built for; it is not evidence about real cohorts.</li>
      <li>The calibrators are small hand-written implementations, not
      scikit-learn's. They are here so the package has no heavy dependency, and
      they have not been benchmarked against it.</li>
    </ul>
  </div>
</section>
"""
    return sk.build(ROOT, META, body, results)
