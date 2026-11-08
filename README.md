# 2.0 — Healthcare AI Systems

[![tests](https://github.com/evelyyyyynnnnn/2.0-Healthcare-Ai-Systems/actions/workflows/tests.yml/badge.svg)](https://github.com/evelyyyyynnnnn/2.0-Healthcare-Ai-Systems/actions/workflows/tests.yml)

Clinical decision-support and patient-safety systems. This repository carries the Healthcare Safety pillar of the endeavor.

Part of a five-repository portfolio on one theme: **optimization-driven,
system-level decision frameworks** —
integrating operations research, mathematical optimization and applied AI — for
domains where a wrong decision carries systemic consequences. The three pillars
are financial stability, healthcare safety and secure digital infrastructure.

| | |
|---|---|
| Projects with measured results | **4 of 4** |
| Tests across the four suites | **126** |
| Running on real third-party data | **4 of 4** |
| Most recent reportable run | 2026-09-06 |

> "Measured" means: original work, a stated method, real data at a stated
> scale, a measured result, and a README a reviewer can follow. Counts exclude
> duplicates, forks of third-party work, retired projects, and asset-only
> folders.

## Projects

| Folder | Project | Pillar | What it measures |
|---|---|---|---|
| [`2-icu-early-warning/`](2-icu-early-warning/) | ICU Early-Warning Models | Healthcare Safety | Calibration of a deployed-style alarm |
| [`4-pyhealth-rhealth-extension/`](4-pyhealth-rhealth-extension/) | PyHealth / RHealth Extension | Healthcare Safety | What subject leakage costs, measured |
| [`3-physiological-waveform-pipeline/`](3-physiological-waveform-pipeline/) | Physiological Waveform Pipeline | Healthcare Safety | PPG heart rate against a bedside monitor |
| [`1-clinical-empathy-analysis/`](1-clinical-empathy-analysis/) | Clinical Empathy Analysis | Healthcare Safety | A lexical empathy measure, on real transcripts |

## What each one is

### 1. ICU Early-Warning Models — [`2-icu-early-warning/`](2-icu-early-warning/)

Hypoxemia and hypotension four hours ahead on the PhysioNet MIMIC-IV open-access
demo — 134 stays, 24,304 observations, 31 features — compared at a matched
sensitivity of 80% against the single-threshold alarm a ward actually runs.

**The finding is calibration, not discrimination.** The ward-style alarm has an
expected calibration error of **0.276** on hypoxemia and
**0.416** on hypotension; plain logistic regression reaches
**0.030** and **0.045**. At matched sensitivity that is worth
**77.2 → 62.3** and **31.8 → 22.9** false alerts per 100 observations.
Gradient boosting does not beat logistic regression here, and that is reported.

### 2. PyHealth / RHealth Extension — [`4-pyhealth-rhealth-extension/`](4-pyhealth-rhealth-extension/)

`icuflow` 0.1.0, an installable package with 14 exports and a working CLI,
covering the three things easiest to get wrong in a clinical pipeline:
subject-grouped splits, calibration, and alarm comparison at matched sensitivity.

**It measures what one of those mistakes costs.** On the same cohort, features
and model, a random row-level split reports **AUROC 0.9058**; a subject-grouped
split reports **0.4989** — chance. That is **81.57%** inflation, from
97 subjects and 22,512 rows.

The package is **not** published to PyPI. It has no downloads and no users, and
neither is claimed.

### 3. Physiological Waveform Pipeline — [`3-physiological-waveform-pipeline/`](3-physiological-waveform-pipeline/)

Heart rate derived from the photoplethysmogram on the PhysioNet BIDMC dataset —
10 records at 125 Hz — checked against the bedside monitor's own
reported rate.

Agreement is close: median difference **0.084 bpm**, worst case
6.372 bpm, **9 of 10 records within 5 bpm**. The more useful number is
the one usually left out: the artifact gate accepts between **2.1% and 100%**
of windows depending on the record, so the quality gate — not the beat detector —
is where the variance lives.

Artifact precision and recall are **not** reported: BIDMC does not annotate
artifact locations, so there is no answer key.

### 4. Clinical Empathy Analysis — [`1-clinical-empathy-analysis/`](1-clinical-empathy-analysis/)

A theory-grounded weighted empathy lexicon fused with structural features of the
conversation, scored against **1,076 real doctor–patient transcripts** from
MTS-Dialog.

**The result is negative and is reported as such.** The lexicon fires no cue at
all on **731 of them (67.9%)**, the median score is **0.0**, and only
25 distinct cues ever fire. Real consultation language does not look like
the language the communication-skills literature uses to describe empathy.

**Validity is withheld rather than reported.** MTS-Dialog carries no
clinician-assigned empathy ratings, so there is nothing honest to correlate
against, and no correlation is published.

## Repository layout

```
2.0-Healthcare-Ai-Systems/
├── 2-icu-early-warning/
├── 4-pyhealth-rhealth-extension/
├── 3-physiological-waveform-pipeline/
└── 1-clinical-empathy-analysis/
```

Every rebuilt project carries the same skeleton: `README.md`, `src/`, `data/`, `results/`, `tests/`, `website/`.

> Earlier and off-theme prior work (dashboards, consumer apps, templates, and other non-endeavor folders) has been moved to a separate `archive-prior-work` repository to keep this repository focused on the endeavor. It remains recoverable from this repository's git history.

## Ground rules

1. **No number without a run log.** Any figure quoted anywhere in this
   repository must appear in that project's `results/README.md` with a run
   date behind it.
2. **No simulated data under a real claim.** Sample data lives in
   `data/sample/`, labelled, and is never the source of a cited figure.
3. **Adoption must be documentable** — named institutions, dated
   correspondence, registry statistics. Never an inflated count.
4. **Third-party and forked code stays labelled** and is never counted.

---
Every figure on this page is read from a project's `results/latest-real.json` and repeated in its `results/README.md` with a run date.

## Licence

[Apache License 2.0](LICENSE). The code, benchmarks and documentation in this
repository may be reused, modified and redistributed under its terms, including
by projects that need a patent grant to incorporate them.

Third-party data and vendored code kept under `*/data/` and
`historical-archive/` carry their own terms and are labelled where they appear.
