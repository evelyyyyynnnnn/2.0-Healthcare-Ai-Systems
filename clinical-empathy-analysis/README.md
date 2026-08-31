# Clinical Empathy Analysis

> **Status: scaffold.** Structure only — no method, data or result is claimed
> yet. Every "not yet measured" below is a real gap, not a placeholder to be
> filled in with an estimate.

**Repository:** `2.0-Healthcare-Ai-Systems`
**NIW pillar (Dhanasar prong 1):** Healthcare Safety
**Evidence value:** Supporting — needs a real corpus to be defensible

## Core idea

Quantify how much empathy a doctor expresses in Chinese clinical consultation transcripts, fusing a weighted empathy lexicon with ML models over engineered text features.

## Why it earns its place

Converts the one existing project from a synthetic-data demo into a defensible result.

## The petition claim it supports

> Clinical communication quality measurement.

**What the portfolio shows today:** The archived version scores 11 ophthalmology transcripts and trains on runtime-synthetic data.

**Action required:** Upgrade with a real labelled corpus and a held-out clinical evaluation.

Prior work to build on: `previous/doctor-empathy-analysis`.

## Petition-grade checklist

A project counts as petition-grade only when all five are true. None are yet.

- [ ] Original work, authored here
- [ ] A stated method (`docs/METHOD.md`)
- [ ] Real data at a stated scale (`docs/DATA.md` — target: Real labelled consultation corpus (scale to be stated))
- [ ] A measured result (`results/README.md`)
- [ ] A README a reviewer can follow, start to finish

## Measured results

Target scale: **Real labelled consultation corpus (scale to be stated)**

| Metric | Baseline | Result | Out-of-sample |
|---|---|---|---|
| Agreement with clinician-rated empathy (held-out) | _not yet measured_ | _not yet measured_ | _pending_ |
| Inter-annotator agreement on the labelled corpus | _not yet measured_ | _not yet measured_ | _pending_ |
| Lexicon vs. ML ablation | _not yet measured_ | _not yet measured_ | _pending_ |

Populate this from `results/`. Do not cite any number in the petition that does
not appear here with a run date behind it.

## Layout

```
clinical-empathy-analysis/
├── README.md        this file
├── docs/
│   ├── METHOD.md    what the method is and why it is non-obvious
│   ├── DATA.md      source, scale, licence, and how to reproduce the pull
│   └── EVIDENCE.md  the petition claim, the gap, and the exhibit it becomes
├── src/             implementation
├── data/            pointers and manifests — never raw licensed data
├── results/         measured results, run logs, and the baseline comparison
└── tests/           tests that establish the result is reproducible
```

---
Scaffold generated from `NIW_Project_Portfolio_and_Gap_Plan.xlsx` (sheets: Repo Build-Out Plan, Core Ideas at a Glance, NIW Claim vs Repo Evidence, Notion 创业 Alignment). Structure only — no results are claimed here yet.
