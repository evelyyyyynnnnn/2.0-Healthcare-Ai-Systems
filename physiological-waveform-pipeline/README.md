# Physiological Waveform Pipeline

> **Status: scaffold.** Structure only — no method, data or result is claimed
> yet. Every "not yet measured" below is a real gap, not a placeholder to be
> filled in with an estimate.

**Repository:** `2.0-Healthcare-Ai-Systems`
**NIW pillar (Dhanasar prong 1):** Healthcare Safety
**Evidence value:** Supporting — supplies the input layer for the ICU models

## Core idea

A high-frequency physiological waveform pipeline: preprocessing plus a released derived dataset.

## Why it earns its place

Supplies the 58,000-waveform-hours claim and gives the ICU models a documented input layer.

## The petition claim it supports

> 58,000 hours of high-frequency physiological waveforms.

**What the portfolio shows today:** Nothing waveform-related exists in any repository.

**Action required:** Build the preprocessing pipeline, publish the derived dataset with a stated scale, and wire it into icu-early-warning/ as its documented input.

No prior work in the portfolio — this starts from scratch.

## Petition-grade checklist

A project counts as petition-grade only when all five are true. None are yet.

- [ ] Original work, authored here
- [ ] A stated method (`docs/METHOD.md`)
- [ ] Real data at a stated scale (`docs/DATA.md` — target: 58,000+ waveform-hours)
- [ ] A measured result (`results/README.md`)
- [ ] A README a reviewer can follow, start to finish

## Measured results

Target scale: **58,000+ waveform-hours**

| Metric | Baseline | Result | Out-of-sample |
|---|---|---|---|
| Waveform-hours processed | _not yet measured_ | _not yet measured_ | _pending_ |
| Signal-quality pass rate | _not yet measured_ | _not yet measured_ | _pending_ |
| Artifact-rejection accuracy | _not yet measured_ | _not yet measured_ | _pending_ |
| Derived-dataset release size | _not yet measured_ | _not yet measured_ | _pending_ |

Populate this from `results/`. Do not cite any number in the petition that does
not appear here with a run date behind it.

## Layout

```
physiological-waveform-pipeline/
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
