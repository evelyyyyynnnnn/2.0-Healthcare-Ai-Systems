# ICU Early-Warning Models

> **Status: scaffold.** Structure only — no method, data or result is claimed
> yet. Every "not yet measured" below is a real gap, not a placeholder to be
> filled in with an estimate.

**Repository:** `2.0-Healthcare-Ai-Systems`
**NIW pillar (Dhanasar prong 1):** Healthcare Safety
**Evidence value:** CORE — highest priority in this repo

## Core idea

Early-warning models for hypoxemia and hypotension on MIMIC-IV or eICU, with calibration and uncertainty estimates.

## Why it earns its place

Supplies the 12,000-patient and 22%-false-alarm claims, which currently have no artifact anywhere. The largest evidence gap in the portfolio.

## The petition claim it supports

> Predictive models over 12,000 ICU patients and 58,000 hours of high-frequency physiological waveforms; false-alert rate cut 22%.

**What the portfolio shows today:** Repo 2.0 contains one project: empathy scoring over 11 ophthalmology consultation transcripts. Nothing in any repo is ICU, waveform, or alarm-related.

**Action required:** Build on MIMIC-IV or eICU as repo 2.0 project #1, with calibration and uncertainty reporting. Without it the healthcare pillar rests on a claim with no artifact.

No prior work in the portfolio — this starts from scratch.

## Petition-grade checklist

A project counts as petition-grade only when all five are true. None are yet.

- [ ] Original work, authored here
- [ ] A stated method (`docs/METHOD.md`)
- [ ] Real data at a stated scale (`docs/DATA.md` — target: 12,000+ ICU patients (MIMIC-IV / eICU))
- [ ] A measured result (`results/README.md`)
- [ ] A README a reviewer can follow, start to finish

## Measured results

Target scale: **12,000+ ICU patients (MIMIC-IV / eICU)**

| Metric | Baseline | Result | Out-of-sample |
|---|---|---|---|
| AUROC / AUPRC for hypoxemia and hypotension | _not yet measured_ | _not yet measured_ | _pending_ |
| False-alert-rate reduction vs. current threshold alarms | _not yet measured_ | _not yet measured_ | _pending_ |
| Calibration (Brier score, reliability curve) | _not yet measured_ | _not yet measured_ | _pending_ |
| Predictive-uncertainty coverage | _not yet measured_ | _not yet measured_ | _pending_ |

Populate this from `results/`. Do not cite any number in the petition that does
not appear here with a run date behind it.

## Layout

```
icu-early-warning/
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
