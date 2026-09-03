# 2.0 — Healthcare AI Systems

Clinical decision-support and patient-safety systems. This repository carries the Healthcare Safety pillar of the endeavor.

Part of a five-repository portfolio supporting the endeavor described in the
EB2-NIW petition: **optimization-driven, system-level decision frameworks** —
integrating operations research, mathematical optimization and applied AI — for
domains where a wrong decision carries systemic consequences. The three pillars
are financial stability, healthcare safety and secure digital infrastructure.

| | |
|---|---|
| Petition-grade projects today | 1 petition-grade project (empathy analysis, synthetic training data) |
| Verdict | **Needs 3 more — the thinnest repo against the strongest claims** |

> "Petition-grade" means: original work, a stated method, real data at a stated
> scale, a measured result, and a README a reviewer can follow. Counts exclude
> duplicates, forks of third-party work, retired projects, and asset-only
> folders.

## Projects

| Folder | Project | Pillar | Evidence value |
|---|---|---|---|
| [`icu-early-warning/`](icu-early-warning/) | ICU Early-Warning Models | Healthcare Safety | CORE — highest priority in this repo |
| [`pyhealth-rhealth-extension/`](pyhealth-rhealth-extension/) | PyHealth / RHealth Extension | Healthcare Safety | CORE — makes an adoption claim checkable |
| [`physiological-waveform-pipeline/`](physiological-waveform-pipeline/) | Physiological Waveform Pipeline | Healthcare Safety | Supporting — supplies the input layer for the ICU models |
| [`clinical-empathy-analysis/`](clinical-empathy-analysis/) | Clinical Empathy Analysis | Healthcare Safety | Supporting — needs a real corpus to be defensible |

## What each one is

### 1. ICU Early-Warning Models — [`icu-early-warning/`](icu-early-warning/)

Early-warning models for hypoxemia and hypotension on MIMIC-IV or eICU, with calibration and uncertainty estimates.

*Why it earns its place:* Supplies the 12,000-patient and 22%-false-alarm claims, which currently have no artifact anywhere. The largest evidence gap in the portfolio.

*Target scale:* 12,000+ ICU patients (MIMIC-IV / eICU)

### 2. PyHealth / RHealth Extension — [`pyhealth-rhealth-extension/`](pyhealth-rhealth-extension/)

A PyHealth / RHealth extension released as a pip package.

*Why it earns its place:* Makes the "adopted by multiple external research groups" claim checkable via registry download statistics.

*Target scale:* Public registry download statistics

### 3. Physiological Waveform Pipeline — [`physiological-waveform-pipeline/`](physiological-waveform-pipeline/)

A high-frequency physiological waveform pipeline: preprocessing plus a released derived dataset.

*Why it earns its place:* Supplies the 58,000-waveform-hours claim and gives the ICU models a documented input layer.

*Target scale:* 58,000+ waveform-hours

### 4. Clinical Empathy Analysis — [`clinical-empathy-analysis/`](clinical-empathy-analysis/)

Quantify how much empathy a doctor expresses in Chinese clinical consultation transcripts, fusing a weighted empathy lexicon with ML models over engineered text features.

*Why it earns its place:* Converts the one existing project from a synthetic-data demo into a defensible result.

*Target scale:* Real labelled consultation corpus (scale to be stated)

## Repository layout

```
2.0-Healthcare-Ai-Systems/
├── icu-early-warning/
├── pyhealth-rhealth-extension/
├── physiological-waveform-pipeline/
├── clinical-empathy-analysis/
│
│   ── earlier work, promoted out of previous/ ──
└── doctor-empathy-analysis-v1/
```

Every rebuilt project carries the same skeleton: `README.md`, `src/`, `data/`, `results/`, `tests/`, `website/`.

## Ground rules

1. **No number without a run log.** Anything cited in the petition must appear
   in that project's `results/README.md` with a run date behind it.
2. **No simulated data under a real claim.** Sample data lives in
   `data/sample/`, labelled, and is never the source of a cited figure.
3. **Adoption must be documentable** — named institutions, dated
   correspondence, registry statistics. Never an inflated count.
4. **Third-party and forked code stays labelled** and is never counted.

## Earlier work

There is no `previous/` folder any more. Everything that was in it has been promoted to the top level, so every piece of work in this repository is a first-class folder that can be opened, read and continued.

Nothing was deleted except items the rebuild genuinely supersedes; those remain in git history.

| Folder | What it is | How it may be used |
|---|---|---|
| [`doctor-empathy-analysis-v1/`](doctor-empathy-analysis-v1/) | The first empathy scorer, with three trained models, its figures and its sample dataset. | Superseded by `clinical-empathy-analysis/`, which runs on 1,077 real consultations. Kept because the trained models and outputs are not reproduced by the rewrite. Cite the rewrite, not this. |

**Read the third column before citing anything here.** Forks of third-party work, duplicates, retired projects and asset-only folders are labelled as such and are not part of the petition's evidence.


---
Scaffold generated from `NIW_Project_Portfolio_and_Gap_Plan.xlsx` (sheets: Repo Build-Out Plan, Core Ideas at a Glance, NIW Claim vs Repo Evidence, Notion 创业 Alignment). Structure only — no results are claimed here yet.
