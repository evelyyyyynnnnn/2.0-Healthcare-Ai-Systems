# Results — ICU Early-Warning Models

This file is the only source for the numbers quoted elsewhere in this
repository; if a figure is not here with a run date, it is not cited.

| File | Data | Reportable |
|---|---|---|
| [`latest-real.json`](latest-real.json) | MIMIC-IV open-access demo, 134 stays | **yes** |
| [`latest.json`](latest.json) | synthetic cohort from `src/cohort.py` | no |

## Measured result

Run `2026-09-06T23:55:09+00:00` · 134 stays · 24,304 observations ·
31 features · matched sensitivity **80%** · horizon 4 h

### Hypoxemia — 6,568 test rows, event rate 0.050

| Model | AUROC | AUPRC | ECE | Sensitivity | PPV | False alerts / 100 obs |
|---|---|---|---|---|---|---|
| single-threshold alarm | 0.6196 | 0.1434 | **0.2761** | 0.846 | 0.0522 | **77.2** |
| logistic | 0.6432 | 0.1155 | **0.0301** | 0.800 | 0.0606 | **62.3** |
| gradient boosting | 0.5663 | 0.0667 | **0.0365** | 0.800 | 0.0538 | **70.7** |
| gradient boosting + isotonic | 0.5657 | 0.0802 | **0.0311** | 0.818 | 0.0539 | **72.2** |

### Hypotension — 5,303 test rows, event rate 0.310

| Model | AUROC | AUPRC | ECE | Sensitivity | PPV | False alerts / 100 obs |
|---|---|---|---|---|---|---|
| single-threshold alarm | 0.7481 | 0.5727 | **0.4158** | 0.816 | 0.4431 | **31.8** |
| logistic | 0.7915 | 0.5911 | **0.0446** | 0.800 | 0.5190 | **22.9** |
| gradient boosting | 0.7814 | 0.5769 | **0.0363** | 0.800 | 0.4968 | **25.1** |
| gradient boosting + isotonic | 0.7830 | 0.5729 | **0.0609** | 0.800 | 0.4916 | **25.6** |

## Run log

| Run | Date | Scale | Command |
|---|---|---|---|
| real | 2026-09-06 | 134 stays, 24,304 obs | `python -m src.demo --real` |

## What this does *not* establish

- **Performance.** about 100 ICU stays. Enough to show the pipeline parses real charting, resamples irregular observations onto a regular grid without looking forward, and produces calibrated risk. Not enough to estimate discrimination for clinical use.
- **Generalisation.** One open-access demo cohort, one institution.
- **Clinical benefit.** A lower false-alert rate at matched sensitivity is a
  property of the model, not evidence of an outcome.
