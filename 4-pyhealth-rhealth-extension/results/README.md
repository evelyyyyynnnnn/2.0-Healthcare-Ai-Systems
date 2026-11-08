# Results — icuflow

This file is the only source for the numbers quoted elsewhere in this
repository; if a figure is not here with a run date, it is not cited.

| File | Data | Reportable |
|---|---|---|
| [`latest-real.json`](latest-real.json) | MIMIC-IV open-access demo | **yes** |
| [`latest.json`](latest.json) | synthetic subjects | no |

## Measured result — subject leakage

Run `2026-09-06T23:57:16+00:00` · sources in [`../docs/DATA.md`](../docs/DATA.md)

| Split | AUROC |
|---|---|
| Row-level (wrong) | **0.9058** |
| Subject-grouped (right) | **0.4989** |
| Absolute inflation | 0.4070 |
| Relative inflation | **81.57%** |

Cohort: 97 subjects · 134 stays · 37
readmissions · 22,512 rows · positive rate 0.02838

## Package

| | |
|---|---|
| Name / version | `icuflow` 0.1.0 |
| Exports | 14 |
| CLI subcommands | `check-split`, `score` |
| CLI exit code | 0 |
| Published to PyPI | **false** |

## Guards, all of which fired on this run

| Guard | Fired |
|---|---|
| `leak_detected` | true |
| `active_event_row_dropped` | true |
| `insufficient_followup_dropped` | true |

## Run log

| Run | Date | Scale | Command |
|---|---|---|---|
| real | 2026-09-06 | 97 subjects, 22,512 rows | `python -m src.demo --real` |

## What this does *not* establish

- **A population estimate.** about 100 ICU stays. The direction of the leakage effect is meaningful; its magnitude, from a cohort this small, is not a population estimate.
- **Adoption.** The package is not on PyPI; there are no users and no downloads.
- **Generalisation.** One task, one cohort, one institution.
