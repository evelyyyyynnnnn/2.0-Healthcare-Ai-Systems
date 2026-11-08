# Method — icuflow

## Problem

Package the three parts of a clinical prediction pipeline that are easiest to
get wrong — subject-grouped splits, calibration, and alarm comparison at matched
sensitivity — and measure what each one costs when it is wrong.

## Why the approach is non-obvious

Each of these three is well understood in isolation and still gets missed in
practice, because nothing fails loudly when it does. A row-level split on
repeated-measures data produces a better number, not an error. The only way to
make the problem visible is to run both versions on the same cohort and publish
the gap.

## Method

`icuflow` 0.1.0 exports 14 functions and a CLI with the subcommands
`check-split`, `score`. Three guards run on every call and each fired on this run:

| Guard | Fired |
|---|---|
| `leak_detected` | true |
| `active_event_row_dropped` | true |
| `insufficient_followup_dropped` | true |

The leakage experiment holds the cohort, the features and the model fixed and
varies only the split: random by row, then grouped by subject.

## What the run found

| Split | AUROC |
|---|---|
| Row-level (wrong) | **0.9058** |
| Subject-grouped (right) | **0.4989** |
| Absolute inflation | 0.4070 |
| Relative inflation | **81.57%** |

A model that looks strong at 0.906 is at
0.499 once the same patient can no longer appear on
both sides of the split — indistinguishable from chance. The model had learned
to recognise patients, not risk.

Cohort: 97 subjects, 134 stays, 37
readmissions, 22,512 rows, positive rate 0.02838.

## Limitations

- **about 100 ICU stays. The direction of the leakage effect is meaningful; its magnitude, from a cohort this small, is not a population estimate.**
- **Not on PyPI.** `published` is `false`. No download
  statistics exist and none are claimed.
- **One task, one cohort.** The direction of the effect is the result; the
  magnitude is specific to this cohort.

## Reproduction

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 29 tests
python -m data.fetch           # needs the sources in docs/DATA.md
python -m src.demo --real      # the measured run -> results/latest-real.json
```
