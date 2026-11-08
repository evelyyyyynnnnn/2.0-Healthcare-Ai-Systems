# Data — icuflow

| Field | Value |
|---|---|
| Source | PhysioNet mimic-iv-demo (open access) |
| Stays in file | 140 |
| Stays built | 134 |
| Subjects | 97 |
| Readmissions | 37 |
| Rows | 22,512 |
| Positive rate | 0.02838 |
| Skipped — too short | 6 |
| Skipped — no vitals | 0 |
| Grid step | 0.5 h |
| Carry-forward limit | 2.0 h |
| Licence | PhysioNet open-access demo — no credentialing required |
| Run recorded | 2026-09-06T23:57:16+00:00 |

## The files

| File | Bytes | SHA-256 (prefix) |
|---|---|---|
| `mimic-iv-demo/hosp/admissions.csv.gz` | 11,072 | `910b9f160ffdf1e0` |
| `mimic-iv-demo/hosp/patients.csv.gz` | 1,083 | `59dcc6cd8450ddf0` |
| `mimic-iv-demo/icu/chartevents.csv.gz` | 5,549,389 | `90f096ad3db847ed` |
| `mimic-iv-demo/icu/d_items.csv.gz` | 57,073 | `1690467a82234522` |
| `mimic-iv-demo/icu/icustays.csv.gz` | 5,667 | `e05e81aa52a3022e` |

These rows are the `leakage.provenance.files` object of
`results/latest-real.json`, written by the run itself.

## Why subject count matters more than row count

22,512 rows sounds like a reasonable sample until you notice they come
from 97 people. A split that ignores that distinction has roughly
232 chances per person to put the same patient on both sides, which
is exactly the mechanism this package exists to block — and exactly why the
row-level AUROC looks so much better than the truth.

## What this cohort cannot tell you

about 100 ICU stays. The direction of the leakage effect is meaningful; its magnitude, from a cohort this small, is not a population estimate.

`cohort_is_a_demonstration_not_a_study` is `true`
in the results file.

## Reproduction

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 29 tests
python -m data.fetch           # needs the sources in docs/DATA.md
python -m src.demo --real      # the measured run -> results/latest-real.json
```
