# Data — ICU Early-Warning Models

| Field | Value |
|---|---|
| Source | PhysioNet mimic-iv-demo (open access) |
| Stays in file | 140 |
| Stays built | 134 |
| Skipped — too short | 6 |
| Skipped — no vitals | 0 |
| Observations | 24,304 |
| Patient-hours | 12,152 |
| Mean age | 61.9 |
| Features | 31 |
| Grid step | 0.5 h |
| Carry-forward limit | 2.0 h |
| Licence | PhysioNet open-access demo — no credentialing required |
| Run recorded | 2026-09-06T23:55:09+00:00 |

## The files

| File | Bytes | SHA-256 (prefix) |
|---|---|---|
| `mimic-iv-demo/hosp/admissions.csv.gz` | 11,072 | `910b9f160ffdf1e0` |
| `mimic-iv-demo/hosp/patients.csv.gz` | 1,083 | `59dcc6cd8450ddf0` |
| `mimic-iv-demo/icu/chartevents.csv.gz` | 5,549,389 | `90f096ad3db847ed` |
| `mimic-iv-demo/icu/d_items.csv.gz` | 57,073 | `1690467a82234522` |
| `mimic-iv-demo/icu/icustays.csv.gz` | 5,667 | `e05e81aa52a3022e` |

Retrieved 2026-09-06T23:54:24+00:00. These rows are the
`provenance.files` object of `results/latest-real.json`, written by the run itself.

## Event prevalence in this cohort

| Event | Events | Rate | Patients affected |
|---|---|---|---|
| Hypoxemia | 212 | 0.00872 | 45 |
| Hypotension | 3,964 | 0.16310 | 102 |

The two events differ in prevalence by a factor of roughly nineteen, which is
why they are reported separately rather than pooled: a single figure across
both would be dominated by hypotension.

## Why the open-access demo

MIMIC-IV proper is credentialed and cannot be redistributed. The demo subset is
open, carries the same schema, and is enough to show that the pipeline parses
real charting, resamples irregular observations without looking forward, and
produces calibrated risk. It is **not** enough to estimate performance, and the
results file says so in its own text.

## Rules for this folder

1. **Never commit raw clinical files.** `data/raw/` is git-ignored; the manifest
   and hashes are committed instead.
2. **Never report a number from `data/sample/`.**
3. **Record the run.** Every reported figure needs a dated entry in
   [`results/README.md`](../results/README.md).

## Reproduction

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 32 tests
python -m data.fetch           # needs the sources in docs/DATA.md
python -m src.demo --real      # the measured run -> results/latest-real.json
```
