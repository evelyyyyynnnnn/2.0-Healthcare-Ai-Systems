# Data — Physiological Waveform Pipeline

| Field | Value |
|---|---|
| Source | PhysioNet BIDMC PPG and Respiration Dataset (open access) |
| Records | 10 |
| Channel | `pleth` |
| Sampling rate | 125 Hz |
| Duration per record | 480 s |
| Windows per record | 48 |
| Signal is PPG, not an arterial line | **True** |
| Artifact locations annotated | **False** |
| Licence | PhysioNet open access |
| Run recorded | 2026-09-06T23:56:50+00:00 |

## The records

| Record | Windows | Accepted | Accept rate | Derived HR | Monitor HR | Difference |
|---|---|---|---|---|---|---|
| 01 | 48 | 1 | 2.1% | 93.52 | 91.0 | +2.52 |
| 05 | 48 | 45 | 93.8% | 98.00 | 98.0 | -0.00 |
| 11 | 48 | 32 | 66.7% | 91.62 | 92.0 | -0.38 |
| 17 | 48 | 48 | 100.0% | 80.43 | 77.0 | +3.43 |
| 23 | 48 | 38 | 79.2% | 63.08 | 63.0 | +0.08 |
| 29 | 48 | 43 | 89.6% | 90.21 | 90.0 | +0.21 |
| 35 | 48 | 43 | 89.6% | 98.51 | 101.0 | -2.49 |
| 41 | 48 | 29 | 60.4% | 79.09 | 79.0 | +0.09 |
| 47 | 48 | 47 | 97.9% | 83.37 | 83.0 | +0.37 |
| 53 | 48 | 48 | 100.0% | 85.63 | 92.0 | -6.37 |

These rows are the `per_record` array of `results/latest-real.json`, written by
the run itself.

## Why this dataset

BIDMC publishes a simultaneous bedside-monitor heart rate alongside the raw
waveform, which gives the derived figure something real to be checked against.
That reference is the reason this dataset was chosen over a larger one without it.

## What this dataset cannot tell you

- **BIDMC does not annotate artifact locations, so rejection precision and recall have no answer key on this dataset**
- **PLETH is a photoplethysmogram, not an arterial line; a systolic pressure in mmHg is not something this signal measures**

## Rules for this folder

1. **Never commit raw waveform files.** `data/raw/` is git-ignored.
2. **Never report a number from `data/sample/`.**
3. **Record the run.** Every reported figure needs a dated entry in
   [`results/README.md`](../results/README.md).

## Reproduction

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 33 tests
python -m data.fetch           # needs the sources in docs/DATA.md
python -m src.demo --real      # the measured run -> results/latest-real.json
```
