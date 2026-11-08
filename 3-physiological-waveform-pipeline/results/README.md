# Results — Physiological Waveform Pipeline

This file is the only source for the numbers quoted elsewhere in this
repository; if a figure is not here with a run date, it is not cited.

| File | Data | Reportable |
|---|---|---|
| [`latest-real.json`](latest-real.json) | BIDMC, 10 records | **yes** |
| [`latest.json`](latest.json) | synthetic waveform | no |

## Measured result

Run `2026-09-06T23:56:50+00:00` · sources in [`../docs/DATA.md`](../docs/DATA.md)

| | |
|---|---|
| Median difference vs monitor | **0.084 bpm** |
| Mean absolute difference | 1.593 bpm |
| Worst absolute difference | 6.372 bpm |
| Records within 5 bpm | **9 of 10** |
| Accept rate range | **2.1% – 100%** |

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

## Withheld

- Artifact rejection precision / recall — BIDMC does not annotate artifact locations, so rejection precision and recall have no answer key on this dataset
- Pressure in mmHg — PLETH is a photoplethysmogram, not an arterial line; a systolic pressure in mmHg is not something this signal measures

## Run log

| Run | Date | Scale | Command |
|---|---|---|---|
| real | 2026-09-06 | 10 records, 480 s each | `python -m src.demo --real` |

## What this does *not* establish

- Artifact-detection accuracy. There is no answer key on this dataset.
- Anything about blood pressure. The channel is a photoplethysmogram.
- Generalisation beyond 10 records from one device.
