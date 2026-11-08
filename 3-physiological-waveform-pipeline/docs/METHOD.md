# Method — Physiological Waveform Pipeline

## Problem

Derive heart rate from a photoplethysmogram, decide which windows are clean
enough to use, and check the answer against the bedside monitor.

## Why the approach is non-obvious

Beat detection on a clean PPG is close to solved. What is not solved is
deciding *which windows are clean* — and that decision silently sets every
downstream number. A pipeline that reports excellent agreement while quietly
discarding most of its input has told you about its quality gate, not about its
detector. So the accept rate is reported per record, beside the agreement
figure, rather than averaged away.

## Method

The `pleth` channel at 125 Hz is cut into fixed windows. Each window
passes an artifact gate before any beat detection is attempted. The gate works
in arbitrary units rather than calibrated ones, with a step limit expressed
relative to each window's own dynamic range:

| Limit | Value |
|---|---|
| Units | arbitrary |
| Lower bound | -inf |
| Upper bound | inf |
| Step limit | 60% of the window's own peak-to-peak |

Heart rate is derived from accepted windows only and compared against the
monitor's own reported rate on the same record.

## Evaluation protocol

The monitor's reported heart rate is the reference. Agreement is summarised by
the median and the worst case, not the mean alone, because a single bad record
would otherwise be hidden.

## What the run found

Across 10 records:

| | |
|---|---|
| Median difference | **0.084 bpm** |
| Mean absolute difference | 1.593 bpm |
| Worst absolute difference | 6.372 bpm |
| Records within 5 bpm | **9 of 10** |

Per record:

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

**The accept rate is the finding.** It ranges from 2.1% to 100% across ten
records of the same length from the same device. Record 01 contributes
1 usable window out of 48, and still lands within
2.52 bpm of the monitor. An agreement figure computed without
publishing that number would be close to meaningless.

## What is deliberately not reported

- **Artifact rejection precision and recall.** BIDMC does not annotate artifact locations, so rejection precision and recall have no answer key on this dataset
- **Blood pressure in mmHg.** PLETH is a photoplethysmogram, not an arterial line; a systolic pressure in mmHg is not something this signal measures

Both are easy numbers to produce and neither would mean what a reader would
take it to mean.

## Limitations

- **10 records**, one dataset, one device.
- **PPG only** — `signal_is_ppg_not_arterial_line` is `true` in the results file.
- The gate is tuned to be conservative; a looser gate would raise the accept
  rate and is untested here.

## Reproduction

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 33 tests
python -m data.fetch           # needs the sources in docs/DATA.md
python -m src.demo --real      # the measured run -> results/latest-real.json
```
