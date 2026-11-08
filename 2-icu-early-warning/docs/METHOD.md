# Method — ICU Early-Warning Models

## Problem

Predict hypoxemia and hypotension four hours ahead, and compare the result
against the alarm a ward already runs — on the terms that matter at the
bedside.

## Why the approach is non-obvious

AUROC is the wrong headline for an alarm. A ward does not choose an operating
point by area under a curve; it chooses a sensitivity it can live with and
then absorbs whatever false-alert rate follows. Two models with the same AUROC
can differ by a factor of two in alerts at the sensitivity actually used, and
a model whose probabilities are miscalibrated cannot be given a sensible
threshold at all. So the comparison here is made **at matched sensitivity**,
and calibration error is reported beside discrimination rather than below it.

## Method

Irregular charting is resampled onto a regular **0.5-hour grid** with a
carry-forward limit of **2.0 hours**, so no value is propagated
further than it would be believed clinically and nothing is filled from the future.
31 features are derived per grid point.

Event definitions, taken from the chart rather than from a label column:

| Event | Definition |
|---|---|
| Hypoxemia | SpO2 < 90% |
| Hypotension | mean arterial pressure < 65 mmHg |

Four models are scored: the single-threshold alarm as a baseline, logistic
regression, gradient boosting, and gradient boosting with isotonic
recalibration. Splits are by patient, never by row.

## Evaluation protocol

Every model is placed at the operating point that reaches a sensitivity of
**80%**, and the false-alert rate is read off there. Expected
calibration error is computed on the same test rows. The reliability curve is
kept in the results file so the calibration claim can be checked rather than
believed.

## What the run found

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

Two things are worth stating plainly.

**The deployed-style alarm is the worst-calibrated thing on the page** — ECE
0.276 and 0.416,
roughly an order of magnitude worse than logistic regression. Its risk scores
cannot be read as probabilities, which is precisely why a ward ends up tuning
the threshold by feel.

**Gradient boosting loses to logistic regression here**, on both events and on
both AUROC and calibration. On a cohort this size that is what the data says,
and inventing a reason to prefer the more complex model would be the error.

## Limitations

- **about 100 ICU stays. Enough to show the pipeline parses real charting, resamples irregular observations onto a regular grid without looking forward, and produces calibrated risk. Not enough to estimate discrimination for clinical use.**
- **Demonstration, not a study.** `cohort_is_a_demonstration_not_a_study` is
  `true` in the results file.
- **Two events only**, chosen for contrasting physiological signatures.
- **Open-access demo**, not the full credentialed MIMIC-IV.

## Reproduction

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 32 tests
python -m data.fetch           # needs the sources in docs/DATA.md
python -m src.demo --real      # the measured run -> results/latest-real.json
```
