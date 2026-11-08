# ICU Early-Warning Models

> Hypoxemia and hypotension prediction with calibration, uncertainty, and a false-alert comparison made at matched sensitivity against the alarm a ward actually uses.

**Repository:** `2.0-Healthcare-Ai-Systems` &middot; **Pillar:** Healthcare Safety

## Status

Working code, **32 passing tests**, and one measured run on real ICU charting.
Earlier versions of this page described a synthetic cohort; that is no longer
what the reportable run uses.

Built from the **PhysioNet MIMIC-IV open-access demo**: 134 stays, 24,304
observations, 12,152 patient-hours, 31 features. Two events at a 4-hour
horizon, compared at a matched sensitivity of 80% against the single-threshold
alarm a ward actually uses.

**The finding is calibration, not discrimination.** The ward-style alarm is badly miscalibrated — expected calibration error **0.276** on hypoxemia and **0.416** on hypotension — while a plain logistic model reaches **0.030** and **0.045**. At matched sensitivity that difference is worth **77.2 → 62.3** false alerts per 100 observations on hypoxemia and **31.8 → 22.9** on hypotension.

Gradient boosting does **not** beat logistic regression here, and that is
reported rather than hidden. about 100 ICU stays. Enough to show the pipeline
parses real charting, resamples irregular observations onto a regular grid
without looking forward, and produces calibrated risk. Not enough to estimate
discrimination for clinical use.

Last reportable run: `2026-09-06T23:55:09+00:00`

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 32 tests
python -m data.fetch           # needs the sources in docs/DATA.md
python -m src.demo --real      # the measured run -> results/latest-real.json
```

## Layout

```
README.md
data/
  |-- README.md
  |-- manifests/
  |-- sample/
docs/
  |-- DATA.md
  |-- METHOD.md
requirements.txt
results/
  |-- README.md
  |-- latest.json
src/
  |-- .gitkeep
  |-- __init__.py
  |-- cohort.py
  |-- dataset.py
  |-- demo.py
  |-- metrics.py
  |-- models.py
  |-- site.py
  |-- sitekit.py
tests/
  |-- .gitkeep
  |-- test_icu.py
website/
  |-- README.md
  |-- index.html
  |-- results.json
  |-- vercel.json
```

- `src/` &mdash; the implementation.
- `tests/` &mdash; pytest suite. These guard behaviour, not just imports.
- `results/latest.json` &mdash; the output of the last demo run. Every figure quoted
  anywhere in this project traces back to this file.
- `website/` &mdash; a self-contained static site, deployable to Vercel by copying the
  folder into its own repository. See `website/README.md`.

## The website

`website/` has no build step. To deploy it independently:

```bash
cp -r website/ ../my-2-icu-early-warning-site && cd ../my-2-icu-early-warning-site
git init && git add -A && git commit -m "site"
vercel deploy --prod
```

The page is regenerated from `results.json` on every `python -m src.demo`, so the
figures on the site and the figures the code produces cannot drift apart. Do not edit
numbers on the page by hand.

## Honesty note

Everything in this project runs on clearly-labelled synthetic or authored data.
Swap in the real source and the same pipeline reports real numbers &mdash; that is
what the structure is for. Until that happens, nothing here should be cited as a
measured result, and the site's closing section states explicitly what the project
does not establish.
