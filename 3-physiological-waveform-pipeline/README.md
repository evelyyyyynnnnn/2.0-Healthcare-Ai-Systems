# Physiological Waveform Pipeline

> Signal-quality assessment, artifact rejection and feature extraction for high-frequency arterial waveforms — scored against known artifact locations rather than assumed to work.

**Repository:** `2.0-Healthcare-Ai-Systems` &middot; **Pillar:** Healthcare Safety

## Status

Working code, **33 passing tests**, and one measured run on real physiological
waveforms.

Run on the **PhysioNet BIDMC PPG and Respiration Dataset**: 10 records, 480 s
each, PPG sampled at 125 Hz.

**Heart rate derived from the photoplethysmogram agrees with the bedside monitor**: median difference **0.084 bpm**, mean absolute difference 1.593 bpm, worst case 6.372 bpm, and **9 of 10 records within 5 bpm**.

**The quality gate, not the beat detector, is where the variance lives.** Across the same 10 records the artifact gate accepts between **2.1% and 100%** of windows — record 01 yields 1 usable window out of 48. Two figures are deliberately **not** reported: bidmc does not annotate artifact locations, so rejection precision and recall have no answer key on this dataset, and pleth is a photoplethysmogram, not an arterial line; a systolic pressure in mmhg is not something this signal measures

Last reportable run: `2026-09-06T23:56:50+00:00`

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 33 tests
python -m data.fetch           # needs the sources in docs/DATA.md
python -m src.demo --real      # the measured run -> results/latest-real.json
```

## Layout

```
README.md
data/
  |-- README.md
  |-- derived_features.csv
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
  |-- demo.py
  |-- features.py
  |-- loaders.py
  |-- pipeline.py
  |-- quality.py
  |-- site.py
  |-- sitekit.py
  |-- synth.py
tests/
  |-- .gitkeep
  |-- test_pipeline.py
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
cp -r website/ ../my-3-physiological-waveform-pipeline-site && cd ../my-3-physiological-waveform-pipeline-site
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
