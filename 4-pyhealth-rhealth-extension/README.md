# icuflow

> An installable package for the three parts of a clinical prediction pipeline that are easiest to get wrong: subject-grouped splits, calibration, and alarm comparison at matched sensitivity.

**Repository:** `2.0-Healthcare-Ai-Systems` &middot; **Pillar:** Healthcare Safety

## Status

Working code, **29 passing tests**, and one measured run on real ICU charting.
Earlier versions of this page said the experiments ran on synthetic subjects;
the reportable run does not.

`icuflow` 0.1.0 is an installable package with 14 exports and a working CLI
(`check-split`, `score`). It is **not published to PyPI** — `published` is
`false` in the results file — so it has no download count and no users, and
neither is claimed.

**What it measures.** Using the **PhysioNet MIMIC-IV open-access demo**, the same cohort, features and model are split two ways. A random **row-level** split reports **AUROC 0.9058**. A **subject-grouped** split reports **0.4989** — chance. The reported figure is inflated by **81.57%**, and the honest one says the model learned nothing.

Cohort: 97 subjects, 134 stays, 37 readmissions, 22,512 rows, positive rate
0.02838. about 100 ICU stays. The direction of the leakage effect is
meaningful; its magnitude, from a cohort this small, is not a population
estimate.

Last reportable run: `2026-09-06T23:57:16+00:00`

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 29 tests
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
icuflow/
  |-- __init__.py
  |-- alarms.py
  |-- calibration.py
  |-- cli.py
  |-- splits.py
  |-- tasks.py
icuflow.egg-info/
  |-- PKG-INFO
  |-- SOURCES.txt
  |-- dependency_links.txt
  |-- entry_points.txt
  |-- requires.txt
  |-- top_level.txt
pyproject.toml
requirements.txt
results/
  |-- README.md
  |-- latest.json
src/
  |-- .gitkeep
  |-- __init__.py
  |-- demo.py
  |-- site.py
  |-- sitekit.py
tests/
  |-- .gitkeep
  |-- test_icuflow.py
website/
  |-- README.md
  |-- index.html
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
cp -r website/ ../my-4-pyhealth-rhealth-extension-site && cd ../my-4-pyhealth-rhealth-extension-site
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
