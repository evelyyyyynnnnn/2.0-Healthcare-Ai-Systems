# Clinical Empathy Analysis

> Measuring empathic behaviour in bilingual consultation transcripts by fusing a theory-grounded weighted lexicon with structural features of the conversation.

**Repository:** `2.0-Healthcare-Ai-Systems` &middot; **Pillar:** Healthcare Safety

## Status

Working code, **32 passing tests**, and one measured run on real consultation
transcripts. The result is a **negative** one and is reported as such.

Scored against **1,076 real doctor–patient transcripts** from MTS-Dialog. The
lexicon fires no cue at all on **731 of them (67.9%)**, the median score
across the corpus is **0.0**, and only **25 distinct cues** in the lexicon
ever fire.

**Validity is withheld, not reported.** MTS-Dialog is annotated for clinical note sections, not for communication quality. Correlation with clinician-assigned empathy ratings cannot be computed because no such ratings exist for these conversations. A correlation is not computed here because there is nothing honest to correlate against.

Last reportable run: `2026-09-06T23:54:23+00:00`

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
  |-- corpus.py
  |-- demo.py
  |-- evaluate.py
  |-- lexicon.py
  |-- scoring.py
  |-- site.py
  |-- sitekit.py
tests/
  |-- .gitkeep
  |-- test_empathy.py
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
cp -r website/ ../my-1-clinical-empathy-analysis-site && cd ../my-1-clinical-empathy-analysis-site
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
