# Clinical Empathy Analysis

> Measuring empathic behaviour in bilingual consultation transcripts by fusing a theory-grounded weighted lexicon with structural features of the conversation.

**Repository:** `2.0-Healthcare-Ai-Systems` &middot; **Pillar:** Healthcare Safety

## Status

This is working code with a runnable demo and 32 tests. It is **not** a
finished result.

The correlation on this page is CIRCULAR and must not be read as validity. The same person wrote the transcripts and the lexicon, so the transcripts contain the cues the lexicon looks for. What is demonstrated is that the pipeline is internally consistent and that the evaluation protocol is sound — not that the measure tracks clinical empathy.

Last run: `2026-08-31T18:32:12+00:00`

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 32 tests
python -m src.demo             # runs everything, rewrites results/ and website/
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
  |-- EVIDENCE.md
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
cp -r website/ ../my-clinical-empathy-analysis-site && cd ../my-clinical-empathy-analysis-site
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
