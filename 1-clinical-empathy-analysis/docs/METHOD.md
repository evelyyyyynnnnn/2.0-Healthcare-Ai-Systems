# Method — Clinical Empathy Analysis

## Problem

Score empathic behaviour in a consultation transcript, and be honest about
whether that score means anything.

## Why the approach is non-obvious

A lexicon is the transparent choice and an LLM judge is the fashionable one, but
the hard part is neither: it is validity. A score correlates with whatever
produced the text, so a lexicon scored against transcripts written by the same
hand will always look excellent. The only way to learn something is to run it on
text nobody wrote for it and report what happens — including the possibility
that it fires on almost nothing.

## Method

A theory-grounded weighted lexicon of empathic cues, fused with two structural
features of the conversation. Each transcript yields a score; the distribution
of scores across the corpus is the object of study, not any single score.

Structural features measured on this corpus:

| Feature | Median |
|---|---|
| Doctor share of tokens | 0.522 |
| Question rate | 0.838 |

## Evaluation protocol

There is no train/test split because nothing is fitted. The corpus is scored
once, in full, and the distribution is reported with the silent share stated
first, because that is the number most likely to be omitted elsewhere.

## What the run found

On 1,076 MTS-Dialog transcripts:

| | |
|---|---|
| Transcripts firing **no cue at all** | **731 (67.9%)** |
| Median score | 0.0 |
| Mean score | 0.5755 |
| Range | -3.5714 to 17.0 |
| Q1 / Q3 | 0.0 / 0.4086 |
| Distinct cues that ever fired | 25 |

**The finding is the silence.** A lexicon built from the communication-skills
literature is mute on roughly two thirds of real consultations. That is a
result about the method, not about the clinicians: real consultation language
does not look like the language the literature uses to describe empathy.

The ten most frequent cues account for most of what does fire:

- `just` — 77
- `please` — 54
- `sorry to hear` — 51
- `how are you feeling` — 45
- `difficult` — 44
- `we can` — 39
- `anything else` — 37
- `tell me more` — 25
- `i understand` — 22
- `thank you for` — 18

## Limitations

- **Validity is not established and is deliberately not claimed.**
  MTS-Dialog is annotated for clinical note sections, not for communication quality. Correlation with clinician-assigned empathy ratings cannot be computed because no such ratings exist for these conversations.
- **Lexical only.** Paraphrased empathy that avoids the listed cues scores zero.
- **English only** on this corpus.
- **225 transcripts were skipped** as too short; none failed to parse.

## Reproduction

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 32 tests
python -m data.fetch           # needs the sources in docs/DATA.md
python -m src.demo --real      # the measured run -> results/latest-real.json
```
