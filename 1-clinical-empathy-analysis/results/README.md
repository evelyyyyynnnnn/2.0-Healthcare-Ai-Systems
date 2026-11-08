# Results — Clinical Empathy Analysis

This file is the only source for the numbers quoted elsewhere in this
repository; if a figure is not here with a run date, it is not cited.

| File | Data | Reportable |
|---|---|---|
| [`latest-real.json`](latest-real.json) | 1,076 MTS-Dialog transcripts | **yes** |
| [`latest.json`](latest.json) | authored transcripts | no — smoke test only |

## Measured result

Run `2026-09-06T23:54:23+00:00` · sources in [`../docs/DATA.md`](../docs/DATA.md)

| | |
|---|---|
| Transcripts scored | 1,076 |
| Firing **no cue at all** | **731 (67.9%)** |
| Median score | 0.0 |
| Mean score | 0.5755 |
| Distinct cues fired | 25 |
| Median doctor token share | 0.522 |
| Median question rate | 0.838 |

## Validity

**Not reported.** MTS-Dialog is annotated for clinical note sections, not for communication quality. Correlation with clinician-assigned empathy ratings cannot be computed because no such ratings exist for these conversations.

## Run log

| Run | Date | Scale | Command |
|---|---|---|---|
| real | 2026-09-06 | 1,076 transcripts | `python -m src.demo --real` |

## What this does *not* establish

- That the score tracks clinical empathy. No rating exists on this corpus to check it against.
- Anything about clinicians. The finding is about the lexicon's coverage of real speech.
- Anything outside English-language consultation transcripts.
