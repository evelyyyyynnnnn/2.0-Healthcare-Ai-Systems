# Data — Clinical Empathy Analysis

| Field | Value |
|---|---|
| Source | MTS-Dialog (public research dataset) |
| Transcripts scored | 1,076 |
| Skipped as too short | 225 |
| Failed to parse | 0 |
| Clinician empathy ratings available | **False** |
| Licence | Public research dataset, used as published |
| Run recorded | 2026-09-06T23:54:23+00:00 |

## The files

| File | Rows | Transcripts | SHA-256 (prefix) |
|---|---|---|---|
| `mts-dialog/MTS-Dialog-TrainingSet.csv` | 1201 | 997 | `65a28681dd59fc15` |
| `mts-dialog/MTS-Dialog-ValidationSet.csv` | 100 | 79 | `227206520c453438` |

Retrieved 2026-09-06T23:54:23+00:00. These rows are the `files` array of
`results/latest-real.json`, written by the run itself.

## Why this corpus

MTS-Dialog is real clinical dialogue that was collected for something else
entirely — note-section annotation. That is exactly what makes it useful here:
nobody wrote these transcripts with an empathy lexicon in mind, so the scorer
cannot be flattered by them.

## What this corpus cannot tell you

MTS-Dialog is annotated for clinical note sections, not for communication quality. Correlation with clinician-assigned empathy ratings cannot be computed because no such ratings exist for these conversations.

This is the reason `validity_reported` is `false` in the results file. A number
is available; it would not mean what a reader would take it to mean, so it is
not published.

## Reproduction

```bash
pip install -r requirements.txt
python -m pytest tests/ -q     # 32 tests
python -m data.fetch           # needs the sources in docs/DATA.md
python -m src.demo --real      # the measured run -> results/latest-real.json
```
