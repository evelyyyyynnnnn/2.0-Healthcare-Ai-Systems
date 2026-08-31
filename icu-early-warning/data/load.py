"""Build Patient records from real MIMIC-IV charted vitals.

The resampling rule lives in mimicvitals, shared with the other ICU projects.
What is specific here is what the records become and what the labels mean:

  Events are defined from the same vitals used as features. In the simulated
  cohort a latent instability drives events, so a model reading vitals cannot
  be perfect. Here hypoxemia IS SpO2 below 90. The task stays honest only
  because features come from time <= t, the label from a strictly later window,
  and rows already in the event are dropped -- but a reader should know the
  label is a threshold on an observed signal, not an independent outcome.
"""
from __future__ import annotations

import pathlib

import numpy as np

from .mimicvitals import (CARRY_FORWARD_H, EVENT_DEFINITIONS, ITEMS, PLAUSIBLE,
                          PROJECT, STEP_H, events_from_vitals, resample,
                          stay_series)

ROOT = pathlib.Path(__file__).resolve().parent


def load_cohort(root=ROOT, min_hours: float = 12.0, max_stays: int = 0):
    """Return (patients, provenance). Refuses when the real cache is empty."""
    from src.cohort import Patient

    stays, prov = stay_series(root, min_hours=min_hours, max_stays=max_stays)
    patients = [
        Patient(pid=s["stay_id"], times=s["times"], age=s["age"],
                baseline_map=float(np.median(s["vitals"]["map_mmhg"])),
                vitals=s["vitals"], events=events_from_vitals(s["vitals"]))
        for s in stays
    ]
    prov = dict(prov)
    prov["n_patients_built"] = prov.pop("n_stays_built")
    prov["event_definitions"] = EVENT_DEFINITIONS
    return patients, prov


# Re-exported so the tests and any reader can reach the resampling rule from
# the module they are already looking at.
_resample = resample
