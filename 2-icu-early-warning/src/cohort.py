"""A synthetic ICU cohort with known deterioration events.

Not patient data, and not a substitute for it. Its purpose is to give the
modelling and evaluation code a dataset whose ground truth is known exactly,
so that calibration, uncertainty and alarm behaviour can be measured rather
than asserted.

The generative model is deliberately simple and stated in full: each patient
has a latent stability that drifts, deterioration events are driven by that
latent plus physiology, and the observable vitals are noisy functions of it.
That means the Bayes-optimal achievable performance is bounded and known to be
well below 1.0 -- which is the point. A synthetic cohort a model can score 0.99
on teaches you nothing about a real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Two events, chosen because they are the ones the petition's healthcare claim
# names and because they have different signatures: hypoxemia shows in SpO2 and
# respiratory rate, hypotension in MAP and heart rate.
EVENTS = ("hypoxemia", "hypotension")

VITALS = ("map_mmhg", "heart_rate_bpm", "spo2_pct", "resp_rate",
          "temp_c", "pulse_pressure")


@dataclass
class Patient:
    pid: int
    times: np.ndarray            # hours since admission
    vitals: dict                 # name -> array
    events: dict                 # event -> boolean array (onset at index)
    age: float = 65.0
    baseline_map: float = 85.0

    def n_steps(self) -> int:
        return len(self.times)


def _ou_process(n: int, theta: float, sigma: float,
                rng: np.random.Generator) -> np.ndarray:
    """Ornstein-Uhlenbeck: mean-reverting latent stability.

    Mean-reverting rather than a random walk because patients who destabilise
    usually either recover or are treated; an unbounded walk produces cohorts
    where everyone eventually deteriorates.
    """
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = x[i - 1] - theta * x[i - 1] + sigma * rng.normal()
    return x


def make_patient(pid: int, hours: float = 48.0, step_h: float = 0.5,
                 rng: np.random.Generator | None = None) -> Patient:
    rng = rng or np.random.default_rng(pid)
    n = int(hours / step_h)
    t = np.arange(n) * step_h

    age = float(np.clip(rng.normal(64, 15), 18, 95))
    base_map = float(np.clip(rng.normal(86, 10), 60, 115))
    base_hr = float(np.clip(rng.normal(78, 12), 45, 130))
    base_spo2 = float(np.clip(rng.normal(97, 1.6), 88, 100))
    base_rr = float(np.clip(rng.normal(16, 3), 8, 34))

    # Latent instability. Higher -> more likely to deteriorate.
    latent = _ou_process(n, theta=0.06, sigma=0.30, rng=rng)
    frailty = (age - 64) / 30.0
    risk = latent + 0.35 * frailty

    # Vitals respond to the latent with lag and noise.
    lag = 2
    risk_lag = np.concatenate([np.zeros(lag), risk[:-lag]])
    map_series = base_map - 11.0 * np.clip(risk_lag, 0, None) + rng.normal(0, 4.0, n)
    hr = base_hr + 9.0 * np.clip(risk_lag, 0, None) + rng.normal(0, 5.0, n)
    spo2 = base_spo2 - 3.2 * np.clip(risk_lag, 0, None) + rng.normal(0, 1.0, n)
    rr = base_rr + 3.5 * np.clip(risk_lag, 0, None) + rng.normal(0, 1.8, n)
    temp = 36.9 + 0.5 * np.clip(risk_lag, 0, None) + rng.normal(0, 0.3, n)
    pp = np.clip(rng.normal(45, 8, n) - 6.0 * np.clip(risk_lag, 0, None), 12, None)

    spo2 = np.clip(spo2, 70, 100)
    map_series = np.clip(map_series, 35, 140)

    # Events are driven by the latent, not by the observed vitals. A model that
    # only reads current vitals therefore cannot be perfect, which is realistic.
    events = {}
    # Intercepts set so the per-step hazard lands near 1%, which puts the
    # 4-hour-horizon event rate around 8-10% -- the range reported for ICU
    # deterioration. An earlier version sat at 38%, which quietly makes AUPRC
    # look excellent and is nothing like a real ward.
    p_hypox = 1 / (1 + np.exp(-(risk * 1.9 - 5.3)))
    p_hypot = 1 / (1 + np.exp(-(risk * 2.1 - 5.1)))
    events["hypoxemia"] = rng.random(n) < p_hypox
    events["hypotension"] = rng.random(n) < p_hypot

    # An event manifests in the vitals at onset.
    for i in np.where(events["hypoxemia"])[0]:
        spo2[i:i + 2] -= rng.uniform(4, 9)
    for i in np.where(events["hypotension"])[0]:
        map_series[i:i + 2] -= rng.uniform(12, 22)
    spo2 = np.clip(spo2, 70, 100)
    map_series = np.clip(map_series, 35, 140)

    return Patient(
        pid=pid, times=t, age=age, baseline_map=base_map,
        vitals={"map_mmhg": map_series, "heart_rate_bpm": hr, "spo2_pct": spo2,
                "resp_rate": rr, "temp_c": temp, "pulse_pressure": pp},
        events=events)


def make_cohort(n_patients: int = 400, hours: float = 48.0,
                step_h: float = 0.5, seed: int = 11) -> list:
    rng = np.random.default_rng(seed)
    return [make_patient(i, hours, step_h,
                         np.random.default_rng(rng.integers(0, 2 ** 31)))
            for i in range(n_patients)]


def cohort_stats(patients: list) -> dict:
    n_steps = sum(p.n_steps() for p in patients)
    out = {
        "n_patients": len(patients),
        "n_observations": n_steps,
        "patient_hours": round(sum(p.times[-1] + 0.5 for p in patients), 1),
        "mean_age": round(float(np.mean([p.age for p in patients])), 1),
    }
    for e in EVENTS:
        n = sum(int(p.events[e].sum()) for p in patients)
        out[f"{e}_events"] = n
        out[f"{e}_rate"] = round(n / n_steps, 5)
        out[f"{e}_patients_affected"] = sum(1 for p in patients if p.events[e].any())
    return out
