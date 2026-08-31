"""Load BIDMC recordings and the monitor's own heart rate.

The monitor numerics are what make this worth doing. On the synthetic corpus
the pipeline is scored against artifacts the corpus placed itself, which proves
the detector finds what was put there. Here the bedside monitor -- an
independent device, running its own algorithm on the same patient -- reports a
heart rate once per second. Comparing the pipeline's derived rate against that
is an external check, and it can fail.

Two limits, stated rather than buried:

  Artifact locations are not annotated in BIDMC, so artifact-rejection
  precision and recall have no answer key here and are not reported.

  PLETH is a photoplethysmogram, not an arterial pressure line. Beat timing
  transfers; absolute pressures do not, so SBP and DBP in mmHg are meaningless
  on this signal and are suppressed rather than printed in units the data
  does not carry.
"""
from __future__ import annotations

import csv
import io
import pathlib

import numpy as np

from .datakit import Fetcher, FetchError

ROOT = pathlib.Path(__file__).resolve().parent
FS = 125.0
PROJECT = "bidmc"

# BIDMC column headers carry a leading space and a unit suffix.
PLETH_KEYS = ("pleth", "ii", "v", "avr", "resp")
HR_KEYS = ("hr",)
PULSE_KEYS = ("pulse",)


def _norm(name: str) -> str:
    return name.strip().lower().replace(" ", "").replace("[", "").replace("]", "")


def read_signals(raw: bytes, column: str = "pleth"):
    """Return the named channel from a BIDMC *_Signals.csv as a float array."""
    text = raw.decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        raise ValueError("signals file has no data rows")
    header = [_norm(h) for h in rows[0]]
    want = _norm(column)
    idx = next((i for i, h in enumerate(header) if h.startswith(want)), None)
    if idx is None:
        raise ValueError(f"channel {column!r} not in {header}")
    vals = []
    for r in rows[1:]:
        if idx >= len(r):
            continue
        s = r[idx].strip()
        if not s or s == "-":
            continue
        try:
            vals.append(float(s))
        except ValueError:
            continue
    if len(vals) < int(FS * 10):
        raise ValueError(f"only {len(vals)} samples in {column!r}; "
                         f"need at least 10 seconds")
    return np.asarray(vals, dtype=float)


def read_numerics(raw: bytes) -> dict:
    """Return {channel: array} from a BIDMC *_Numerics.csv (1 Hz)."""
    text = raw.decode("utf-8", errors="replace")
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        raise ValueError("numerics file has no data rows")
    header = [_norm(h) for h in rows[0]]
    out = {h: [] for h in header}
    for r in rows[1:]:
        for i, h in enumerate(header):
            if i >= len(r):
                out[h].append(np.nan)
                continue
            s = r[i].strip()
            try:
                out[h].append(float(s))
            except ValueError:
                out[h].append(np.nan)   # the monitor writes '-' when unavailable
    return {h: np.asarray(v, dtype=float) for h, v in out.items()}


def reference_rate(numerics: dict):
    """The monitor's own pulse rate, preferring PULSE over ECG-derived HR.

    PULSE is derived from the same plethysmogram the pipeline reads, so it is
    the like-for-like comparison; HR comes from the ECG and is the fallback.
    """
    for keys in (PULSE_KEYS, HR_KEYS):
        for h, arr in numerics.items():
            if any(h.startswith(k) for k in keys):
                finite = arr[np.isfinite(arr)]
                if len(finite) >= 10:
                    return arr, h
    return None, None


def load_records(root=ROOT, channel: str = "pleth"):
    """Return (records, provenance). Refuses when nothing real is cached."""
    f = Fetcher(root)
    man = f.load_manifest()
    sig_files = sorted(k for k in man["files"] if k.endswith("_Signals.csv"))
    if not sig_files:
        raise FetchError(
            "no real waveform recordings cached. Run `python -m data.fetch` in "
            "a networked environment first; this pipeline will not report "
            "synthetic segments as if they were recorded from patients.")

    records, prov = [], []
    for dest in sig_files:
        rid = pathlib.Path(dest).name.split("_")[1]
        num_dest = dest.replace("_Signals.csv", "_Numerics.csv")
        try:
            sig = read_signals((f.raw / dest).read_bytes(), channel)
        except (ValueError, OSError) as exc:
            prov.append({"record": rid, "status": f"unusable: {exc}"})
            continue

        ref, ref_channel = None, None
        if (f.raw / num_dest).exists():
            try:
                ref, ref_channel = reference_rate(
                    read_numerics((f.raw / num_dest).read_bytes()))
            except ValueError:
                ref = None

        records.append({"record": rid, "signal": sig, "fs": FS,
                        "reference_rate": ref, "reference_channel": ref_channel})
        rec = man["files"][dest]
        prov.append({
            "record": rid, "status": "ok", "channel": channel,
            "n_samples": int(len(sig)),
            "duration_s": round(len(sig) / FS, 1),
            "has_monitor_reference": ref is not None,
            "reference_channel": ref_channel,
            "sha256": rec["sha256"][:16], "url": rec["url"],
            "retrieved_utc": rec.get("retrieved_utc"),
        })

    if not records:
        raise FetchError("no BIDMC recording could be read from the cache")
    return records, {
        "database": "PhysioNet BIDMC PPG and Respiration Dataset (open access)",
        "channel": channel, "fs_hz": FS,
        "n_records": len(records),
        "signal_is_ppg_not_arterial_line": True,
        "artifact_locations_annotated": False,
        "artifact_scoring_withheld_because":
            "BIDMC does not annotate artifact locations, so rejection precision "
            "and recall have no answer key on this dataset",
        "records": prov,
    }


def to_segments(records):
    """Wrap loaded records in the SynthSegment shape the pipeline consumes."""
    from src.synth import SynthSegment
    return [SynthSegment(signal=r["signal"], fs=r["fs"],
                         label=f"bidmc_{r['record']}")
            for r in records]
