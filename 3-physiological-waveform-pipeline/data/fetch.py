"""Pull the BIDMC PPG and Respiration dataset -- real recorded waveforms.

    python -m data.fetch --list
    python -m data.fetch
    python -m data.fetch --verify

Why this dataset and not MIMIC-IV Waveform: BIDMC is open access. MIMIC-IV
Waveform needs a credentialed PhysioNet account and CITI human-subjects
training, which means nobody can reproduce a result from it by reading this
repository. BIDMC is 53 eight-minute recordings from adult ICU patients,
released without a gate.

What it is: photoplethysmogram (PLETH) at 125 Hz, plus respiration and ECG,
and -- the part that makes it useful here -- a per-second numerics file
carrying the bedside monitor's own heart rate, pulse rate and SpO2.

That numerics file is a genuine reference. Beat detection can be scored against
the monitor's reported rate rather than against the pipeline's own assumptions.
Artifact LOCATIONS are still unannotated in BIDMC, so artifact-rejection
precision cannot be scored here the way it can on the synthetic corpus, and it
is reported as unavailable rather than estimated.

One caveat stated plainly: PPG is not an arterial line. The pulse morphology is
similar enough that the same quality checks apply, but a pressure in mmHg is
not what PLETH measures, so absolute SBP/DBP figures are meaningless on it.
"""
from __future__ import annotations

import pathlib
import sys

from .datakit import Fetcher, FetchError, NetworkBlocked
from .physionet import bidmc

ROOT = pathlib.Path(__file__).resolve().parent

# A spread of recordings rather than the first few, so the sample is not all
# from consecutively-admitted patients.
RECORDS = [1, 5, 11, 17, 23, 29, 35, 41, 47, 53]


def sources() -> list:
    out = []
    for n in RECORDS:
        rid = f"{n:02d}"
        out.append(bidmc(f"bidmc_csv/bidmc_{rid}_Signals.csv",
                         "125 Hz PLETH, RESP and ECG"))
        out.append(bidmc(f"bidmc_csv/bidmc_{rid}_Numerics.csv",
                         "1 Hz monitor-reported HR, PULSE and SpO2 -- the reference"))
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    f = Fetcher(ROOT)
    srcs = sources()

    if args.list:
        for s in srcs[:4]:
            print(f"{s.name}\n  {s.url}\n  {s.note}")
        print(f"...\n\n{len(srcs)} files: {len(RECORDS)} recordings "
              f"(signals + numerics), records {RECORDS}")
        return 0
    if args.verify:
        problems = f.verify()
        for p in problems:
            print("  " + p)
        print("VERIFICATION FAILED" if problems else
              f"all {len(f.load_manifest()['files'])} cached file(s) verified")
        return 1 if problems else 0

    print(f"fetching {len(srcs)} files from the BIDMC dataset ...")
    try:
        f.get_all(srcs, refresh=args.refresh)
    except NetworkBlocked as e:
        print(f"\nBLOCKED: {e}", file=sys.stderr)
        return 2
    except FetchError as e:
        print(f"\nFAILED: {e}", file=sys.stderr)
        return 1
    print(f"\nwrote {f.manifest_path}")
    print("run `python -m src.demo --real` to process real recordings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
