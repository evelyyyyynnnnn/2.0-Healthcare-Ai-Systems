"""Pull MTS-Dialog -- real doctor-patient consultation transcripts.

    python -m data.fetch --list
    python -m data.fetch
    python -m data.fetch --verify

MTS-Dialog is a public research dataset of clinical conversations, released on
GitHub with no access gate. It gives this project the one thing an authored
corpus cannot: consultation language nobody wrote to be scored.

What it does NOT give is empathy ratings. MTS-Dialog is annotated for clinical
note sections, not for communication quality, and there is no public corpus of
consultations carrying clinician-assigned empathy scores that can be downloaded
without an ethics approval. So the real run reports how the instrument BEHAVES
on real speech -- score distribution, which cues fire, how much of the
vocabulary is covered -- and does not report validity, because validity means
correlation with human ratings and there are none here.

That is the honest state of this project: the instrument runs on real clinical
language; whether its scores agree with clinicians remains unmeasured.
"""
from __future__ import annotations

import pathlib
import sys

from .datakit import Fetcher, FetchError, NetworkBlocked, Source

ROOT = pathlib.Path(__file__).resolve().parent

REPO = "abachaa/MTS-Dialog"
REFS = ("main", "master")
RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"

FILES = [
    ("Main-Dataset/MTS-Dialog-TrainingSet.csv", "1,200 consultation excerpts"),
    ("Main-Dataset/MTS-Dialog-ValidationSet.csv", "a held-out split"),
]


def sources(ref: str) -> list:
    return [Source(name=f"MTS-Dialog {pathlib.Path(p).name}",
                   url=RAW.format(repo=REPO, ref=ref, path=p),
                   dest=f"mts-dialog/{pathlib.Path(p).name}",
                   publisher="MTS-Dialog (abachaa/MTS-Dialog)",
                   terms="public research dataset; see the repository's licence",
                   note=note)
            for p, note in FILES]


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args(argv)
    f = Fetcher(ROOT)

    if args.list:
        for s in sources(REFS[0]):
            print(f"{s.name}\n  {s.url}\n  {s.note}")
        print("\nno empathy ratings in this dataset: the real run reports how "
              "the instrument behaves, not whether it is valid")
        return 0
    if args.verify:
        problems = f.verify()
        for p in problems:
            print("  " + p)
        print("VERIFICATION FAILED" if problems else
              f"all {len(f.load_manifest()['files'])} cached file(s) verified")
        return 1 if problems else 0

    errors = []
    for ref in REFS:
        try:
            print(f"fetching MTS-Dialog from ref '{ref}' ...")
            f.get_all(sources(ref), refresh=args.refresh)
            print(f"\nwrote {f.manifest_path}")
            print("run `python -m src.demo --real` to score real consultations")
            return 0
        except NetworkBlocked as e:
            print(f"\nBLOCKED: {e}", file=sys.stderr)
            return 2
        except FetchError as exc:
            errors.append(f"{ref}: {exc}")
    print("\nFAILED on every known ref:\n  " + "\n  ".join(errors), file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
