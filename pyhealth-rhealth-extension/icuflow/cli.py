"""icuflow command line: check a split, or score a set of predictions."""

from __future__ import annotations

import argparse
import csv
import json
import sys

import numpy as np

from .alarms import compare_at_matched_sensitivity
from .calibration import ece, reliability_table
from .splits import assert_no_subject_leak, SubjectLeakError, split_by_subject


def _read_csv(path):
    with open(path, newline="", encoding="utf8") as fh:
        return list(csv.DictReader(fh))


def cmd_check_split(args) -> int:
    rows = _read_csv(args.csv)
    subj = np.array([r[args.subject_col] for r in rows])
    split = np.array([r[args.split_col] for r in rows])
    try:
        assert_no_subject_leak(subj, split == "train", split == "test")
    except SubjectLeakError as exc:
        print(f"LEAK: {exc}", file=sys.stderr)
        return 1
    print(f"ok: {len(np.unique(subj))} subjects, no overlap between splits")
    return 0


def cmd_score(args) -> int:
    rows = _read_csv(args.csv)
    y = np.array([int(float(r[args.label_col])) for r in rows])
    p = np.array([float(r[args.score_col]) for r in rows])
    out = {"n": len(y), "event_rate": float(y.mean()),
           "ece": ece(y, p), "reliability": reliability_table(y, p)}
    if args.baseline_col:
        b = np.array([float(r[args.baseline_col]) for r in rows])
        out["matched_sensitivity"] = compare_at_matched_sensitivity(
            y, b, p, args.target_sensitivity)
    json.dump(out, sys.stdout, indent=2)
    print()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="icuflow", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("check-split", help="fail if a subject spans train and test")
    a.add_argument("csv")
    a.add_argument("--subject-col", default="subject_id")
    a.add_argument("--split-col", default="split")
    a.set_defaults(fn=cmd_check_split)

    b = sub.add_parser("score", help="calibration and alarm metrics for predictions")
    b.add_argument("csv")
    b.add_argument("--label-col", default="label")
    b.add_argument("--score-col", default="score")
    b.add_argument("--baseline-col", default=None)
    b.add_argument("--target-sensitivity", type=float, default=0.80)
    b.set_defaults(fn=cmd_score)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
