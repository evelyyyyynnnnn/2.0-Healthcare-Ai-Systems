"""Demonstrate what the package prevents, measured rather than asserted."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

import icuflow
from icuflow import (IsotonicCalibrator, PlattCalibrator,
                     compare_at_matched_sensitivity, ece, split_by_subject,
                     assert_no_subject_leak, group_kfold, horizon_label)
from icuflow.splits import SubjectLeakError

ROOT = pathlib.Path(__file__).resolve().parent.parent


def synth(n_subjects=200, steps=80, seed=5):
    """Subjects with a fingerprint and an idiosyncratic label propensity.

    Getting this right took a second attempt, and the first failure is the
    interesting part. Giving each subject an offset that both shifts a feature
    and shifts the label produces no leakage at all: the offset-to-label
    relationship is real and generalises, so a model learning it does just as
    well on unseen subjects. Row-splitting looked harmless.

    Leakage needs a subject effect that does NOT generalise. Here each subject
    gets a random label propensity unrelated to anything predictive, plus a
    near-noise-free fingerprint feature that identifies them. A model trained on
    rows can memorise fingerprint -> propensity and score well on held-out rows
    from subjects it has already seen. On held-out subjects the fingerprints are
    new and the memorised mapping is worthless.

    That is exactly the structure of a patient baseline in ICU data, which is
    why row-splitting inflates published clinical results.
    """
    rng = np.random.default_rng(seed)
    X, y, subj = [], [], []
    for s in range(n_subjects):
        fingerprint = rng.normal(0, 3.0)          # identifies the subject
        propensity = rng.normal(0, 1.6)           # idiosyncratic, non-generalising
        for _ in range(steps):
            f = rng.normal(0, 1, 5)
            f[0] = fingerprint + rng.normal(0, 0.02)
            logit = 0.5 * f[1] - 0.35 * f[2] + propensity - 1.2
            X.append(f)
            y.append(int(rng.random() < 1 / (1 + np.exp(-logit))))
            subj.append(s)
    return np.array(X), np.array(y), np.array(subj)


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    pos, neg = int(y.sum()), int((1 - y).sum())
    if not pos or not neg:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s), float); r[order] = np.arange(1, len(s) + 1)
    return float((r[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg))


def leakage_experiment() -> dict:
    """The number that justifies the splits module existing."""
    X, y, subj = synth()
    rng = np.random.default_rng(0)

    # wrong: split rows at random
    m = rng.permutation(len(X)) < int(0.7 * len(X))
    mdl = HistGradientBoostingClassifier(max_iter=120, random_state=0).fit(X[m], y[m])
    leaked = auroc(y[~m], mdl.predict_proba(X[~m])[:, 1])

    # right: split subjects
    sp = split_by_subject(subj, test_frac=0.3, seed=0)
    assert_no_subject_leak(subj, sp.train, sp.test)
    mdl2 = HistGradientBoostingClassifier(max_iter=120, random_state=0).fit(
        X[sp.train], y[sp.train])
    honest = auroc(y[sp.test], mdl2.predict_proba(X[sp.test])[:, 1])

    return {
        "row_split_auroc": round(leaked, 4),
        "subject_split_auroc": round(honest, 4),
        "inflation": round(leaked - honest, 4),
        "inflation_pct": round(100.0 * (leaked - honest) / honest, 2),
        "n_subjects": int(len(np.unique(subj))),
        "n_rows": int(len(X)),
    }


def _calibrate_case(name, distort, X, y, subj) -> dict:
    sp = split_by_subject(subj, 0.3, seed=1)
    mdl = HistGradientBoostingClassifier(max_iter=120, random_state=0).fit(
        X[sp.train], y[sp.train])

    # Calibrators are fitted on a held-out slice of TRAIN, never on test.
    inner = group_kfold(subj[sp.train], n_splits=3, seed=2)[0]
    cal_raw = distort(mdl.predict_proba(X[sp.train][inner.test])[:, 1])
    cal_y = y[sp.train][inner.test]
    raw = distort(mdl.predict_proba(X[sp.test])[:, 1])

    iso = IsotonicCalibrator().fit(cal_raw, cal_y)
    platt = PlattCalibrator().fit(cal_raw, cal_y)
    return {
        "case": name,
        "raw_ece": round(ece(y[sp.test], raw), 5),
        "isotonic_ece": round(ece(y[sp.test], iso.transform(raw)), 5),
        "platt_ece": round(ece(y[sp.test], platt.transform(raw)), 5),
        "raw_auroc": round(auroc(y[sp.test], raw), 4),
        "isotonic_auroc": round(auroc(y[sp.test], iso.transform(raw)), 4),
    }


def calibration_experiment() -> dict:
    """Two cases, because the honest answer is "it depends".

    Boosted trees trained on log loss come out close to calibrated already, and
    on those, calibration is a no-op that can add variance and make ECE slightly
    worse. The module earns its place on scores that are NOT probabilities --
    a margin, a rank, an overconfident output -- which is the common case in
    clinical code. Reporting only the flattering half would misrepresent when to
    reach for it.
    """
    X, y, subj = synth(seed=9)
    already = _calibrate_case(
        "already well calibrated (boosted trees, log loss)",
        lambda p: p, X, y, subj)
    # Overconfidence: push probabilities toward 0 and 1. Ranking is untouched,
    # so AUROC is identical and only calibration changes.
    over = _calibrate_case(
        "overconfident (monotone distortion, same ranking)",
        lambda p: np.clip(p ** 2.6 / (p ** 2.6 + (1 - p) ** 2.6), 1e-6, 1 - 1e-6),
        X, y, subj)
    return {"cases": [already, over]}


def alarm_experiment() -> dict:
    X, y, subj = synth(seed=13)
    sp = split_by_subject(subj, 0.3, seed=4)
    mdl = HistGradientBoostingClassifier(max_iter=120, random_state=0).fit(
        X[sp.train], y[sp.train])
    p = mdl.predict_proba(X[sp.test])[:, 1]
    baseline = X[sp.test][:, 1]         # one vital, the ward alarm
    cmp = compare_at_matched_sensitivity(y[sp.test], baseline, p, 0.80)
    return {
        "target_sensitivity": cmp["target_sensitivity"],
        "baseline_false_alerts": cmp["baseline"]["false_alerts"],
        "model_false_alerts": cmp["model"]["false_alerts"],
        "reduction_pct": round(cmp["false_alert_reduction_pct"], 2),
        "baseline_sensitivity": round(cmp["baseline"]["sensitivity"], 4),
        "model_sensitivity": round(cmp["model"]["sensitivity"], 4),
    }


def guard_experiment() -> dict:
    """The leak guard has to actually fire."""
    _, _, subj = synth(n_subjects=20, steps=10)
    bad_train = np.ones(len(subj), bool)
    bad_test = np.ones(len(subj), bool)
    caught = False
    try:
        assert_no_subject_leak(subj, bad_train, bad_test)
    except SubjectLeakError:
        caught = True
    active = horizon_label(np.array([0, 1, 0, 0, 0], bool), 1, 2)
    short = horizon_label(np.array([0, 0, 0], bool), 2, 3)
    return {
        "leak_detected": caught,
        "active_event_row_dropped": active is None,
        "insufficient_followup_dropped": short is None,
    }


def cli_check() -> dict:
    out = subprocess.run([sys.executable, "-m", "icuflow.cli", "--help"],
                         capture_output=True, text=True, timeout=60)
    return {"cli_exit": out.returncode,
            "subcommands": ["check-split", "score"],
            "help_ok": "check-split" in out.stdout}


def run() -> dict:
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_synthetic": True,
        "data_source": "synthetic subjects with a per-subject offset (src/demo.py)",
        "package": {"name": "icuflow", "version": icuflow.__version__,
                    "exports": len(icuflow.__all__),
                    "published": False},
        "leakage": leakage_experiment(),
        "calibration": calibration_experiment(),
        "alarms": alarm_experiment(),
        "guards": guard_experiment(),
        "cli": cli_check(),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "latest.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf8")
    return results


def leakage_experiment_real(root):
    """The same comparison, on real patients instead of constructed ones.

    The synthetic version guarantees inflation by construction. This one does
    not: whether a real patient's physiological baseline is memorable enough to
    inflate a row-split score is an empirical question, and the answer is
    whatever it is. Reporting it either way is the point.
    """
    import sys as _sys
    _sys.path.insert(0, str(ROOT))
    from data.load import load_xy

    X, y, subj, stay_ids, names, prov = load_xy(root=root)
    rng = np.random.default_rng(0)

    m = rng.permutation(len(X)) < int(0.7 * len(X))
    mdl = HistGradientBoostingClassifier(max_iter=120, random_state=0).fit(X[m], y[m])
    leaked = auroc(y[~m], mdl.predict_proba(X[~m])[:, 1])

    sp = split_by_subject(subj, test_frac=0.3, seed=0)
    assert_no_subject_leak(subj, sp.train, sp.test)
    mdl2 = HistGradientBoostingClassifier(max_iter=120, random_state=0).fit(
        X[sp.train], y[sp.train])
    honest = auroc(y[sp.test], mdl2.predict_proba(X[sp.test])[:, 1])

    # A patient readmitted to the ICU has two stay_ids and one baseline.
    readmitted = len(np.unique(stay_ids)) - len(np.unique(subj))

    return {
        "row_split_auroc": round(leaked, 4),
        "subject_split_auroc": round(honest, 4),
        "inflation": round(leaked - honest, 4),
        "inflation_pct": round(100.0 * (leaked - honest) / honest, 2)
                         if honest else None,
        "n_subjects": prov["n_subjects"],
        "n_stays": prov["n_stays_used"],
        "n_readmissions": int(readmitted),
        "n_rows": prov["n_rows"],
        "positive_rate": prov["positive_rate"],
        "provenance": prov,
    }


def run_real() -> dict:
    from data.load import ROOT as DATA_ROOT

    lk = leakage_experiment_real(DATA_ROOT)
    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "is_synthetic": False,
        "data_source": "PhysioNet MIMIC-IV clinical database demo (open access); "
                       "see data/MANIFEST.json for file hashes and retrieval times",
        "cohort_is_a_demonstration_not_a_study": True,
        "sample_size_caveat":
            "about 100 ICU stays. The direction of the leakage effect is "
            "meaningful; its magnitude, from a cohort this small, is not a "
            "population estimate.",
        "package": {"name": "icuflow", "version": icuflow.__version__,
                    "exports": len(icuflow.__all__), "published": False},
        "leakage": lk,
        "guards": guard_experiment(),
        "cli": cli_check(),
    }
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "latest-real.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf8")
    return results


def main_real() -> int:
    from data.datakit import FetchError
    try:
        r = run_real()
    except FetchError as exc:
        print(f"cannot run on real data: {exc}", file=sys.stderr)
        return 2
    lk = r["leakage"]
    pv = lk["provenance"]
    print(f"source: {r['data_source']}")
    print(f"{lk['n_subjects']} patients across {lk['n_stays']} ICU stays "
          f"({lk['n_readmissions']} readmissions), {lk['n_rows']:,} rows")
    print(f"target: {pv['event_definition']} within {pv['horizon_hours']:.0f} h "
          f"(positive rate {lk['positive_rate']:.2%})")
    print(f"grouping key: {pv['grouping_key']}")
    print(f"\n  random row split   AUROC {lk['row_split_auroc']:.4f}")
    print(f"  subject split      AUROC {lk['subject_split_auroc']:.4f}")
    if lk["inflation"] > 0:
        print(f"  inflation          {lk['inflation']:+.4f} "
              f"({lk['inflation_pct']:+.1f}%) -- row splitting flatters the model")
    else:
        print(f"  difference         {lk['inflation']:+.4f} -- no inflation "
              f"measured on this cohort, which is a finding, not a failure")
    print("\n" + r["sample_size_caveat"])
    print("wrote results/latest-real.json")
    return 0


def main() -> int:
    if "--real" in sys.argv[1:]:
        return main_real()
    r = run()
    lk = r["leakage"]
    print(f"icuflow {r['package']['version']}, {r['package']['exports']} exports")
    print(f"\nsubject leakage ({lk['n_subjects']} subjects, {lk['n_rows']:,} rows):")
    print(f"  random row split   AUROC {lk['row_split_auroc']:.4f}   <- inflated")
    print(f"  subject split      AUROC {lk['subject_split_auroc']:.4f}   <- honest")
    print(f"  inflation          {lk['inflation']:+.4f} ({lk['inflation_pct']:+.1f}%)")
    print("\ncalibration (ECE, lower is better):")
    for c in r["calibration"]["cases"]:
        print(f"  {c['case']}")
        print(f"    raw {c['raw_ece']:.5f}  isotonic {c['isotonic_ece']:.5f}  "
              f"platt {c['platt_ece']:.5f}   (AUROC {c['raw_auroc']} -> "
              f"{c['isotonic_auroc']})")
    a = r["alarms"]
    print(f"\nalarms at matched sensitivity {a['target_sensitivity']:.0%}:")
    print(f"  false alerts {a['baseline_false_alerts']} -> {a['model_false_alerts']} "
          f"({a['reduction_pct']:+.1f}%)")
    g = r["guards"]
    print(f"\nguards: {sum(bool(v) for v in g.values())}/{len(g)} fired correctly")
    try:
        from .site import build_site
        build_site(r)
        print("\nwebsite/ rebuilt from this run")
    except Exception as exc:
        print(f"\n(site not rebuilt: {exc})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
