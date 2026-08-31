# PyHealth / RHealth Extension

> **Status: scaffold.** Structure only — no method, data or result is claimed
> yet. Every "not yet measured" below is a real gap, not a placeholder to be
> filled in with an estimate.

**Repository:** `2.0-Healthcare-Ai-Systems`
**NIW pillar (Dhanasar prong 1):** Healthcare Safety
**Evidence value:** CORE — makes an adoption claim checkable

## Core idea

A PyHealth / RHealth extension released as a pip package.

## Why it earns its place

Makes the "adopted by multiple external research groups" claim checkable via registry download statistics.

## The petition claim it supports

> Open-source tools including PyHealth / RHealth extensions used by multiple external research groups.

**What the portfolio shows today:** No PyHealth or RHealth extension exists in any of the five repositories. The named open-source contribution is absent.

**Action required:** Publish the extension as a pip package; download counts then become real adoption evidence. Cite only adoption you can document with names and dates.

No prior work in the portfolio — this starts from scratch.

## Petition-grade checklist

A project counts as petition-grade only when all five are true. None are yet.

- [ ] Original work, authored here
- [ ] A stated method (`docs/METHOD.md`)
- [ ] Real data at a stated scale (`docs/DATA.md` — target: Public registry download statistics)
- [ ] A measured result (`results/README.md`)
- [ ] A README a reviewer can follow, start to finish

## Measured results

Target scale: **Public registry download statistics**

| Metric | Baseline | Result | Out-of-sample |
|---|---|---|---|
| PyPI downloads (documented, never inflated) | _not yet measured_ | _not yet measured_ | _pending_ |
| Named institutional users with dated correspondence | _not yet measured_ | _not yet measured_ | _pending_ |
| Upstream issues / PRs referencing the package | _not yet measured_ | _not yet measured_ | _pending_ |

Populate this from `results/`. Do not cite any number in the petition that does
not appear here with a run date behind it.

## Layout

```
pyhealth-rhealth-extension/
├── README.md        this file
├── docs/
│   ├── METHOD.md    what the method is and why it is non-obvious
│   ├── DATA.md      source, scale, licence, and how to reproduce the pull
│   └── EVIDENCE.md  the petition claim, the gap, and the exhibit it becomes
├── src/             implementation
├── data/            pointers and manifests — never raw licensed data
├── results/         measured results, run logs, and the baseline comparison
└── tests/           tests that establish the result is reproducible
```

---
Scaffold generated from `NIW_Project_Portfolio_and_Gap_Plan.xlsx` (sheets: Repo Build-Out Plan, Core Ideas at a Glance, NIW Claim vs Repo Evidence, Notion 创业 Alignment). Structure only — no results are claimed here yet.
