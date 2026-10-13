#!/usr/bin/env python3
"""Aggregate the per-run results into the numbers the paper reports.

Reads <results>/per_run/*.json (one per configuration, fold and seed) and writes:
  metrics.csv       one row per configuration and evaluation set: mean and standard deviation
                    over folds, the spread between seeds, and the number of runs
  per_fold.csv      one row per configuration, evaluation set and fold (seeds averaged)
  per_class.csv     per class: mean and standard deviation of AP50-95 over folds
  comparisons.csv   paired comparisons between configurations: the difference is taken within
                    each fold (seeds averaged first, because seeds of the same fold share the
                    test images), so the five folds are the independent units; reports the mean
                    difference, a 95% confidence interval and a paired t-test, with Holm
                    correction over the family of comparisons, and the Wilcoxon test when SciPy
                    is available
  models.csv        cost and training progress per configuration

Example
-------
python aggregate.py --results /scratch/results --out /scratch/results
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import write_csv

DEFAULT_PAIRS = ("E13:E3,E13:E8,E3:E8,E19:E3,E19:E13,E20:E13,E16:E13,E17:E13,E18:E13,E14:E13,E11:E13,"
                 "E21:E13,E22:E13")
METRICS = ("mAP50_95", "mAP50", "mAP75", "AP_S", "AP_M", "AP_L", "AR100", "P", "R")


def mean_sd(x):
    x = [v for v in x if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not x:
        return None, None, 0
    return float(np.mean(x)), (float(np.std(x, ddof=1)) if len(x) > 1 else 0.0), len(x)


def t_interval(d, conf=0.95):
    """Mean, half-width of the confidence interval and two-sided p of a paired t-test."""
    d = np.asarray([v for v in d if v is not None], float)
    n = len(d)
    if n < 2:
        return (float(d[0]) if n else float("nan")), float("nan"), float("nan")
    m, se = float(d.mean()), float(d.std(ddof=1) / math.sqrt(n))
    if se == 0:
        return m, 0.0, (0.0 if m != 0 else 1.0)
    t = m / se
    try:
        from scipy import stats
        tc = float(stats.t.ppf(0.5 + conf / 2, n - 1))
        p = float(2 * stats.t.sf(abs(t), n - 1))
    except Exception:  # normal approximation if SciPy is missing
        tc, p = 1.96, float(2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2)))))
    return m, tc * se, p


def how_it_ended(r, patience: int = 100) -> str:
    """Why a training run stopped: its budget, the patience criterion, or something else.

    Ultralytics stops when `epoch - best_epoch >= patience`, so a run that ends at or near
    its own best epoch did not stop for lack of improvement: it was interrupted.
    """
    ep, be, budget = r.get("epochs_run"), r.get("best_epoch"), r.get("epochs_configured") or 300
    if ep is None:
        return "unknown"
    if ep >= budget:
        return "full"
    if be is not None and ep - be >= patience - 1:
        return "patience"
    return "interrupted"


def wilcoxon(d):
    try:
        from scipy import stats
        d = [v for v in d if v is not None and v != 0]
        if len(d) < 3:
            return float("nan")
        return float(stats.wilcoxon(d).pvalue)
    except Exception:
        return float("nan")


def holm(pvals: dict) -> dict:
    items = sorted(((k, v) for k, v in pvals.items() if v == v), key=lambda kv: kv[1])
    n, out, prev = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, max(prev, (n - i) * p))
        out[k] = adj
        prev = adj
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", default=DEFAULT_PAIRS)
    a = ap.parse_args()
    res, out = Path(a.results), Path(a.out)
    runs = [json.loads(p.read_text()) for p in sorted((res / "per_run").glob("*.json"))]
    if not runs:
        raise SystemExit(f"no per-run results in {res/'per_run'}")
    print(f"{len(runs)} runs, {len({r['id'] for r in runs})} configurations")

    sets = ["common", "v1_origin", "v2_origin"]
    # value[(id, set, metric)][fold][seed]
    value = defaultdict(lambda: defaultdict(dict))
    cls_value = defaultdict(lambda: defaultdict(dict))
    for r in runs:
        for s in sets:
            d = r.get(s) or {}
            for m in METRICS:
                if d.get(m) is not None:
                    value[(r["id"], s, m)][r["fold"]][r["seed"]] = d[m]
            pc = (r.get("per_class") if s == "common" else d.get("per_class_AP")) or {}
            for cls, v in pc.items():
                ap50_95 = v["AP50_95"] if isinstance(v, dict) else v
                if ap50_95 is not None:
                    cls_value[(r["id"], s, cls)][r["fold"]][r["seed"]] = ap50_95

    def fold_means(key):
        return {f: float(np.mean(list(sd.values()))) for f, sd in value[key].items() if sd}

    rows = []
    for (eid, s, m), byfold in sorted(value.items()):
        fm = fold_means((eid, s, m))
        mean, sd, n_folds = mean_sd(list(fm.values()))
        seed_sds = [np.std(list(sd_.values()), ddof=1) for sd_ in byfold.values() if len(sd_) > 1]
        rows.append({"id": eid, "eval_set": s, "metric": m, "mean": mean, "sd_folds": sd,
                     "sd_seeds": float(np.mean(seed_sds)) if seed_sds else None,
                     "n_folds": n_folds, "n_runs": sum(len(v) for v in byfold.values())})
    write_csv(out / "metrics.csv", rows)
    write_csv(out / "per_fold.csv", [{"id": eid, "eval_set": s, "metric": m, "fold": f, "value": v,
                                      "n_seeds": len(value[(eid, s, m)][f])}
                                     for (eid, s, m) in sorted(value) for f, v in sorted(fold_means((eid, s, m)).items())])
    crows = []
    for (eid, s, cls), byfold in sorted(cls_value.items()):
        fm = {f: float(np.mean(list(sd.values()))) for f, sd in byfold.items() if sd}
        mean, sd, n = mean_sd(list(fm.values()))
        crows.append({"id": eid, "eval_set": s, "class": cls, "mean": mean, "sd_folds": sd, "n_folds": n})
    write_csv(out / "per_class.csv", crows)

    comps, praw = [], {}
    for pair in [p for p in a.pairs.split(",") if p.strip()]:
        x, y = pair.split(":")
        for s in sets:
            for m in METRICS:
                fx, fy = fold_means((x, s, m)), fold_means((y, s, m))
                folds = sorted(set(fx) & set(fy))
                if len(folds) < 2:
                    continue
                d = [fx[f] - fy[f] for f in folds]
                mean, hw, p = t_interval(d)
                row = {"pair": pair, "a": x, "b": y, "eval_set": s, "metric": m, "n_folds": len(folds),
                       "value_a": float(np.mean([fx[f] for f in folds])), "value_b": float(np.mean([fy[f] for f in folds])),
                       "delta": mean, "ci_low": mean - hw, "ci_high": mean + hw, "p_paired_t": p,
                       "p_wilcoxon": wilcoxon(d),
                       "positive_folds": sum(1 for v in d if v > 0)}
                comps.append(row)
                if s == "common" and m == "mAP50_95":
                    praw[pair] = p
    adj = holm(praw)
    for r in comps:
        if r["eval_set"] == "common" and r["metric"] == "mAP50_95":
            r["p_holm"] = adj.get(r["pair"])
    write_csv(out / "comparisons.csv", comps)

    mrows = {}
    for r in runs:
        d = mrows.setdefault(r["id"], {"id": r["id"], "kind": r["kind"], "corpus": r["corpus"], "model": r["model"],
                                       "runs": 0, "epochs_run": [], "latency": [], "size_mb": r.get("size_mb"),
                                       "ended": []})
        d["runs"] += 1
        d["ended"].append(how_it_ended(r))
        if r.get("params"):
            d["params"], d["gflops"] = r["params"], r.get("gflops")
        if r.get("epochs_run"):
            d["epochs_run"].append(r["epochs_run"])
        if r.get("latency_ms_median"):
            d["latency"].append(r["latency_ms_median"])
    for d in mrows.values():
        ep = d.pop("epochs_run")
        d["epochs_mean"] = float(np.mean(ep)) if ep else None
        d["runs_below_300_epochs"] = sum(1 for e in ep if e < 300)
        ended = d.pop("ended")
        for k in ("full", "patience", "interrupted", "unknown"):
            d[f"runs_{k}"] = sum(1 for e in ended if e == k)
        lat = d.pop("latency")
        d["latency_ms_median"] = float(np.median(lat)) if lat else None
    write_csv(out / "models.csv", list(mrows.values()))

    print(f"\n{'config':16s} {'mAP50-95 (mean +- sd over folds)':>34s} {'seed sd':>9s} {'runs':>5s}")
    for eid in sorted({r["id"] for r in runs}):
        r = next((x for x in rows if x["id"] == eid and x["eval_set"] == "common" and x["metric"] == "mAP50_95"), None)
        if r and r["mean"] is not None:
            print(f"{eid:16s} {r['mean']:>22.4f} +- {r['sd_folds']:.4f} {(r['sd_seeds'] or 0):>9.4f} {r['n_runs']:>5d}")
    print(f"\nWrote {out}/metrics.csv, per_fold.csv, per_class.csv, comparisons.csv, models.csv")


if __name__ == "__main__":
    main()
