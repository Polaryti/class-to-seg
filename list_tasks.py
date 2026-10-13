#!/usr/bin/env python3
"""Write the task lists the SLURM arrays iterate over (one line per run).

  train_tasks.txt   the runs that have to be trained here (manifest weights = 'auto')
  eval_tasks.txt    every run to score: the existing ones plus the ones trained here
Each line is "<id> <fold> <seed> <flags>", flags being a comma-separated list or '-':
  gflops    also measure parameters and GFLOPs (set once per configuration)
  savedets  keep the detections of this run (for the qualitative figures)
"""
import argparse
from pathlib import Path

from common import read_manifest
from folds import enumerate_runs, folds_available, planned_runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="experiments.tsv")
    ap.add_argument("--runs", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", default="0", help="seeds for the runs trained here, e.g. '0 1 2'")
    ap.add_argument("--maskrcnn-seeds", default=None, help="seeds for the Mask R-CNN runs (default: --seeds)")
    ap.add_argument("--figure-id", default="E13", help="configuration whose detections are kept for the figures")
    a = ap.parse_args()
    folds = folds_available(a.report)
    if not folds:
        raise SystemExit(f"no common_test_*.txt in {a.report}")
    seeds = [int(x) for x in a.seeds.split()]
    seeds_by_kind = {"yolo": seeds, "maskrcnn": [int(x) for x in (a.maskrcnn_seeds or a.seeds).split()]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    train = planned_runs(a.manifest, a.runs, folds, seeds_by_kind)
    (out / "train_tasks.txt").write_text("".join(f"{t['id']} {t['fold']} {t['seed']} -\n" for t in train))

    tasks = [(t["id"], t["fold"], t["seed"]) for t in train]
    for r in enumerate_runs(a.manifest, a.runs, folds=folds):
        if not r["trained_here"]:
            tasks.append((r["id"], r["fold"], r["seed"]))
    order = {eid: i for i, eid in enumerate(read_manifest(a.manifest))}
    tasks = sorted(set(tasks), key=lambda t: (order.get(t[0], 99), t[1], t[2]))
    seen, lines = set(), []
    for eid, fold, seed in tasks:
        flags = []
        if eid not in seen:
            flags.append("gflops")
            seen.add(eid)
        if eid == a.figure_id and fold == folds[0] and seed == 0:
            flags.append("savedets")
        lines.append(f"{eid} {fold} {seed} {','.join(flags) or '-'}\n")
    (out / "eval_tasks.txt").write_text("".join(lines))
    print(f"{len(train)} runs to train -> {out/'train_tasks.txt'}")
    print(f"{len(lines)} runs to evaluate -> {out/'eval_tasks.txt'}")
    for eid in sorted({t[0] for t in tasks}, key=lambda e: order.get(e, 99)):
        print(f"  {eid:16s} {sum(1 for t in tasks if t[0] == eid):3d} runs")


if __name__ == "__main__":
    main()
