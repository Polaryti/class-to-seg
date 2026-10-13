"""Fold bookkeeping: which runs exist, and which images each fold is scored on.

Layout assumed (the one of this study):
  <folds_root>/<corpus>/<fold>/data.yaml        the per-fold datasets used for training
  <runs_root>/<config>/<fold>_seed<k>/weights/  one run per fold and seed
  <report>/common_test_<fold>.txt               images of the fold that no model of that
                                                fold trained or validated on (check_folds.py)
"""
from __future__ import annotations

import re
from pathlib import Path

from common import read_manifest

RUN_RE = re.compile(r"^(?P<fold>fold\d+)_seed(?P<seed>\d+)$")


def parse_run_dir(d: Path):
    m = RUN_RE.match(d.name)
    return (m.group("fold"), int(m.group("seed"))) if m else (None, None)


def common_test_images(report_dir: str | Path, fold: str) -> list[Path]:
    f = Path(report_dir) / f"common_test_{fold}.txt"
    if not f.exists():
        raise FileNotFoundError(f"{f} not found (run check_folds.py first)")
    return [Path(x) for x in f.read_text().split() if x]


def folds_available(report_dir: str | Path) -> list[str]:
    return sorted(p.name[len("common_test_"):-len(".txt")] for p in Path(report_dir).glob("common_test_*.txt"))


def enumerate_runs(manifest: str | Path, runs_dir: str | Path, folds: list[str] | None = None,
                   seeds: list[int] | None = None) -> list[dict]:
    """One entry per (configuration, fold, seed) with the checkpoint to evaluate.

    Manifest column 'weights': 'auto' (trained here, under <runs_dir>/<id>/<fold>_seed<seed>)
    or the folder of the existing runs of that configuration (containing fold*_seed* folders).
    """
    out = []
    for eid, exp in read_manifest(manifest).items():
        auto = exp["weights"] == "auto"
        base = Path(runs_dir) / eid if auto else Path(exp["weights"])
        found = []
        if base.is_dir():
            for d in sorted(base.iterdir()):
                fold, seed = parse_run_dir(d)
                if fold is None:
                    continue
                if folds and fold not in folds:
                    continue
                if seeds is not None and seed not in seeds:
                    continue
                w = d / "best.pth" if exp["kind"] == "maskrcnn" else d / "weights" / "best.pt"
                found.append({"id": eid, "kind": exp["kind"], "corpus": exp["corpus"], "model": exp["model"],
                              "args": exp["args"], "fold": fold, "seed": seed, "run_dir": str(d),
                              "weights": str(w), "exists": w.exists(), "trained_here": auto})
        out += found
    return out


def planned_runs(manifest: str | Path, runs_dir: str | Path, folds: list[str], seeds_by_kind: dict) -> list[dict]:
    """The runs that have to be trained here ('auto' in the manifest), as (id, fold, seed)."""
    out = []
    for eid, exp in read_manifest(manifest).items():
        if exp["weights"] != "auto":
            continue
        for fold in folds:
            for seed in seeds_by_kind.get(exp["kind"], [0]):
                d = Path(runs_dir) / eid / f"{fold}_seed{seed}"
                out.append({"id": eid, "kind": exp["kind"], "corpus": exp["corpus"], "model": exp["model"],
                            "args": exp["args"], "fold": fold, "seed": seed, "run_dir": str(d)})
    return out


def corpus_yaml(protocol_dir: str | Path, corpus: str, fold: str) -> Path:
    """Data YAML of a corpus for one fold (control corpora are written by prepare_folds.py)."""
    p = Path(protocol_dir) / f"{corpus}_{fold}.yaml"
    return p
