#!/usr/bin/env python3
"""Fail-fast checks before anything is submitted.

* prints Python / PyTorch / CUDA / Ultralytics versions and the GPU (for the paper);
* reads the Ultralytics version stored inside every existing checkpoint of the
  manifest and warns if it differs from the installed one (the retrained runs must
  use the same software stack as E11-E17);
* checks that data YAMLs, existing checkpoints and pretrained weights exist.

Example
-------
python check_env.py --manifest experiments.tsv --data /data/v1/data.yaml /data/v2/data.yaml /data/v3/data.yaml \
    --weights-dir /scratch/weights
"""
import argparse
import os
import sys
from pathlib import Path

from common import environment_info, experiment_weights, read_manifest, split_sources, torch_load


def gpu_problem():
    """A GPU is allocated to this job but PyTorch cannot use it (usually a CUDA build newer than the driver)."""
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        return None
    try:
        import torch
    except Exception:
        return None
    if torch.cuda.is_available():
        return None
    drv = None
    try:
        v = torch._C._cuda_getDriverVersion()
        drv = f"{v // 1000}.{(v % 1000) // 10}"
    except Exception:
        pass
    return (f"a GPU is allocated but PyTorch {torch.__version__} (built for CUDA {torch.version.cuda}) cannot use it"
            + (f"; the driver supports CUDA {drv}" if drv else "")
            + ". Install a PyTorch build for that CUDA version (README, 'PyTorch and the NVIDIA driver').")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", default="experiments.tsv")
    ap.add_argument("--data", nargs="*", default=[])
    ap.add_argument("--weights-dir", default=None)
    ap.add_argument("--strict", action="store_true", help="exit with an error on any problem")
    a = ap.parse_args()
    problems = []
    env = environment_info()
    print("Environment:")
    for k, v in env.items():
        print(f"  {k}: {v}")
    installed = env.get("ultralytics")
    gpu_note = gpu_problem()
    if gpu_note:
        problems.append(gpu_note)

    for d in a.data:
        try:
            srcs = split_sources(d)
            for s, lst in srcs.items():
                for p in lst:
                    if not Path(p).exists():
                        problems.append(f"{d}: split '{s}' points to missing {p}")
            print(f"data OK: {d} ({', '.join(srcs)})")
        except Exception as e:
            problems.append(f"{d}: {e}")

    exps = read_manifest(a.manifest)
    seen_versions = {}
    for eid, e in exps.items():
        if e["weights"] != "auto":
            runs = sorted(d for d in Path(e["weights"]).glob("fold*_seed*") if d.is_dir()) \
                if Path(e["weights"]).is_dir() else []
            if runs:
                found = [d for d in runs if (d / ("best.pth" if e["kind"] == "maskrcnn" else "weights/best.pt")).exists()]
                print(f"{eid}: {len(found)} of {len(runs)} runs have a checkpoint in {e['weights']}")
                if not found:
                    problems.append(f"{eid}: {e['weights']} has fold*_seed* folders but none contains a checkpoint")
                    continue
                if len(found) < len(runs):
                    problems.append(f"{eid}: {len(runs) - len(found)} run(s) in {e['weights']} have no checkpoint "
                                    f"({', '.join(d.name for d in runs if d not in found)})")
                p = found[0] / ("best.pth" if e["kind"] == "maskrcnn" else "weights/best.pt")
            else:
                p = experiment_weights(e, ".")
                if not p.exists():
                    problems.append(f"{eid}: no fold*_seed* run found under {e['weights']} "
                                    f"(the folder must contain one subfolder per fold and seed)")
                    continue
            if e["kind"] == "yolo":
                try:
                    ck = torch_load(p)
                    v = ck.get("version", "unknown") if isinstance(ck, dict) else "unknown"
                    seen_versions[eid] = v
                    ta = ck.get("train_args", {}) if isinstance(ck, dict) else {}
                    print(f"{eid}: {p.name} trained with ultralytics {v}; epochs={ta.get('epochs')} batch={ta.get('batch')} "
                          f"imgsz={ta.get('imgsz')} optimizer={ta.get('optimizer')} seed={ta.get('seed')} "
                          f"copy_paste={ta.get('copy_paste')} model={ta.get('model')}")
                except Exception as e2:
                    problems.append(f"{eid}: cannot read {p}: {e2}")
        elif e["kind"] == "yolo" and a.weights_dir:
            w = Path(a.weights_dir) / e["model"]
            if not w.exists():
                problems.append(f"{eid}: pretrained weights {w} missing (run fetch_weights.py on a node with internet)")
    vs = {v for v in seen_versions.values() if v != "unknown"}
    if vs and installed and installed not in vs:
        problems.append(f"installed ultralytics {installed} differs from the version(s) of the existing checkpoints {sorted(vs)}; "
                        f"install the same version (pip install ultralytics=={sorted(vs)[0]}) so that retrained runs are comparable")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print("  - " + p)
        if a.strict:
            sys.exit(1)
    else:
        print("\nAll checks passed.")


if __name__ == "__main__":
    main()
