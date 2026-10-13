#!/usr/bin/env python3
"""Structural checks on what the pipeline writes, so a broken table is caught here and not
by LaTeX three days later.

  * every generated tabular has as many cells per row as its column specification declares
    (\\multicolumn and \\hline / \\cline are taken into account);
  * every macro of results_macros.tex is a well-formed definition with balanced braces;
  * the aggregated CSVs are consistent: p-values in [0, 1], metrics in [0, 1], the
    confidence intervals contain their difference, and every comparison is paired on folds.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import read_csv  # noqa: E402

SPEC_RE = re.compile(r"\\begin\{tabular\}\{(.*)\}")
MULTI_RE = re.compile(r"\\multicolumn\{(\d+)\}")


def n_columns(spec: str) -> int:
    spec = re.sub(r"@\{[^}]*\}", "", spec)
    spec = re.sub(r"[*]\{(\d+)\}\{([^}]*)\}", lambda m: m.group(2) * int(m.group(1)), spec)
    spec = re.sub(r"[pmb]\{[^}]*\}", "c", spec)
    return sum(1 for ch in spec if ch in "lcr")


def check_table(path: Path) -> list[str]:
    text = path.read_text()
    m = SPEC_RE.search(text)
    if not m:
        return [f"{path.name}: no \\begin{{tabular}}"]
    want = n_columns(m.group(1))
    if want == 0:
        return [f"{path.name}: empty column specification '{m.group(1)}'"]
    problems = []
    body = text[m.end():text.index("\\end{tabular}")] if "\\end{tabular}" in text else None
    if body is None:
        return [f"{path.name}: no \\end{{tabular}}"]
    for k, row in enumerate(body.split("\\\\")):
        row = re.sub(r"\\[hc]line(\{[^}]*\})?", "", row).strip()
        if not row or row.startswith("%"):
            continue
        cells = row.split("&")
        got = len(cells) + sum(int(x) - 1 for x in MULTI_RE.findall(row))
        if got != want:
            problems.append(f"{path.name}: row {k + 1} has {got} cells, the specification declares {want}: "
                            f"{row[:70]!r}")
    if text.count("{") != text.count("}"):
        problems.append(f"{path.name}: unbalanced braces")
    return problems


def check_macros(path: Path) -> list[str]:
    problems, seen = [], set()
    for k, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip() or line.startswith("%"):
            continue
        m = re.match(r"\\expandafter\\def\\csname res@([^@]+)@([^\\]+)\\endcsname\{(.*)\}$", line)
        if not m:
            problems.append(f"results_macros.tex:{k}: not a well-formed definition: {line[:60]!r}")
            continue
        key = (m.group(1), m.group(2))
        if key in seen:
            problems.append(f"results_macros.tex:{k}: duplicate {key}")
        seen.add(key)
        if m.group(3).count("{") != m.group(3).count("}"):
            problems.append(f"results_macros.tex:{k}: unbalanced braces in the value")
    if not seen:
        problems.append("results_macros.tex: no macros defined")
    return problems


def check_results(results: Path) -> list[str]:
    problems = []
    for name in ("metrics.csv", "per_fold.csv", "comparisons.csv"):
        if not (results / name).exists():
            return [f"{results/name} does not exist"]
    metrics = read_csv(results / "metrics.csv")
    if not metrics:
        return ["metrics.csv is empty"]
    for r in metrics:
        v = r.get("mean")
        if v in (None, "") or not (0.0 <= float(v) <= 1.0):
            problems.append(f"metrics.csv: {r['id']} {r['eval_set']} {r['metric']} = {v}")
        if int(r["n_folds"]) < 1:
            problems.append(f"metrics.csv: {r['id']} has no folds")
    for r in read_csv(results / "comparisons.csv"):
        d, lo, hi = float(r["delta"]), float(r["ci_low"]), float(r["ci_high"])
        if not lo - 1e-9 <= d <= hi + 1e-9:
            problems.append(f"comparisons.csv: {r['pair']} {r['metric']}: {d} outside [{lo}, {hi}]")
        for f in ("p_paired_t", "p_wilcoxon", "p_holm"):
            p = r.get(f)
            if p not in (None, "", "nan") and not 0.0 <= float(p) <= 1.0:
                problems.append(f"comparisons.csv: {r['pair']} {f} = {p}")
        if int(r["n_folds"]) < 2:
            problems.append(f"comparisons.csv: {r['pair']} paired on {r['n_folds']} fold(s)")
    per_fold = read_csv(results / "per_fold.csv")
    if not per_fold:
        problems.append("per_fold.csv is empty")
    return problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--tables", required=True)
    a = ap.parse_args()
    tables, work = Path(a.tables), Path(a.work)
    problems = []
    tabs = sorted(tables.glob("tab_*.tex"))
    if not tabs:
        problems.append(f"no tables in {tables}")
    for t in tabs:
        problems += check_table(t)
    macros = tables / "results_macros.tex"
    problems += check_macros(macros) if macros.exists() else [f"{macros} is missing"]
    problems += check_results(work / "results")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for p in problems:
            print(f"  {p}")
        raise SystemExit(1)
    print(f"checked {len(tabs)} tables, {macros.name} and the aggregated CSVs: no problems")


if __name__ == "__main__":
    main()
