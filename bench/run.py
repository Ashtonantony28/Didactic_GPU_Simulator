#!/usr/bin/env python3
"""bench/run.py — drive every kernel test, snapshot results to JSON.

Usage:
    python3 bench/run.py [--phase N] [--params "ICACHE_LINES=16 COALESCE=1"] [tests...]

Each test writes its own JSON via runner.write_bench_result(); this driver
re-runs the regression and aggregates into bench/phase-N.json.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TESTS = [
    "matadd", "matmul", "vector_dot", "relu", "conv1d",
    "transpose", "divergent_relu", "tiled_matmul",
]


def run_one(name: str, params: str) -> dict | None:
    test_file = ROOT / "test" / f"test_{name}.py"
    if not test_file.exists():
        return None
    iv_params = " ".join(f"-Pgpu.{p}" for p in params.split()) if params else ""
    env = {**os.environ}
    cmd = ["make", f"test_{name}", f"PARAMS={iv_params}"]
    res = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, env=env)
    passed = "PASS=1 FAIL=0" in res.stdout
    result_file = ROOT / "bench" / "results" / f"{name}.json"
    metrics = {}
    if result_file.exists():
        metrics = json.loads(result_file.read_text())
    metrics["passed"] = passed
    metrics["name"] = name
    if not passed:
        metrics["stdout_tail"] = res.stdout.splitlines()[-12:]
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=0)
    ap.add_argument("--params", default="")
    ap.add_argument("--label", default=None)
    ap.add_argument("tests", nargs="*", default=DEFAULT_TESTS)
    args = ap.parse_args()

    rollup = {"phase": args.phase, "params": args.params, "results": []}
    fails = 0
    for t in args.tests:
        m = run_one(t, args.params)
        if m is None:
            print(f"  -- {t}: no test file, skipping")
            continue
        marker = "PASS" if m.get("passed") else "FAIL"
        cycles = m.get("cycles", "?")
        mreqs  = m.get("data_mem_requests", "?")
        coal   = m.get("data_mem_requests_coalesced", "?")
        ihits  = m.get("icache_hits", "?")
        imiss  = m.get("icache_misses", "?")
        print(f"  [{marker}] {t:18s} cycles={cycles}  mem_req={mreqs}  coal={coal}  icache hit/miss={ihits}/{imiss}")
        rollup["results"].append(m)
        if not m.get("passed"):
            fails += 1

    label = args.label or f"phase-{args.phase}"
    out = ROOT / "bench" / f"{label}.json"
    out.write_text(json.dumps(rollup, indent=2, sort_keys=True))
    print(f"\nWrote {out}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
