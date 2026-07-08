#!/usr/bin/env python3
"""
tools/generate_mlsd_training_data.py

Generate labelled training data for the ML MLSD Cmax predictor.

For each random MlsdInstance, runs mlsd-exact (small) or mlsd-ga (large) via
the dls_mlsd_solve C binding and records the achieved Cmax.

MLSD instance text format (cli/class_io.hpp readMlsdInstance):
  task <size> [beta]
  proc <S> <C> <A> <B>

Output: tools/mlsd_training_data.csv

Usage:
    python3 tools/generate_mlsd_training_data.py [--n 3000] [--seed 42]
"""

import argparse
import ctypes
import glob
import json
import math
import os
import random
import time

# ── locate libdls_c.so ─────────────────────────────────────────────────────
def find_lib():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for pattern in ["build-highs/bin/libdls_c.*", "build*/bin/libdls_c.*"]:
        hits = sorted(glob.glob(os.path.join(root, pattern)))
        if hits:
            return hits[0]
    raise RuntimeError("libdls_c not found — build with cmake first")

LIB_PATH = os.environ.get("DLS_LIB") or find_lib()
lib = ctypes.CDLL(LIB_PATH)
lib.dls_mlsd_solve.restype  = ctypes.c_void_p
lib.dls_mlsd_solve.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.dls_free.argtypes = [ctypes.c_void_p]

def _mlsd_solve(instance_text, solver):
    ptr = lib.dls_mlsd_solve(instance_text.encode(), solver.encode(), b"")
    raw = ctypes.cast(ptr, ctypes.c_char_p).value
    lib.dls_free(ptr)
    return json.loads(raw)

EXACT_LIMIT = 4   # n_tasks × n_procs <= this → use mlsd-exact
TIME_CAP    = 8.0

FEATURE_NAMES = ["nTasks", "nProcs", "meanV", "cvV", "maxMinV", "totalLoadPerProc",
                 "memoryRatio", "meanA", "heteroA", "speedupA", "hasStartups", "hasCommCost"]


# ── instance generation ────────────────────────────────────────────────────

def random_mlsd_instance(rng, regime):
    """Return (instance_text, n_tasks, n_procs, tasks, procs)."""
    if regime == 0:
        n, m = rng.randint(2, 4), rng.randint(2, 6)
        tasks = [{"V": rng.uniform(100, 3000), "beta": 0.0} for _ in range(n)]
        procs = [{"S": 0.0, "C": rng.uniform(0.01, 1.0),
                  "A": rng.uniform(0.1, 2.0), "B": 1e18} for _ in range(m)]
    elif regime == 1:
        n, m = rng.randint(2, 5), rng.randint(2, 5)
        tasks = [{"V": rng.uniform(100, 5000), "beta": 0.0} for _ in range(n)]
        procs = [{"S": rng.uniform(0, 0.5), "C": rng.uniform(0, 1.0),
                  "A": rng.uniform(0.1, 2.0), "B": 1e18} for _ in range(m)]
    elif regime == 2:
        n, m = rng.randint(2, 4), rng.randint(2, 6)
        tasks = [{"V": rng.uniform(100, 2000), "beta": 0.0} for _ in range(n)]
        B_each = sum(t["V"] for t in tasks) / m * rng.uniform(0.4, 0.9)
        procs = [{"S": rng.uniform(0, 0.3), "C": rng.uniform(0, 0.5),
                  "A": rng.uniform(0.1, 2.0), "B": B_each * rng.uniform(0.5, 1.5)}
                 for _ in range(m)]
    else:
        n, m = rng.randint(3, 6), rng.randint(3, 8)
        tasks = [{"V": rng.uniform(50, 8000), "beta": 0.0} for _ in range(n)]
        procs = [{"S": rng.uniform(0, 1.0) if rng.random() > 0.4 else 0.0,
                  "C": rng.uniform(0, 1.0), "A": rng.uniform(0.05, 3.0), "B": 1e18}
                 for _ in range(m)]

    lines = []
    for t in tasks:
        lines.append(f"task {t['V']:.6f} {t['beta']:.6f}")
    for p in procs:
        lines.append(f"proc {p['S']:.6f} {p['C']:.6f} {p['A']:.6f} {p['B']:.6e}")
    return "\n".join(lines), n, m, tasks, procs


def compute_mlsd_features(tasks, procs):
    n = len(tasks)
    m = len(procs)
    V = [t["V"] for t in tasks]
    A = [p["A"] for p in procs]
    C = [p["C"] for p in procs]
    S = [p["S"] for p in procs]
    B = [min(p["B"], 1e18) for p in procs]

    def cv(vals):
        mn = sum(vals) / len(vals)
        if mn < 1e-12: return 0.0
        var = sum((v - mn)**2 for v in vals) / len(vals)
        return math.sqrt(max(var, 0)) / mn

    mV = sum(V) / n
    mA = sum(A) / m
    sumV = sum(V)
    speedupA = max(A) / min(A) if min(A) > 1e-12 else 1.0
    maxMinV  = max(V) / min(V) if min(V) > 1e-12 else 1.0
    memRatio = min(sum(B) / sumV, 1e6) if sumV > 1e-12 else 1e6

    return [
        float(n),
        float(m),
        mV,
        cv(V),
        maxMinV,
        sumV / m,
        memRatio,
        mA,
        cv(A),
        speedupA,
        1.0 if any(s > 0 for s in S) else 0.0,
        1.0 if any(c > 0 for c in C) else 0.0,
    ]


# ── main loop ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",  default=os.path.join(os.path.dirname(__file__),
                                                    "mlsd_training_data.csv"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    regimes = [0, 1, 2, 3]
    weights = [0.25, 0.25, 0.25, 0.25]

    rows    = []
    skipped = 0
    t_start = time.time()

    print(f"Generating {args.n} MLSD instances using {LIB_PATH}")

    for i in range(args.n):
        regime = rng.choices(regimes, weights=weights)[0]
        inst_text, n, m, tasks, procs = random_mlsd_instance(rng, regime)
        features = compute_mlsd_features(tasks, procs)

        solver = "mlsd-exact" if n * m <= EXACT_LIMIT else "mlsd-ga"

        t0 = time.time()
        try:
            res = _mlsd_solve(inst_text, solver)
            if time.time() - t0 > TIME_CAP:
                skipped += 1
                continue
            if res.get("status") not in ("Optimal", "Feasible"):
                skipped += 1
                continue
            cmax = float(res["makespan"])
            if cmax <= 0:
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        rows.append(features + [cmax])

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed
            eta     = (args.n - i - 1) / rate
            print(f"  {i+1}/{args.n}  rows={len(rows)}  skipped={skipped}  "
                  f"{elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

    header = FEATURE_NAMES + ["cmax"]
    with open(args.out, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s. {len(rows)} rows → {args.out}  (skipped {skipped})")


if __name__ == "__main__":
    main()
