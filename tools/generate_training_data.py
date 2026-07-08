#!/usr/bin/env python3
"""
tools/generate_training_data.py

Generate labelled training data for the ML solver selector and difficulty predictor.

For each random DLSInstance:
  - Runs applicable solvers, records which achieved the best makespan → training_data.csv
  - Computes the gap between the best heuristic and exact (when N <= 6), derives an
    easy/medium/hard difficulty label → difficulty_data.csv

Features match InstanceFeatures in core/instance_features.hpp.

Solver selection is adaptive to N:
  N <= 6  : single-round, ga, best-rate, online, exact
  N >  6  : single-round, ga, best-rate, online
  (exact is O(N^k) and becomes prohibitively slow for N > 6)

Output: tools/training_data.csv, tools/difficulty_data.csv

Usage:
    python3 tools/generate_training_data.py [--n 5000] [--seed 42]
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
lib.dls_solve.restype  = ctypes.c_void_p
lib.dls_solve.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.dls_free.argtypes  = [ctypes.c_void_p]

def _call_solve(instance_text, solver, opts_dict=None):
    opts = json.dumps(opts_dict).encode() if opts_dict else b""
    ptr  = lib.dls_solve(instance_text.encode(), solver.encode(), opts)
    raw  = ctypes.cast(ptr, ctypes.c_char_p).value
    lib.dls_free(ptr)
    return json.loads(raw)

FAST_SOLVERS     = ["single-round", "ga", "best-rate", "online"]
EXACT_SOLVERS    = ["exact"]        # only when N <= 6 (B&B stays tractable)
EXACT_N_LIMIT    = 6

TIME_CAP = 5.0   # seconds per solver per instance (safety net)

# ── instance generation ────────────────────────────────────────────────────

def random_instance(rng, regime):
    if regime == 0:
        N = rng.randint(2, 12)
        V = rng.uniform(100, 5000)
        procs = [{"S": 0.0,
                  "C": rng.uniform(0.01, 1.0),
                  "A": rng.uniform(0.1, 2.0),
                  "B": 1e18} for _ in range(N)]
    elif regime == 1:
        N = rng.randint(2, 7)
        V = rng.uniform(100, 3000)
        procs = [{"S": rng.uniform(0, 0.5),
                  "C": rng.uniform(0, 1.0),
                  "A": rng.uniform(0.1, 2.0),
                  "B": 1e18} for _ in range(N)]
    elif regime == 2:
        N = rng.randint(2, 10)
        V = rng.uniform(100, 5000)
        B_each = V / N * rng.uniform(0.3, 0.9)
        procs = [{"S": rng.uniform(0, 0.3),
                  "C": rng.uniform(0, 0.5),
                  "A": rng.uniform(0.1, 2.0),
                  "B": B_each * rng.uniform(0.5, 1.5)} for _ in range(N)]
    elif regime == 3:
        N = rng.randint(7, 15)
        V = rng.uniform(500, 10000)
        procs = [{"S": rng.uniform(0.1, 2.0),
                  "C": rng.uniform(0.01, 1.0),
                  "A": rng.uniform(0.1, 2.0),
                  "B": 1e18} for _ in range(N)]
    else:
        N = rng.randint(2, 12)
        V = rng.uniform(50, 8000)
        mem_limited = rng.random() < 0.3
        B_base = (V / N * rng.uniform(0.3, 0.8)) if mem_limited else 1e18
        procs = [{"S": rng.uniform(0, 1.5) if rng.random() > 0.3 else 0.0,
                  "C": rng.uniform(0, 1.0),
                  "A": rng.uniform(0.05, 3.0),
                  "B": B_base * rng.uniform(0.5, 2.0) if mem_limited else 1e18}
                 for _ in range(N)]

    lines = [f"V {V:.6f}"]
    for p in procs:
        lines.append(f"{p['S']:.6f} {p['C']:.6f} {p['A']:.6f} {p['B']:.6e}")
    return "\n".join(lines), V, procs

# ── feature extraction (mirrors core/instance_features.hpp) ───────────────

def compute_features(V, procs):
    n = len(procs)
    A = [p["A"] for p in procs]
    C = [p["C"] for p in procs]
    S = [p["S"] for p in procs]
    B = [min(p["B"], 1e18) for p in procs]

    def cv(vals):
        m = sum(vals) / len(vals)
        if m < 1e-12:
            return 0.0
        var = sum((v - m)**2 for v in vals) / len(vals)
        return math.sqrt(max(var, 0)) / m

    mA = sum(A) / n
    mC = sum(C) / n
    mS = sum(S) / n
    sumB = sum(B)
    mem_ratio = min(sumB / V, 1e6) if V > 1e-12 else 1e6
    startup_frac = (n * mS) / (n * mS + V * mA + 1e-12)
    speedupA = max(A) / min(A) if min(A) > 1e-12 else 1.0
    speedupC = max(C) / min(C) if min(C) > 1e-12 else 1.0

    return [
        float(n),
        mem_ratio,
        1.0 if any(s > 0 for s in S) else 0.0,
        1.0 if any(c > 0 for c in C) else 0.0,
        cv(A),
        cv(C),
        cv(S),
        startup_frac,
        0.0,   # hasBeta (not generated in this script)
        0.0,   # hasCost (not generated in this script)
        mA,
        mC,
        mS,
        speedupA,
        speedupC,
        V / n,
    ]

FEATURE_NAMES = ["N", "memoryRatio", "hasStartups", "hasCommCost",
                 "heteroA", "heteroC", "heteroS", "startupFraction",
                 "hasBeta", "hasCost",
                 "meanA", "meanC", "meanS", "speedupA", "speedupC", "loadPerProc"]

# ── main loop ─────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",    type=int, default=5000, help="number of instances")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out",  default=os.path.join(os.path.dirname(__file__), "training_data.csv"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    regimes = [0, 1, 2, 3, 4]
    regime_weights = [0.15, 0.25, 0.20, 0.20, 0.20]

    rows            = []   # solver-selector training data
    difficulty_rows = []   # difficulty predictor training data
    skipped = 0

    diff_out = os.path.join(os.path.dirname(args.out), "difficulty_data.csv")

    print(f"Generating {args.n} instances using {LIB_PATH}")
    print(f"Fast solvers: {FAST_SOLVERS}  |  Exact solvers (N<={EXACT_N_LIMIT}): {EXACT_SOLVERS}")
    t_start = time.time()

    for i in range(args.n):
        regime = rng.choices(regimes, weights=regime_weights)[0]
        inst_text, V, procs = random_instance(rng, regime)
        features = compute_features(V, procs)
        N = len(procs)

        solvers = FAST_SOLVERS + (EXACT_SOLVERS if N <= EXACT_N_LIMIT else [])

        best_makespan = float("inf")
        best_solver   = None
        results       = {}

        for solver in solvers:
            t0 = time.time()
            try:
                res     = _call_solve(inst_text, solver)
                elapsed = time.time() - t0
                if elapsed > TIME_CAP:
                    continue
                sol = res.get("solution", {})
                if sol.get("status") in ("Feasible", "Optimal"):
                    ms = float(sol.get("makespan", float("inf")))
                    results[solver] = ms
                    if ms < best_makespan - 1e-9:
                        best_makespan = ms
                        best_solver   = solver
            except Exception:
                pass

        if best_solver is None or len(results) < 2:
            skipped += 1
            continue

        rows.append(features + [best_solver, best_makespan])

        # Difficulty label: gap between best heuristic and exact optimum.
        exact_ms = results.get("exact")
        heuristic_ms = min(
            (results[s] for s in FAST_SOLVERS if s in results),
            default=float("inf"))
        if exact_ms is not None and heuristic_ms < float("inf"):
            gap = (heuristic_ms - exact_ms) / (exact_ms + 1e-12)
            if gap < 0.01:
                difficulty = "easy"
            elif gap < 0.10:
                difficulty = "medium"
            else:
                difficulty = "hard"
        elif N > EXACT_N_LIMIT:
            difficulty = "hard"   # exact intractable; heuristic quality unknown
        else:
            difficulty = None     # not enough data for this instance

        if difficulty is not None:
            difficulty_rows.append(features + [difficulty])

        if (i + 1) % 500 == 0:
            elapsed = time.time() - t_start
            rate    = (i + 1) / elapsed
            eta     = (args.n - i - 1) / rate
            print(f"  {i+1}/{args.n}  rows={len(rows)}  diff={len(difficulty_rows)}  "
                  f"skipped={skipped}  {elapsed:.0f}s  ETA {eta:.0f}s", flush=True)

    header = FEATURE_NAMES + ["best_solver", "best_makespan"]
    with open(args.out, "w") as f:
        f.write(",".join(header) + "\n")
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")

    diff_header = FEATURE_NAMES + ["difficulty"]
    with open(diff_out, "w") as f:
        f.write(",".join(diff_header) + "\n")
        for row in difficulty_rows:
            f.write(",".join(str(v) for v in row) + "\n")

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s.")
    print(f"  {len(rows)} solver-selector rows → {args.out}  (skipped {skipped})")
    print(f"  {len(difficulty_rows)} difficulty rows → {diff_out}")

    from collections import Counter
    labels = [r[-2] for r in rows]   # last column is best_makespan; label is second-to-last
    dist   = Counter(labels)
    print("\nSolver label distribution:")
    for solver, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {solver:20s} {count:6d}  ({100*count/len(rows):.1f}%)")

    dlabels = [r[-1] for r in difficulty_rows]
    ddist   = Counter(dlabels)
    print("\nDifficulty label distribution:")
    for label, count in sorted(ddist.items(), key=lambda x: -x[1]):
        print(f"  {label:20s} {count:6d}  ({100*count/len(difficulty_rows):.1f}%)")

if __name__ == "__main__":
    main()
