#!/usr/bin/env python3
"""
tools/generate_training_data.py

Generate labelled training data for the ML solver selector, difficulty predictor,
and makespan predictor.

For each random DLSInstance:
  - Runs applicable solvers, records which achieved the best makespan → training_data.csv
    (features + best_solver + best_makespan)
  - Computes the gap between the best heuristic and exact (when N <= 6), derives an
    easy/medium/hard difficulty label → difficulty_data.csv

Features match InstanceFeatures in core/instance_features.hpp.

Solver selection is adaptive to N:
  N <= 6  : single-round, ga, best-rate, online, exact
  N >  6  : single-round, ga, best-rate, online
  (exact is O(N^k) and becomes prohibitively slow for N > 6)

Every row also carries provenance columns (genlib.PROVENANCE_COLS): the derived
seed, the regime, the oracle that set the label, whether that label is exact
(exact ran, N <= 6), and the oracle wall-clock. `label_is_exact` lets a trainer
select only proven-optimal rows.

Phase 4: ~40% of exact-labelable instances (N <= 6) carry result-return β > 0,
activating the `hasBeta` feature. Because β-blind heuristics (single-round,
online) misreport makespan for β > 0, any solver whose makespan falls below the
proven exact optimum is dropped before labelling — so β rows keep correct labels.

Output: tools/training_data.csv, tools/difficulty_data.csv

Usage:
    python3 tools/generate_training_data.py [--n 5000] [--seed 42] [--workers N]
"""

import argparse
import os
import math
import random
from collections import Counter

import genlib   # shared harness: lib loading, parallel driver, provenance, CSV

FAST_SOLVERS  = ["single-round", "ga", "best-rate", "online"]
EXACT_SOLVERS = ["exact"]        # only when N <= 6 (B&B stays tractable)
EXACT_N_LIMIT = 6

REGIMES        = [0, 1, 2, 3, 4]
REGIME_WEIGHTS = [0.15, 0.25, 0.20, 0.20, 0.20]

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

    # Phase 4: result-return β. Only for exact-labelable instances (N <= EXACT_N_LIMIT)
    # with AMPLE memory (ΣBᵢ >= V). Two reasons: β-blind heuristics misreport makespan
    # for β > 0 (so a correct label needs the exact optimum), and β on memory-limited
    # instances forces many result-return round-trips whose optimum makespan explodes
    # (up to ~1e8), a pathological corner that wrecks the regressor. ~40% of eligible.
    ample_memory = all(p["B"] >= 1e17 for p in procs)
    beta = 0.0
    if N <= EXACT_N_LIMIT and ample_memory and rng.random() < 0.4:
        beta = rng.uniform(0.1, 0.6)

    lines = ([f"beta {beta:.6f}"] if beta > 0.0 else []) + [f"V {V:.6f}"]
    for p in procs:
        lines.append(f"{p['S']:.6f} {p['C']:.6f} {p['A']:.6f} {p['B']:.6e}")
    return "\n".join(lines), V, procs, beta

# ── feature extraction (mirrors core/instance_features.hpp) ───────────────

FEATURE_NAMES = ["N", "memoryRatio", "hasStartups", "hasCommCost",
                 "heteroA", "heteroC", "heteroS", "startupFraction",
                 "hasBeta", "hasCost",
                 "meanA", "meanC", "meanS", "speedupA", "speedupC", "loadPerProc",
                 "beta"]

def compute_features(V, procs, beta=0.0):
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

    return {
        "N": float(n),
        "memoryRatio": mem_ratio,
        "hasStartups": 1.0 if any(s > 0 for s in S) else 0.0,
        "hasCommCost": 1.0 if any(c > 0 for c in C) else 0.0,
        "heteroA": cv(A),
        "heteroC": cv(C),
        "heteroS": cv(S),
        "startupFraction": startup_frac,
        "hasBeta": 1.0 if beta > 0.0 else 0.0,   # result-return present (Phase 4)
        "hasCost": 0.0,   # cost model not generated in this script
        "meanA": mA,
        "meanC": mC,
        "meanS": mS,
        "speedupA": speedupA,
        "speedupC": speedupC,
        "loadPerProc": V / n,
        "beta": beta,   # result-return magnitude (Phase 4); hasBeta is its 0/1 flag
    }

# ── per-instance task (runs in a worker process) ───────────────────────────

_SEED = 42   # overwritten in main() before the pool forks; inherited by workers

def task(index):
    """Goal: sample one instance, run the solvers, and return its training +
    difficulty rows. Output: dict {"_i", "train": {...}, "diff": {...}|None} or
    None if too few solvers succeeded to be useful."""
    seed = genlib.derive_seed(_SEED, index)
    rng  = random.Random(seed)
    regime = rng.choices(REGIMES, weights=REGIME_WEIGHTS)[0]
    inst_text, V, procs, beta = random_instance(rng, regime)
    feats = compute_features(V, procs, beta)
    N = len(procs)

    solvers = FAST_SOLVERS + (EXACT_SOLVERS if N <= EXACT_N_LIMIT else [])
    exact_ran = N <= EXACT_N_LIMIT

    import time as _t
    results = {}   # solver -> (makespan, wall_seconds)
    for solver in solvers:
        t0 = _t.time()
        try:
            # Pin the solver seed so stochastic solvers (GA, LP anti-cycling) give
            # reproducible labels; otherwise GASolver draws from random_device.
            res = genlib.solve(inst_text, solver, {"seed": seed % (2**63)})
            elapsed = _t.time() - t0   # wall-clock, informational only (not a filter)
            sol = res.get("solution", {})
            if sol.get("status") in ("Feasible", "Optimal"):
                results[solver] = (float(sol.get("makespan", float("inf"))), elapsed)
        except Exception:
            pass

    if len(results) < 2:
        return None

    exact_ms = results["exact"][0] if exact_ran and "exact" in results else None

    # A β > 0 instance needs the exact optimum for a correct label (β-blind
    # heuristics misreport it). If exact didn't return one, drop the instance.
    if beta > 0.0 and exact_ms is None:
        return None

    # When the exact optimum is known, any solver reporting BELOW it is invalid for
    # this instance — e.g. single-round / online ignore result-return β and misreport
    # a too-low makespan. Drop them so they can't corrupt the label.
    if exact_ms is not None:
        valid = {s: mw for s, mw in results.items() if mw[0] >= exact_ms - 1e-6}
    else:
        valid = results

    best_solver = min(valid, key=lambda s: valid[s][0])
    best_makespan, best_wall = valid[best_solver]

    prov = genlib.provenance(seed, regime,
                             oracle="exact" if exact_ms is not None else "best-heuristic",
                             label_is_exact=(exact_ms is not None),
                             wall_ms=best_wall * 1000.0)
    train_row = {**feats, "best_solver": best_solver,
                 "best_makespan": best_makespan, **prov, "_i": index}

    # Difficulty label: gap between the best VALID heuristic and the exact optimum.
    diff_row = None
    if exact_ms is not None:
        heuristic_ms = min((valid[s][0] for s in FAST_SOLVERS if s in valid),
                           default=float("inf"))
        if heuristic_ms < float("inf"):
            gap = (heuristic_ms - exact_ms) / (exact_ms + 1e-12)
            difficulty = "easy" if gap < 0.01 else "medium" if gap < 0.10 else "hard"
            dprov = genlib.provenance(seed, regime, oracle="exact",
                                      label_is_exact=True, wall_ms=best_wall * 1000.0)
            diff_row = {**feats, "difficulty": difficulty, **dprov, "_i": index}
    elif N > EXACT_N_LIMIT:
        # exact intractable; heuristic quality unknown → labelled hard (not exact).
        dprov = genlib.provenance(seed, regime, oracle="best-heuristic",
                                  label_is_exact=False, wall_ms=best_wall * 1000.0)
        diff_row = {**feats, "difficulty": "hard", **dprov, "_i": index}

    return {"_i": index, "train": train_row, "diff": diff_row}

# ── main ────────────────────────────────────────────────────────────────────

def main():
    global _SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=5000, help="number of instances")
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 1),
                    help="worker processes (1 = sequential; >1 needs multiple cores)")
    ap.add_argument("--out",     default=os.path.join(os.path.dirname(__file__), "training_data.csv"))
    args = ap.parse_args()
    _SEED = args.seed

    diff_out = os.path.join(os.path.dirname(args.out), "difficulty_data.csv")
    print(f"Generating {args.n} instances using {genlib.find_lib()}")
    print(f"Fast solvers: {FAST_SOLVERS}  |  Exact (N<={EXACT_N_LIMIT}): {EXACT_SOLVERS}")

    results = genlib.run(task, args.n, workers=args.workers,
                         progress_every=500, label="instances")

    train_rows = [r["train"] for r in results]
    diff_rows  = [r["diff"]  for r in results if r["diff"] is not None]

    train_cols = FEATURE_NAMES + ["best_solver", "best_makespan"] + genlib.PROVENANCE_COLS
    diff_cols  = FEATURE_NAMES + ["difficulty"] + genlib.PROVENANCE_COLS
    genlib.write_csv(args.out, train_rows, train_cols)
    genlib.write_csv(diff_out, diff_rows,  diff_cols)

    print(f"  {len(train_rows)} solver-selector rows → {args.out}")
    print(f"  {len(diff_rows)} difficulty rows → {diff_out}")

    exact_train = sum(r["label_is_exact"] for r in train_rows)
    print(f"  exact-labelled: {exact_train}/{len(train_rows)} training "
          f"({100*exact_train/max(len(train_rows),1):.1f}%)")

    dist = Counter(r["best_solver"] for r in train_rows)
    print("\nSolver label distribution:")
    for solver, count in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"  {solver:20s} {count:6d}  ({100*count/len(train_rows):.1f}%)")

    ddist = Counter(r["difficulty"] for r in diff_rows)
    print("\nDifficulty label distribution:")
    for lab, count in sorted(ddist.items(), key=lambda x: -x[1]):
        print(f"  {lab:20s} {count:6d}  ({100*count/max(len(diff_rows),1):.1f}%)")

if __name__ == "__main__":
    main()
