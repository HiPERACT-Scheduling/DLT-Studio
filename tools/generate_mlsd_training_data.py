#!/usr/bin/env python3
"""
tools/generate_mlsd_training_data.py

Generate labelled training data for the ML MLSD Cmax predictor.

For each random MlsdInstance, runs mlsd-exact (small) or mlsd-ga (large) via
the dls_mlsd_solve C binding and records the achieved Cmax.

MLSD instance text format (cli/class_io.hpp readMlsdInstance):
  task <size> [beta]
  proc <S> <C> <A> <B>

Every row carries provenance columns (genlib.PROVENANCE_COLS): the derived seed,
the regime, the oracle (mlsd-exact vs mlsd-ga), whether the label is exact
(mlsd-exact ran, i.e. nTasks·nProcs <= 4), and the oracle wall-clock. Downstream
training can filter on label_is_exact to keep only proven-optimal Cmax labels.

Result-return (beta): mlsd_evaluator.hpp models beta correctly for both mlsd-exact
and mlsd-ga (they share the same LP evaluator), so beta > 0 is safe there. It is
sampled only on ample-memory regimes: the return-communication term sums a
per-processor-slot startup cost across every (task, slot) in the whole schedule,
so on memory-limited instances (many slots, forced by tight buffers) beta can
blow up Cmax the same way it did for the single-load generator before that fix.
mlsd-milp does not model beta at all (see exact/milp/mlsd_milp_solver.hpp) and is
never selected when beta > 0, regardless of its size gate.

Output: tools/mlsd_training_data.csv

Usage:
    python3 tools/generate_mlsd_training_data.py [--n 3000] [--seed 42] [--workers N]
"""

import argparse
import os
import math
import random

import genlib   # shared harness: lib loading, parallel driver, provenance, CSV

EXACT_LIMIT = 4     # nTasks × nProcs <= this → use mlsd-exact (else mlsd-ga)

REGIMES = [0, 1, 2, 3]
WEIGHTS = [0.25, 0.25, 0.25, 0.25]

FEATURE_NAMES = ["nTasks", "nProcs", "meanV", "cvV", "maxMinV", "totalLoadPerProc",
                 "memoryRatio", "meanA", "heteroA", "speedupA", "hasStartups", "hasCommCost",
                 "beta"]

# Regimes with ample memory (B = 1e18) — safe to add beta to. Regime 2 is
# memory-limited and stays beta = 0 (see the module docstring for why).
AMPLE_MEMORY_REGIMES = {0, 1, 3}

# ── instance generation ────────────────────────────────────────────────────

def random_mlsd_instance(rng, regime):
    """Return (instance_text, n_tasks, n_procs, tasks, procs, beta)."""
    beta = 0.0
    if regime in AMPLE_MEMORY_REGIMES and rng.random() < 0.4:
        beta = rng.uniform(0.1, 0.6)

    if regime == 0:
        n, m = rng.randint(2, 4), rng.randint(2, 6)
        tasks = [{"V": rng.uniform(100, 3000), "beta": beta} for _ in range(n)]
        procs = [{"S": 0.0, "C": rng.uniform(0.01, 1.0),
                  "A": rng.uniform(0.1, 2.0), "B": 1e18} for _ in range(m)]
    elif regime == 1:
        n, m = rng.randint(2, 5), rng.randint(2, 5)
        tasks = [{"V": rng.uniform(100, 5000), "beta": beta} for _ in range(n)]
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
        tasks = [{"V": rng.uniform(50, 8000), "beta": beta} for _ in range(n)]
        procs = [{"S": rng.uniform(0, 1.0) if rng.random() > 0.4 else 0.0,
                  "C": rng.uniform(0, 1.0), "A": rng.uniform(0.05, 3.0), "B": 1e18}
                 for _ in range(m)]

    lines = []
    for t in tasks:
        lines.append(f"task {t['V']:.6f} {t['beta']:.6f}")
    for p in procs:
        lines.append(f"proc {p['S']:.6f} {p['C']:.6f} {p['A']:.6f} {p['B']:.6e}")
    return "\n".join(lines), n, m, tasks, procs, beta

def compute_mlsd_features(tasks, procs, beta=0.0):
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

    return {
        "nTasks": float(n),
        "nProcs": float(m),
        "meanV": mV,
        "cvV": cv(V),
        "maxMinV": maxMinV,
        "totalLoadPerProc": sumV / m,
        "memoryRatio": memRatio,
        "meanA": mA,
        "heteroA": cv(A),
        "speedupA": speedupA,
        "hasStartups": 1.0 if any(s > 0 for s in S) else 0.0,
        "hasCommCost": 1.0 if any(c > 0 for c in C) else 0.0,
        "beta": beta,
    }

# ── oracle policy ──────────────────────────────────────────────────────────
# The oracle that sets each Cmax label is chosen by a *deterministic size gate*
# (never wall-clock) so labels are reproducible regardless of host/worker count:
#
#   mlsd-milp  (exact, HiGHS)   when n <= MILP_MAX_N and n*m <= MILP_MAX_NM
#   mlsd-exact (exact, brute)   when n*m <= EXACT_LIMIT
#   mlsd-ga    (heuristic)      otherwise
#
# The MILP gate is OFF by default (MILP_MAX_N = 0): mlsd-milp is validated correct
# (matches mlsd-exact) but each solve costs seconds — tractable only for tiny
# (n, m) on a single core, so enable it via --milp-max-n / --milp-max-nm when
# generating on capable hardware. `label_is_exact` marks the exact-oracle rows.

_SEED        = 42   # overwritten in main() before the pool forks; inherited by workers
_MILP_MAX_N  = 0    # enable mlsd-milp for n <= this   (0 = MILP oracle disabled)
_MILP_MAX_NM = 0    # ... and n*m <= this

def choose_oracle(n, m, beta):
    """Goal: pick the label oracle for an (n tasks, m procs) instance by the
    deterministic size gate. Output: (solver_name, label_is_exact).
    mlsd-milp does not model result-return at all (mlsd_milp_solver.hpp: "beta = 0
    only"), so it is never selected when beta > 0, regardless of its size gate."""
    if beta == 0.0 and _MILP_MAX_N and n <= _MILP_MAX_N and n * m <= _MILP_MAX_NM:
        return "mlsd-milp", True
    if n * m <= EXACT_LIMIT:
        return "mlsd-exact", True
    return "mlsd-ga", False

def task(index):
    """Goal: sample one MLSD instance, run the oracle, and return its row.
    Output: dict {feature..., cmax, provenance..., "_i"} or None on failure."""
    import time as _t
    seed = genlib.derive_seed(_SEED, index)
    rng  = random.Random(seed)
    regime = rng.choices(REGIMES, weights=WEIGHTS)[0]
    inst_text, n, m, tasks, procs, beta = random_mlsd_instance(rng, regime)
    feats = compute_mlsd_features(tasks, procs, beta)

    solver, exact = choose_oracle(n, m, beta)

    t0 = _t.time()
    try:
        res = genlib.mlsd_solve(inst_text, solver)
        elapsed = _t.time() - t0   # wall-clock, informational only (not a filter)
        if res.get("status") not in ("Optimal", "Feasible"):
            return None
        cmax = float(res["makespan"])
        if cmax <= 0:
            return None
    except Exception:
        return None

    prov = genlib.provenance(seed, regime, oracle=solver,
                             label_is_exact=exact, wall_ms=elapsed * 1000.0)
    return {**feats, "cmax": cmax, **prov, "_i": index}

# ── main ────────────────────────────────────────────────────────────────────

def main():
    global _SEED, _MILP_MAX_N, _MILP_MAX_NM
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",       type=int, default=3000)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 1),
                    help="worker processes (1 = sequential; >1 needs multiple cores)")
    ap.add_argument("--milp-max-n",  type=int, default=0,
                    help="use exact mlsd-milp when nTasks <= this (0 = disabled). "
                         "Each MILP costs seconds; enable only on capable hardware.")
    ap.add_argument("--milp-max-nm", type=int, default=12,
                    help="... and additionally require nTasks*nProcs <= this")
    ap.add_argument("--out",     default=os.path.join(os.path.dirname(__file__),
                                                      "mlsd_training_data.csv"))
    args = ap.parse_args()
    _SEED        = args.seed
    _MILP_MAX_N  = args.milp_max_n
    _MILP_MAX_NM = args.milp_max_nm

    oracle_desc = (f"mlsd-milp (n<={_MILP_MAX_N}, n*m<={_MILP_MAX_NM}) → "
                   if _MILP_MAX_N else "")
    print(f"Generating {args.n} MLSD instances using {genlib.find_lib()}")
    print(f"Oracle policy: {oracle_desc}mlsd-exact (n*m<={EXACT_LIMIT}) → mlsd-ga")
    results = genlib.run(task, args.n, workers=args.workers,
                         progress_every=200, label="MLSD instances")

    cols = FEATURE_NAMES + ["cmax"] + genlib.PROVENANCE_COLS
    genlib.write_csv(args.out, results, cols)

    exact_n = sum(r["label_is_exact"] for r in results)
    print(f"{len(results)} rows → {args.out}")
    print(f"  exact-labelled: {exact_n}/{len(results)} "
          f"({100*exact_n/max(len(results),1):.1f}%)")

if __name__ == "__main__":
    main()
