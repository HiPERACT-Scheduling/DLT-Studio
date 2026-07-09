#!/usr/bin/env python3
"""
tools/generate_energy_training_data.py

Generate exact-labelled training data for the ML minimum-energy predictor.

For each random energy-model DLSInstance (per-processor power states P^I/P^S/P^N,
a running-energy slope k, and originator power), calls the single-installment
exact-milp solver with minimizeCost=1 — which, when usesEnergyModel() is true,
minimizes the four-state energy directly (core/dls_lp_model.hpp) instead of the
default Cmax objective. This mode was previously unreachable from any solver
name/CLI flag (only via constructing MilpParams{minimizeCost=true} directly in
C++); it is now exposed through the existing "exact-milp" solver via the
minimizeCost=1 option (see core/solver_registry.hpp, bindings/dls_c.cpp).

Scope of this first version: single-installment (each processor used at most
once — the MilpSolver's scope), no result-return (beta=0, kept as a separate
dimension already covered by generate_training_data.py), no piecewise compute
time (energy mode does not support that combination — milp_solver.cpp fails
if both are set). Every processor gets a single energy piece {intercept 0,
slope k}, the classic proportional running-energy model (thesis Ch.3 baseline).

Every row carries provenance columns (genlib.PROVENANCE_COLS); the oracle is
"exact-milp-energy" and label_is_exact is always 1 (HiGHS MIP, mipGap=0 -> proven
optimal), gated by a deterministic N ceiling (see --exact-max-n) since exact-milp
cost grows steeply with N (calibrated: N=3 ~0.1s, N=5 ~0.6s, N=7 ~3s, N=8 ~5s,
N=9 already exceeds 20s) — the energy mode's extra McCormick/used_i columns make
it slower than the plain Cmax MILP at the same N, so the default gate is lower.

Output: tools/energy_training_data.csv

Usage:
    python3 tools/generate_energy_training_data.py [--n 20000] [--exact-max-n 8] [--workers N]
"""

import argparse
import os
import math
import random

import genlib   # shared harness: lib loading, parallel driver, provenance, CSV

REGIMES = [0, 1, 2]
WEIGHTS = [0.4, 0.4, 0.2]

FEATURE_NAMES = ["N", "loadPerProc", "meanA", "heteroA", "meanC", "heteroC",
                 "hasStartups", "meanPowerIdle", "meanPowerStartup", "meanPowerNetwork",
                 "heteroPowerNetwork", "meanEnergySlope", "heteroEnergySlope",
                 "originatorPowerNetwork", "originatorPowerIdle"]

# ── instance generation ────────────────────────────────────────────────────

def random_energy_instance(rng, regime):
    """Return (instance_text, N, feature_inputs_dict). Ample memory, single
    installment, beta = 0 — isolates the energy dimension from the others."""
    if regime == 0:                                    # small, mild heterogeneity
        N = rng.randint(2, 5)
        S_range, k_range = (0.0, 0.3), (0.2, 2.0)
    elif regime == 1:                                   # with startups
        N = rng.randint(2, 6)
        S_range, k_range = (0.0, 1.0), (0.1, 3.0)
    else:                                                # larger N, wider energy spread
        N = rng.randint(4, 8)
        S_range, k_range = (0.0, 0.6), (0.05, 4.0)

    procs = []
    for _ in range(N):
        S = rng.uniform(*S_range)
        C = rng.uniform(0.01, 0.8)
        A = rng.uniform(0.1, 2.0)
        pI = rng.uniform(0.5, 5.0)
        pS = rng.uniform(0.5, 6.0)
        pN = rng.uniform(0.5, 6.0)
        k  = rng.uniform(*k_range)
        procs.append({"S": S, "C": C, "A": A, "pI": pI, "pS": pS, "pN": pN, "k": k})

    origPN = rng.uniform(0.5, 6.0)
    origPI = rng.uniform(0.5, 4.0)
    V = rng.uniform(50, 2000)

    lines = [f"V {V:.6f}"]
    for p in procs:
        lines.append(f"{p['S']:.6f} {p['C']:.6f} {p['A']:.6f} 1.0e18")
        lines.append(f"power {p['pI']:.6f} {p['pS']:.6f} {p['pN']:.6f}")
        lines.append(f"energy 0.0,{p['k']:.6f}")
    lines.append(f"originator {origPN:.6f} {origPI:.6f}")
    return "\n".join(lines), N, V, procs, origPN, origPI

def _cv(vals):
    m = sum(vals) / len(vals)
    if m < 1e-12:
        return 0.0
    var = sum((v - m) ** 2 for v in vals) / len(vals)
    return math.sqrt(max(var, 0.0)) / m

def compute_energy_features(N, V, procs, origPN, origPI):
    A  = [p["A"] for p in procs]
    C  = [p["C"] for p in procs]
    S  = [p["S"] for p in procs]
    pI = [p["pI"] for p in procs]
    pS = [p["pS"] for p in procs]
    pN = [p["pN"] for p in procs]
    k  = [p["k"] for p in procs]
    return {
        "N": float(N),
        "loadPerProc": V / N,
        "meanA": sum(A) / N,
        "heteroA": _cv(A),
        "meanC": sum(C) / N,
        "heteroC": _cv(C),
        "hasStartups": 1.0 if any(s > 0 for s in S) else 0.0,
        "meanPowerIdle": sum(pI) / N,
        "meanPowerStartup": sum(pS) / N,
        "meanPowerNetwork": sum(pN) / N,
        "heteroPowerNetwork": _cv(pN),
        "meanEnergySlope": sum(k) / N,
        "heteroEnergySlope": _cv(k),
        "originatorPowerNetwork": origPN,
        "originatorPowerIdle": origPI,
    }

# ── per-instance task (runs in a worker process) ───────────────────────────

_SEED         = 42
_EXACT_MAX_N  = 8    # deterministic size gate: exact-milp energy mode for N <= this (N=9+ exceeds 20s)

def task(index):
    """Goal: sample one energy instance, solve it exactly for min energy,
    return its row. Output: dict {features..., energy, makespan, provenance...,
    "_i"} or None if N exceeds the gate or the solver fails."""
    import time as _t
    seed = genlib.derive_seed(_SEED, index)
    rng  = random.Random(seed)
    regime = rng.choices(REGIMES, weights=WEIGHTS)[0]
    text, N, V, procs, origPN, origPI = random_energy_instance(rng, regime)
    if N > _EXACT_MAX_N:
        return None
    feats = compute_energy_features(N, V, procs, origPN, origPI)

    t0 = _t.time()
    try:
        res = genlib.solve(text, "exact-milp", {"minimizeCost": 1})
        elapsed = _t.time() - t0
        sol = res.get("solution", {})
        if sol.get("status") != "Optimal":
            return None
        energy = float(sol.get("energy", 0.0))
        makespan = float(sol.get("makespan", 0.0))
        if energy <= 0:
            return None
    except Exception:
        return None

    prov = genlib.provenance(seed, regime, oracle="exact-milp-energy",
                             label_is_exact=True, wall_ms=elapsed * 1000.0)
    return {**feats, "energy": energy, "makespan": makespan, **prov, "_i": index}

# ── main ────────────────────────────────────────────────────────────────────

def main():
    global _SEED, _EXACT_MAX_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--n",            type=int, default=20000)
    ap.add_argument("--seed",         type=int, default=42)
    ap.add_argument("--workers",      type=int, default=(os.cpu_count() or 1))
    ap.add_argument("--exact-max-n",  type=int, default=8,
                    help="use exact-milp (minimizeCost) for N <= this; larger "
                         "instances are skipped (no energy-aware heuristic oracle exists yet)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                  "energy_training_data.csv"))
    args = ap.parse_args()
    _SEED        = args.seed
    _EXACT_MAX_N = args.exact_max_n

    print(f"Generating {args.n} energy instances using {genlib.find_lib()}")
    print(f"Oracle: exact-milp (minimizeCost=1), N <= {_EXACT_MAX_N} (single-installment, beta=0)")
    results = genlib.run(task, args.n, workers=args.workers,
                         progress_every=1000, label="energy instances")

    cols = FEATURE_NAMES + ["energy", "makespan"] + genlib.PROVENANCE_COLS
    genlib.write_csv(args.out, results, cols)
    print(f"{len(results)} rows → {args.out} (all exact; {args.n - len(results)} skipped: N > gate or solver failure)")

if __name__ == "__main__":
    main()
