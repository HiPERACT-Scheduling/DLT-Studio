#!/usr/bin/env python3
"""
tools/genlib.py

Shared harness for the ML training-data generators. It centralises the three
things every generator needs so each generator is reduced to "sample an instance,
call an oracle, emit a row":

  1. libdls_c loading — one CDLL handle per worker process (lazy, fork-safe).
  2. A parallel driver — maps a picklable per-index task over a worker pool,
     with progress/ETA reporting; falls back to a plain sequential loop when
     workers <= 1 (avoids pool/IPC overhead on a single-core host).
  3. Provenance — every row carries how it was labelled, so downstream training
     can filter to exact-oracle labels or weight by confidence instead of
     silently mixing exact and heuristic labels:
         seed            derived per-instance seed (reproduce one row)
         regime          instance-family index the sampler drew
         oracle          solver name whose objective became the label
         label_is_exact  1 if that oracle is provably optimal here, else 0
         solver_wall_ms  oracle wall-clock (ms)

Determinism: each instance index gets a seed derived from the master seed alone
(derive_seed), so the dataset is reproducible regardless of worker count or the
order the pool happens to finish tasks in.

Note on this deployment: the generation host has a single core, so `--workers`
above 1 yields no wall-clock speedup here (the solvers are CPU-bound). The pool
path exists so the same generator scales when run on a multi-core machine.
"""

import ctypes
import glob
import json
import os
import time

# ── libdls_c loading (one handle per process) ──────────────────────────────

_LIB = None   # per-process cache; stays None until first solve() in each worker

def find_lib():
    """Goal: locate the built libdls_c shared object. Output: absolute path."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for pattern in ["build-highs/bin/libdls_c.*", "build*/bin/libdls_c.*"]:
        hits = sorted(glob.glob(os.path.join(root, pattern)))
        if hits:
            return hits[0]
    raise RuntimeError("libdls_c not found — build with cmake first")

def get_lib():
    """Goal: return this process's CDLL handle, loading it once. Output: CDLL."""
    global _LIB
    if _LIB is None:
        lib = ctypes.CDLL(os.environ.get("DLS_LIB") or find_lib())
        lib.dls_solve.restype       = ctypes.c_void_p
        lib.dls_solve.argtypes      = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.dls_mlsd_solve.restype  = ctypes.c_void_p
        lib.dls_mlsd_solve.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        lib.dls_free.argtypes       = [ctypes.c_void_p]
        _LIB = lib
    return _LIB

def _encode_opts(opts_dict):
    """Goal: encode options in the C binding's "key=value;..." format (NOT JSON).
    Output: bytes."""
    if not opts_dict:
        return b""
    return ";".join(f"{k}={v}" for k, v in opts_dict.items()).encode()

def solve(instance_text, solver, opts_dict=None):
    """Goal: run a single-load DLS solver. Output: parsed JSON response dict."""
    lib  = get_lib()
    ptr  = lib.dls_solve(instance_text.encode(), solver.encode(), _encode_opts(opts_dict))
    raw  = ctypes.cast(ptr, ctypes.c_char_p).value
    lib.dls_free(ptr)
    return json.loads(raw)

def mlsd_solve(instance_text, solver, opts_dict=None):
    """Goal: run an MLSD solver. Output: parsed JSON response dict."""
    lib  = get_lib()
    ptr  = lib.dls_mlsd_solve(instance_text.encode(), solver.encode(), _encode_opts(opts_dict))
    raw  = ctypes.cast(ptr, ctypes.c_char_p).value
    lib.dls_free(ptr)
    return json.loads(raw)

# ── deterministic per-index seeding ────────────────────────────────────────

def derive_seed(master, index):
    """Goal: a stable seed for instance `index` given the master seed, so output
    is reproducible independent of worker count/order. Output: 64-bit int."""
    x  = ((master & 0xFFFFFFFF) * 0x9E3779B1 + index * 0x85EBCA77 + 0x165667B1)
    x &= 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 31)
    x  = (x * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 27)
    return x

# ── provenance ─────────────────────────────────────────────────────────────

PROVENANCE_COLS = ["seed", "regime", "oracle", "label_is_exact", "solver_wall_ms"]

def provenance(seed, regime, oracle, label_is_exact, wall_ms):
    """Goal: the provenance fields for one row. Output: dict over PROVENANCE_COLS."""
    return {
        "seed": seed,
        "regime": regime,
        "oracle": oracle,
        "label_is_exact": 1 if label_is_exact else 0,
        "solver_wall_ms": round(wall_ms, 3),
    }

# ── parallel driver ────────────────────────────────────────────────────────

def run(task_fn, n, workers=1, progress_every=2000, label="instances"):
    """
    Goal:   evaluate task_fn(index) for index in [0, n) and collect the results.
    Input:  task_fn - a *module-level* (picklable) function returning a result
                      dict (which must carry key "_i" = its index) or None to skip;
            n - instance count; workers - process count (<=1 ⇒ sequential);
            progress_every - print an ETA line every this many completed tasks.
    Output: list of result dicts, sorted by "_i" so file order is deterministic.
    """
    results = []
    t0      = time.time()
    done    = 0

    def _tick():
        el   = time.time() - t0
        rate = done / el if el > 0 else 0.0
        eta  = (n - done) / rate if rate > 0 else 0.0
        print(f"  {done}/{n}  kept={len(results)}  {el:.0f}s  "
              f"ETA {eta:.0f}s  ({rate:.0f}/s)", flush=True)

    if workers and workers > 1:
        import multiprocessing as mp
        with mp.Pool(workers) as pool:
            for res in pool.imap_unordered(task_fn, range(n), chunksize=16):
                done += 1
                if res is not None:
                    results.append(res)
                if done % progress_every == 0:
                    _tick()
    else:
        for i in range(n):
            res  = task_fn(i)
            done += 1
            if res is not None:
                results.append(res)
            if done % progress_every == 0:
                _tick()

    results.sort(key=lambda r: r.get("_i", 0))
    el = time.time() - t0
    print(f"\nGenerated {len(results)}/{n} {label} in {el:.0f}s "
          f"({(n/el if el>0 else 0):.0f}/s, {workers or 1} worker(s)).")
    return results

# ── CSV output ─────────────────────────────────────────────────────────────

def write_csv(path, rows, columns):
    """
    Goal:   write `rows` (list of dicts) to `path` with `columns` as the header.
    Input:  columns - explicit column order; keys starting with "_" are internal
            (e.g. "_i") and never written.
    Output: number of rows written.
    """
    cols = [c for c in columns if not c.startswith("_")]
    with open(path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) for c in cols) + "\n")
    return len(rows)
