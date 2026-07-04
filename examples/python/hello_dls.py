#!/usr/bin/env python3
"""
examples/python/hello_dls.py

Solve a 3-processor instance via the C-ABI binding (libdls_c.so).
Uses only the Python standard library — no pip installs required.

Run:
    DLS_LIB=../../build/bin/libdls_c.so python hello_dls.py
"""

import ctypes
import glob
import json
import os

# ── locate the shared library ──────────────────────────────────────────────
path = os.environ.get("DLS_LIB")
if not path:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    hits = sorted(glob.glob(os.path.join(root, "build*", "bin", "libdls_c.*")))
    if not hits:
        raise RuntimeError(
            "libdls_c not found. Build it first:\n"
            "  cmake -B build && cmake --build build --target dls_c\n"
            "or set DLS_LIB to its path."
        )
    path = hits[0]

lib = ctypes.CDLL(path)
lib.dls_solve.restype  = ctypes.c_void_p
lib.dls_solve.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
lib.dls_free.argtypes  = [ctypes.c_void_p]


def solve(instance_text: str, solver: str = "auto", opts: str = "") -> dict:
    """Call dls_solve and return the parsed JSON result dict."""
    ptr = lib.dls_solve(instance_text.encode(), solver.encode(), opts.encode())
    raw = ctypes.cast(ptr, ctypes.c_char_p).value
    lib.dls_free(ptr)
    return json.loads(raw)


# ── define the instance ────────────────────────────────────────────────────
# Text format:
#   V <totalLoad>
#   <S> <C> <A> <B> <p> <r> <d> <f> <l>   (one line per processor)
#
# S = comm startup [s], C = comm rate [s/unit], A = compute rate [s/unit]
# B = 1e18 (unbounded buffer), p/r/d/f/l = 0 (unused here)
instance = """\
V 1000.0
0.0 0.1 0.5 1e+18 0.0 0.0 0.0 0.0 0.0
0.1 0.2 0.3 1e+18 0.0 0.0 0.0 0.0 0.0
0.2 0.3 0.2 1e+18 0.0 0.0 0.0 0.0 0.0
"""

# ── solve and print ────────────────────────────────────────────────────────
result = solve(instance)
sol = result["solution"]

print(f"Status   : {sol['status']}")
print(f"Makespan : {sol['makespan']:.4f}")
print(f"Sequence : {' '.join(f'P{p}' for p in sol['sequence'])}")
print("Fragments:")
for f in sol["fragments"]:
    print(f"  P{f['proc']}  load={f['load']:.2f}"
          f"  compute [{f['computeStart']:.4f}, {f['computeFinish']:.4f}]")
