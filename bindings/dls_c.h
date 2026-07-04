//---------------------------------------------------------------------------
// bindings/dls_c.h
//
// Public C header for the DLS scheduling library C-ABI binding (libdls_c.so).
// Any language with a C FFI can use this header to drive the full solver
// portfolio without linking against C++ or Python.
//
// Contract:
//   - Instances go IN as the library's text format (see below).
//   - Solver options go in as a flat "key=value;key=value" string (or NULL).
//   - Results come OUT as a heap-allocated JSON string.
//   - Every non-void function returns a heap string that MUST be released with
//     dls_free() — failure to do so is a memory leak.
//
// Instance text format (same as core/instance_io.hpp):
//   V <totalLoad>
//   <S> <C> <A> <B> <p> <r> <d> <f> <l>     ← one line per processor
//   [power <P_idle> <P_startup> <P_network>]  ← optional energy state powers
//   [energy <intercept,slope> ...]            ← optional piecewise energy ε(α)
//
// Common option keys (all optional):
//   maxInstallments=5   nodeBudget=100000   seed=1
//   deadline=50.0       epsilon=0.1         backend=simplex|highs
//   allowRepeats=0|1    chunk=ssc|gss       psr=compute|comm|...
//---------------------------------------------------------------------------

#ifndef DLS_C_H
#define DLS_C_H

#ifdef __cplusplus
extern "C" {
#endif

// Goal:   list the solver IDs available in this build.
// Input:  none.
// Output: heap JSON {"solvers":[{"id":"auto","label":"..."},...]}
char* dls_solvers(void);

// Goal:   solve an instance and return the schedule.
// Input:  instanceText - problem definition (text format).
//         solverName   - solver ID from dls_solvers() (NULL or "" = "auto").
//         optsText     - algorithm options ("key=value;..." or NULL).
// Output: heap JSON {"solver":"...","solution":{...fragments...}}
char* dls_solve(const char* instanceText, const char* solverName, const char* optsText);

// Goal:   compute the time-energy Pareto front for an instance.
// Input:  instanceText - same as dls_solve.
//         solverName   - base solver used to evaluate each point.
//         optsText     - options; "points=N" controls front density.
// Output: heap JSON {"front":[{"makespan":...,"energy":...},...]}
char* dls_pareto(const char* instanceText, const char* solverName, const char* optsText);

// Goal:   compute an isoefficiency / isoenergy heatmap over a parameter range.
// Input:  optsText - axis names, ranges, metric, and base instance params.
//                   Keys: x, y (procs|load|comm|compute|startup|...), xmin,
//                   xmax, xsteps, ymin, ymax, ysteps, metric (makespan|energy).
// Output: heap JSON {"xAxis":"...","yAxis":"...","xs":[...],"ys":[...],"grid":[[...]]}
char* dls_map(const char* optsText);

// Goal:   benchmark the solver portfolio over random instances.
// Input:  optsText - keys: solvers (comma list), procs, load, memory,
//                   instances, seed, homogeneous (0|1), installments.
// Output: heap JSON {instances, provenOptimal, solvers:[{name, feasible,...}]}
char* dls_bench(const char* optsText);

// Goal:   parse and solve a network topology (chain, tree, or general graph).
// Input:  klass    - "chain", "tree", or "graph".
//         text     - topology description in the matching text format.
//         optsText - solver options.
// Output: heap JSON with the topology schedule.
char* dls_topology(const char* klass, const char* text, const char* optsText);

// Goal:   release a heap string returned by any function above.
// Input:  p - pointer returned by a previous call (NULL is safe).
void  dls_free(char* p);

#ifdef __cplusplus
}
#endif

#endif /* DLS_C_H */
