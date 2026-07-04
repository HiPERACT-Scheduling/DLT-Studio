//---------------------------------------------------------------------------
// examples/portfolio_compare.cpp
//
// Runs the DLS solver portfolio on one instance and prints a comparison table:
//   - GA            (heuristic, multi-installment, exactly L installments)
//   - Exact B&B     (branch-and-bound, multi-installment, <= L)
//   - Exact B&B     (branch-and-bound, single-installment, <= L)
//   - Exact MILP    (HiGHS monolithic MILP, single-installment, <= L)
//
// It illustrates the cross-relationships:
//   * Exact B&B (multi)  <= GA            (exact bounds the heuristic),
//   * Exact MILP (single) == Exact B&B (single)  (two exact methods agree),
//   * single-installment optimum >= multi-installment optimum.
//
// Built only with -DDLS_WITH_HIGHS=ON (it uses the HiGHS-based MilpSolver).
//---------------------------------------------------------------------------

#include <chrono>
#include <cstdio>
#include <functional>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "heuristics/ga/ga_solver.hpp"
#include "exact/enumerative/exact_solver.hpp"
#include "exact/milp/milp_solver.hpp"

using namespace dls;

// Goal:   readable name for a solve status.
static const char* statusName(SolveStatus s) {
    switch (s) {
        case SolveStatus::Optimal:    return "Optimal";
        case SolveStatus::Feasible:   return "Feasible";
        case SolveStatus::Infeasible: return "Infeasible";
        case SolveStatus::Unbounded:  return "Unbounded";
        case SolveStatus::Failure:    return "Failure";
        default:                      return "NotSolved";
    }
}

// Goal:   time a solver, print one comparison-table row.
// Input:  name/variant labels; solveFn returning a DLSSolution.
static void runRow(const char* name, const char* variant,
                   const std::function<DLSSolution()>& solveFn) {
    auto t0 = std::chrono::steady_clock::now();
    DLSSolution s = solveFn();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("%-12s %-16s %-10s %14.2f %12ld %10.1f\n",
                name, variant, statusName(s.status), s.makespan, s.iterations, ms);
}

int main() {
    // Sample instance: 5 processors {S, C, A, B}, total load V.
    DLSInstance inst({{0.0, 0.23, 0.12, 1e9},
                      {0.0, 0.11, 0.23, 1e9},
                      {0.0, 0.11, 0.11, 1e9},
                      {0.0, 0.11, 0.31, 1e9},
                      {0.0, 0.11, 0.11, 1e9}}, 1e6);
    inst.setName("canonical-5");
    const int L = 5;   // max installments

    std::printf("DLS solver portfolio  —  instance '%s', N=%zu, V=%g, L=%d\n\n",
                inst.name().c_str(), inst.numProcessors(), inst.totalLoad(), L);
    std::printf("%-12s %-16s %-10s %14s %12s %10s\n",
                "Solver", "Variant", "Status", "Makespan", "Iter/Nodes", "Time(ms)");
    std::printf("%s\n", "----------------------------------------------------------------------------");

    runRow("GA", "multi (=L)", [&] {
        GAParams p; p.installments = L; p.populationSize = 12;
        p.maxGenerations = 150; p.noImprovementLimit = 60;
        SolverConfig c; c.seed = 42;
        return GASolver(p).solve(inst, c);
    });
    runRow("Exact B&B", "multi (<=L)", [&] {
        ExactParams p; p.maxInstallments = L; p.allowRepeats = true;
        return ExactSolver(p).solve(inst, SolverConfig{});
    });
    runRow("Exact B&B", "single (<=L)", [&] {
        ExactParams p; p.maxInstallments = L; p.allowRepeats = false;
        return ExactSolver(p).solve(inst, SolverConfig{});
    });
    runRow("Exact MILP", "single (<=L)", [&] {
        MilpParams p; p.maxInstallments = L;
        return MilpSolver(p).solve(inst, SolverConfig{});
    });

    std::printf("\nNotes: Exact B&B (multi) is the optimum the GA approximates; "
                "Exact MILP (single) == Exact B&B (single).\n");
    return 0;
}
