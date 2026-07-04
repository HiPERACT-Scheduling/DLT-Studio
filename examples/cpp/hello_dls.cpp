//---------------------------------------------------------------------------
// examples/hello_dls.cpp
//
// Minimal C++ usage: build a 3-processor instance, solve it with the Auto
// solver, and print the schedule.
//
// Build (from repo root):
//   cmake -B build && cmake --build build --target hello_dls
// Run:
//   ./build/bin/hello_dls
//---------------------------------------------------------------------------

#include <cstdio>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/schedule_expand.hpp"
#include "heuristics/auto/auto_solver.hpp"

int main() {
    using namespace dls;

    // Three processors described by (S, C, A, B):
    //   S = communication startup time [s]
    //   C = communication rate         [s / data-unit]
    //   A = computation rate           [s / data-unit]
    //   B = memory limit per installment (1e18 = effectively unbounded)
    DLSInstance inst(
        {
            Processor{.commStartup=0.0, .commRate=0.1, .computeRate=0.50, .memoryLimit=1e18},  // P0
            Processor{.commStartup=0.1, .commRate=0.2, .computeRate=0.30, .memoryLimit=1e18},  // P1
            Processor{.commStartup=0.2, .commRate=0.3, .computeRate=0.20, .memoryLimit=1e18},  // P2
        },
        /*totalLoad=*/1000.0
    );

    DLSSolution sol = AutoSolver{}.solve(inst, SolverConfig{});
    expandSchedule(inst, sol);   // fill in commStart/computeStart/computeFinish timings

    if (!sol.feasible()) {
        std::puts("No feasible schedule found.");
        return 1;
    }

    std::printf("Makespan : %.4f\n", sol.makespan);
    std::printf("Sequence :");
    for (int pid : sol.sequence) std::printf(" P%d", pid);
    std::puts("");
    std::puts("Fragments:");
    for (const LoadFragment& f : sol.fragments)
        std::printf("  P%d  load=%.2f  compute [%.4f, %.4f]\n",
                    f.processorId, f.loadSize, f.computeStart, f.computeFinish);
    return 0;
}
