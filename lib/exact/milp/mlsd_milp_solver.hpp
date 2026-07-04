//---------------------------------------------------------------------------
// exact/milp/mlsd_milp_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Exact MLSD solver via mixed-integer linear programming (HiGHS), an INDEPENDENT
// check on the brute-force MlsdSolver enumerator. For a fixed task order it lets
// the MILP decide, for every task, which processors take part and in what
// activation order (slot assignment binaries) plus the load fractions — exactly
// the per-task ordered-subset search that blows up factorially in the
// enumerator. The n! task orders are enumerated on top (cheap in the exact
// regime), mirroring MlsdSolver.
//
// The per-(task, slot) completion constraint of the shared LP (mlsd_evaluator)
// is linear in the assignment/load variables — the compute term is
// A_p · Σ_{ll>=l} (p's total load in task ll) — so a big-M formulation
// reproduces the evaluator's model exactly. Scope: the "related" case with no
// result return (β = 0), matching the validated MlsdSolver tests; result return
// (model A) is out of scope here.
//
// Declaration only (no HiGHS headers); compiled and linked under -DDLS_WITH_HIGHS.
//---------------------------------------------------------------------------

#ifndef DLS_EXACT_MLSD_MILP_SOLVER_HPP
#define DLS_EXACT_MLSD_MILP_SOLVER_HPP

#include <string>

#include "mlsd/mlsd_instance.hpp"

namespace dls {

struct MlsdMilpParams {
    double timeLimitSec = 0.0;   // 0 = no limit; else per-task-order MILP cap
    double mipGap       = 0.0;   // 0 = prove optimality; else relative MIP gap
};

class MlsdMilpSolver {
public:
    explicit MlsdMilpSolver(MlsdMilpParams params = {}) : params_(params) {}

    std::string name() const { return "mlsd-milp"; }

    // Goal:   solve the MLSD problem to optimality via MILP (β = 0 only).
    // Input:  inst - the MLSD instance.
    // Output: an MlsdSolution; Optimal with the best task order, per-task loads
    //         and makespan, or Infeasible/Failure.
    MlsdSolution solve(const MlsdInstance& inst) const;

private:
    MlsdMilpParams params_;
};

}  // namespace dls

#endif  // DLS_EXACT_MLSD_MILP_SOLVER_HPP
