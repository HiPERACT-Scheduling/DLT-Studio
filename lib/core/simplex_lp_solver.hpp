//---------------------------------------------------------------------------
// core/simplex_lp_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Thin helper that solves a backend-neutral LP (objective + "coeffs·x <= rhs"
// rows) with the dependency-free CSimplex, returning a uniform result. It is the
// CSimplex counterpart of engines/highs/highs_lp_solver.hpp's solveLpViaHighs,
// so callers that already assemble dense rows (the topology allocation LPs, …)
// don't each repeat the CSimplex setup / solve / extract plumbing.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_SIMPLEX_LP_SOLVER_HPP
#define DLS_CORE_SIMPLEX_LP_SOLVER_HPP

#include <cstdint>
#include <vector>

#include "core/simplex.h"
#include "core/dls_solution.hpp"   // SolveStatus

namespace dls {

struct SimplexLpResult {
    SolveStatus         status = SolveStatus::Failure;
    std::vector<double> values;   // x[0..nVars-1] on Optimal (empty otherwise)
};

// Goal:   minimize objective·x subject to rows[i]·x <= rhs[i], x >= 0, via CSimplex.
// Input:  nVars; objective (size nVars); rows (each size nVars) and rhs; seed for
//         anti-cycling; maxIter cap. Taken by value (CSimplex's API is not
//         const-correct); callers may std::move into it.
// Output: a SimplexLpResult — Optimal with x filled, else Infeasible/Unbounded/Failure.
inline SimplexLpResult solveLpViaSimplex(int nVars, std::vector<double> objective,
                                         std::vector<std::vector<double>> rows,
                                         std::vector<double> rhs,
                                         std::uint64_t seed = 0, int maxIter = 1000) {
    SimplexLpResult r;
    CSimplex lp(nVars, static_cast<int>(rows.size()), seed);
    lp.SetMaxNumberOfIter(maxIter);
    lp.SetObjectiveFunction(objective.data());
    for (std::size_t i = 0; i < rows.size(); ++i) lp.AddConstraintLE(rows[i].data(), rhs[i]);
    switch (lp.Solve()) {
        case OPTIMAL:
            r.status = SolveStatus::Optimal;
            r.values.resize(nVars);
            for (int i = 0; i < nVars; ++i) r.values[i] = lp.GetSolution(i + 1);  // CSimplex is 1-based
            break;
        case INFEASIBLE: r.status = SolveStatus::Infeasible; break;
        case UNBOUNDED:  r.status = SolveStatus::Unbounded;  break;
        default:         r.status = SolveStatus::Failure;    break;
    }
    lp.Clear();
    return r;
}

}  // namespace dls

#endif  // DLS_CORE_SIMPLEX_LP_SOLVER_HPP
