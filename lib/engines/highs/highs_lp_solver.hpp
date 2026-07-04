//---------------------------------------------------------------------------
// engines/highs/highs_lp_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Generic "minimize objective·x subject to rows·x <= rhs, x >= 0" LP solved by
// HiGHS. Declaration only (no HiGHS headers); the implementation is compiled
// into the dls_highs library and linked only in -DDLS_WITH_HIGHS=ON builds.
// Used by the MLSD evaluator (and reusable elsewhere) to get a HiGHS backend
// without duplicating model-passing boilerplate.
//---------------------------------------------------------------------------

#ifndef DLS_BACKENDS_HIGHS_LP_SOLVER_HPP
#define DLS_BACKENDS_HIGHS_LP_SOLVER_HPP

#include <vector>

#include "core/dls_solution.hpp"   // dls::SolveStatus

namespace dls {

struct HighsLpSolution {
    SolveStatus         status = SolveStatus::NotSolved;
    std::vector<double> values;   // variable values on Optimal
};

// Goal:   minimize objective·x subject to rows[r]·x <= rhs[r] and x >= 0.
// Input:  numVars; objective (size numVars); rows (each size numVars) and rhs;
//         epsilon - feasibility tolerance.
// Output: status + variable values (filled on Optimal).
HighsLpSolution solveLpViaHighs(int numVars,
                                const std::vector<double>& objective,
                                const std::vector<std::vector<double>>& rows,
                                const std::vector<double>& rhs,
                                double epsilon = 1e-6);

}  // namespace dls

#endif  // DLS_BACKENDS_HIGHS_LP_SOLVER_HPP
