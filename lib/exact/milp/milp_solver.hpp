//---------------------------------------------------------------------------
// exact/milp/milp_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Optional exact DLS solver that formulates the WHOLE single-installment
// problem — processor selection, activation order, and loads — as one mixed-
// integer linear program and solves it with HiGHS. Unlike ExactSolver (which
// branch-and-bounds over sequences and solves an LP per sequence), this hands
// the entire combinatorial + continuous problem to the MILP engine at once.
//
// Single-installment scope: each processor is used at most once. The model
// matches the single-installment case of the shared LP (core/dls_lp_model.hpp),
// so its optimum equals ExactSolver(allowRepeats=false) — the two exact methods
// cross-validate each other.
//
// Only the declaration lives here (no HiGHS headers); the implementation
// (highs) is compiled separately and linked only in -DDLS_WITH_HIGHS=ON builds.
//---------------------------------------------------------------------------

#ifndef DLS_EXACT_MILP_SOLVER_HPP
#define DLS_EXACT_MILP_SOLVER_HPP

#include <limits>
#include <string>
#include <utility>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"

namespace dls {

// Algorithm-specific parameters for the MILP solver.
struct MilpParams {
    int    maxInstallments = 5;     // K = min(maxInstallments, N) activation slots
    double timeLimitSec    = 0.0;   // 0 = no limit; else cap (best-so-far => Feasible)
    double mipGap          = 0.0;   // 0 = prove optimality; else relative MIP gap
    double costLimit =              // G-bar: minimize Cmax subject to cost G <= this
        std::numeric_limits<double>::infinity();   // (inf = no cost limit)
    bool   minimizeCost = false;    // true: minimize cost G subject to Cmax <= makespanLimit
    double makespanLimit =          // C-bar: makespan limit (used when minimizeCost)
        std::numeric_limits<double>::infinity();

    // Goal:   check the parameters are usable.
    // Input:  error - optional out-param set to a reason when invalid.
    // Output: true if valid, false otherwise.
    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        if (maxInstallments < 1) { if (error) *error = "maxInstallments must be >= 1"; return false; }
        if (timeLimitSec    < 0.0) { if (error) *error = "timeLimitSec must be >= 0"; return false; }
        if (mipGap          < 0.0) { if (error) *error = "mipGap must be >= 0"; return false; }
        if (error) error->clear();
        return true;
    }
};

class MilpSolver : public DLSSolver {
public:
    // Goal:   build a MILP solver with a fixed configuration.
    // Input:  params - MILP parameters (see MilpParams).
    explicit MilpSolver(MilpParams params) : params_(std::move(params)) {}

    // Goal:   stable identifier for registry / CLI selection.
    // Output: "exact-milp".
    std::string name() const override { return "exact-milp"; }

    // Goal:   report this solver's category.
    // Output: SolverCategory::Exact.
    SolverCategory category() const override { return SolverCategory::Exact; }

    // Goal:   solve the single-installment DLS problem to optimality via MILP.
    // Input:  instance - the problem; config - shared runtime config (the MILP
    //         optimum is RNG-independent; seed is unused).
    // Output: a DLSSolution. Optimal when proved; Feasible if a time/gap limit
    //         stopped it with an incumbent; Infeasible if no schedule fits
    //         (e.g. total memory < V); Failure on invalid params / solver error.
    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override;

private:
    MilpParams params_;   // immutable configuration
};

}  // namespace dls

#endif  // DLS_EXACT_MILP_SOLVER_HPP
