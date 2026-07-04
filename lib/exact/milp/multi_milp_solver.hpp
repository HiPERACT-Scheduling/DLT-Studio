//---------------------------------------------------------------------------
// exact/milp/multi_milp_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Exact MULTI-installment DLS solver as a single mixed-integer linear program
// (HiGHS). Where MilpSolver allows each processor at most once, this lets a
// processor occupy several of the K = maxInstallments port slots, and models the
// per-slot finish times with a big-M same-processor FIFO coupling. It is the
// multi-installment counterpart of MilpSolver and an INDEPENDENT exact method to
// cross-check ExactSolver's branch-and-bound.
//
// Makespan model. A slot k is the k-th communication, starting at t_k; its full
// comm+compute cost is cost_k = S + (C+A)·α_k (+p). Matching the carried-load
// semantics of the shared LP (core/dls_lp_model.hpp), a chunk on a processor
// adds its cost AFTER the previous chunk on that processor finished (no same-
// processor comm/compute overlap):
//   f_k = max(t_k,  f_prev) + cost_k,
// where prev is the previous slot on the same processor. Hence on instances
// where the per-chunk buffer is the only binding memory limit (ample Bᵢ, the
// usual pipelining-for-speed case), this MILP's optimum EQUALS
// ExactSolver(allowRepeats=true). The full carried-load buffer coupling under
// tight memory (αᵢ + yᵢ ≤ Bᵢ) is NOT modeled here (only the per-chunk αₖ ≤ Bᵢ);
// that case is deferred. Cost/availability (r,d) are likewise out of scope.
//
// Declaration only (no HiGHS headers); the implementation is compiled and linked
// solely in -DDLS_WITH_HIGHS=ON builds.
//---------------------------------------------------------------------------

#ifndef DLS_EXACT_MULTI_MILP_SOLVER_HPP
#define DLS_EXACT_MULTI_MILP_SOLVER_HPP

#include <string>
#include <utility>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"

namespace dls {

// Algorithm-specific parameters for the multi-installment MILP.
struct MultiMilpParams {
    int    maxInstallments = 5;     // K = number of port slots (a processor may repeat)
    double timeLimitSec    = 0.0;   // 0 = no limit; else cap (best-so-far => Feasible)
    double mipGap          = 0.0;   // 0 = prove optimality; else relative MIP gap

    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        if (maxInstallments < 1) { if (error) *error = "maxInstallments must be >= 1"; return false; }
        if (timeLimitSec    < 0.0) { if (error) *error = "timeLimitSec must be >= 0"; return false; }
        if (mipGap          < 0.0) { if (error) *error = "mipGap must be >= 0"; return false; }
        if (error) error->clear();
        return true;
    }
};

class MultiMilpSolver : public DLSSolver {
public:
    explicit MultiMilpSolver(MultiMilpParams params) : params_(std::move(params)) {}

    std::string    name() const override     { return "milp-multi"; }
    SolverCategory category() const override  { return SolverCategory::Exact; }

    // Goal:   solve the multi-installment DLS makespan problem via MILP.
    // Input:  instance - the problem; config - runtime (MILP is RNG-independent).
    // Output: a DLSSolution; Optimal when proved, Feasible on a time/gap limit,
    //         Infeasible / Failure otherwise.
    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override;

private:
    MultiMilpParams params_;
};

}  // namespace dls

#endif  // DLS_EXACT_MULTI_MILP_SOLVER_HPP
