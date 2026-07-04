//---------------------------------------------------------------------------
// heuristics/auto/auto_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// AutoSolver: a meta-solver that inspects a single-load instance and delegates
// to whichever portfolio solver fits its features best, so a caller who doesn't
// want to choose can just use "auto". The policy (see docs/CHOOSING.md):
//
//   - small instance (N <= smallThreshold)         -> exact B&B (proven optimum)
//   - memory-limited (Σ Bᵢ < V, single round can't  -> best rate (fast,
//     carry the load)                                  multi-installment)
//   - ample memory, no startups (all Sᵢ = 0)        -> single round (exact here,
//                                                       instant, closed form)
//   - ample memory, with startups                   -> GA (general quality)
//
// If the chosen solver fails to find a feasible schedule, it falls back to best
// rate (which handles arbitrary multi-installment). The solver it delegated to
// is reported by chosenSolver() after solve() (useful for logging).
//
// It builds the concrete solvers directly (not via the registry) so the
// registry can register "auto" without an include cycle.
//---------------------------------------------------------------------------

#ifndef DLS_HEURISTICS_AUTO_SOLVER_HPP
#define DLS_HEURISTICS_AUTO_SOLVER_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"
#include "heuristics/ga/ga_solver.hpp"
#include "heuristics/best_rate/best_rate_solver.hpp"
#include "heuristics/single_round/single_round_solver.hpp"
#include "exact/enumerative/exact_solver.hpp"

namespace dls {

struct AutoParams {
    int smallThreshold = 6;     // N <= this ⇒ solve exactly (proven optimum)
    int maxInstallments = 5;    // base search depth (raised for memory-limited exact)
    std::string evaluatorBackend = "simplex";   // LP backend forwarded to the pick

    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (smallThreshold  < 1) return fail("smallThreshold must be >= 1");
        if (maxInstallments < 1) return fail("maxInstallments must be >= 1");
        if (error) error->clear();
        return true;
    }
};

class AutoSolver : public DLSSolver {
public:
    explicit AutoSolver(AutoParams params = {}) : params_(std::move(params)) {}

    std::string    name() const override     { return "auto"; }
    SolverCategory category() const override  { return SolverCategory::Heuristic; }

    // Goal:   the solver this last delegated to (set during solve()).
    const std::string& chosenSolver() const { return chosen_; }

    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override {
        DLSSolution result;
        std::string error;
        if (!instance.validate(&error)) { result.status = SolveStatus::Infeasible; return result; }
        if (!params_.validate(&error))  { result.status = SolveStatus::Failure;    return result; }

        // ---- instance features -------------------------------------------
        const int    N = static_cast<int>(instance.numProcessors());
        const double V = instance.totalLoad();
        double sumB = 0.0, maxB = 0.0;
        bool allZeroStartup = true, anyUnbounded = false;
        for (const Processor& p : instance.processors()) {
            if (p.memoryLimit <= 0.0) anyUnbounded = true;
            else { sumB += p.memoryLimit; maxB = std::max(maxB, p.memoryLimit); }
            if (p.commStartup != 0.0) allZeroStartup = false;
        }
        if (anyUnbounded) sumB = std::numeric_limits<double>::infinity();

        // ---- pick a solver -----------------------------------------------
        if (N <= params_.smallThreshold) {
            chosen_ = "exact";
            ExactParams ep;
            ep.allowRepeats     = true;
            ep.evaluatorBackend = params_.evaluatorBackend;
            const int needed = (maxB > 0.0) ? static_cast<int>(std::ceil(V / maxB)) : 1;
            ep.maxInstallments = std::min(7, std::max({N, needed, params_.maxInstallments}));
            result = ExactSolver(ep).solve(instance, config);
        } else if (sumB < V) {
            chosen_ = "best-rate";
            result = bestRate(instance, config);
        } else if (allZeroStartup) {
            chosen_ = "single-round";
            result = SingleRoundSolver().solve(instance, config);
        } else {
            chosen_ = "ga";
            GAParams gp;
            gp.installments       = std::max(N, params_.maxInstallments);
            gp.populationSize     = 12;
            gp.maxGenerations     = 200;
            gp.noImprovementLimit = 50;
            gp.evaluatorBackend   = params_.evaluatorBackend;
            result = GASolver(gp).solve(instance, config);
        }

        // ---- safety net: best rate handles any multi-installment case -----
        if (!result.feasible() && chosen_ != "best-rate") {
            DLSSolution alt = bestRate(instance, config);
            if (alt.feasible()) { chosen_ += "→best-rate"; result = alt; }
        }
        return result;
    }

private:
    AutoParams  params_;
    std::string chosen_;   // which solver solve() delegated to (mutable run state)

    DLSSolution bestRate(const DLSInstance& instance, const SolverConfig& config) const {
        BestRateParams bp; bp.evaluatorBackend = params_.evaluatorBackend;
        return BestRateSolver(bp).solve(instance, config);
    }
};

}  // namespace dls

#endif  // DLS_HEURISTICS_AUTO_SOLVER_HPP
