//---------------------------------------------------------------------------
// heuristics/auto/auto_ml_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// AutoMlSolver: a meta-solver that uses an ML-trained policy (GBM embedded as
// plain C++ in solver_selector.hpp) to select the portfolio solver most likely
// to achieve the best makespan for a given instance.
//
// Unlike AutoSolver (which uses hand-coded rules), the policy is learned from
// labelled benchmark instances. The training pipeline is:
//   1. tools/generate_training_data.py — label each instance with the winning solver
//   2. tools/train_solver_selector.py  — train GBM and overwrite solver_selector.hpp
//
// solver_selector.hpp ships as a placeholder (heuristic fallback identical to
// AutoSolver) until the training script overwrites it with the trained model.
//
// It builds concrete solver objects directly (not via the registry) to avoid an
// include cycle — the same pattern used by AutoSolver.
//---------------------------------------------------------------------------

#ifndef DLS_HEURISTICS_AUTO_ML_SOLVER_HPP
#define DLS_HEURISTICS_AUTO_ML_SOLVER_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"
#include "core/instance_features.hpp"
#include "heuristics/auto/solver_selector.hpp"
#include "heuristics/ga/ga_solver.hpp"
#include "heuristics/best_rate/best_rate_solver.hpp"
#include "heuristics/online/online_solver.hpp"
#include "heuristics/single_round/single_round_solver.hpp"
#include "exact/enumerative/exact_solver.hpp"
#include "exact/dual/dual_bisection_solver.hpp"
#ifdef DLS_WITH_HIGHS
#include "exact/milp/milp_solver.hpp"
#include "exact/milp/multi_milp_solver.hpp"
#endif

namespace dls {

struct AutoMlParams {
    int    maxInstallments   = 5;
    std::string evaluatorBackend = "simplex";

    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        if (maxInstallments < 1) { if (error) *error = "maxInstallments must be >= 1"; return false; }
        if (error) error->clear();
        return true;
    }
};

class AutoMlSolver : public DLSSolver {
public:
    explicit AutoMlSolver(AutoMlParams params = {}) : params_(std::move(params)) {}

    std::string    name() const override     { return "auto-ml"; }
    SolverCategory category() const override  { return SolverCategory::Heuristic; }

    // Goal:   the solver this last delegated to (set during solve()).
    const std::string& chosenSolver() const { return chosen_; }

    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override {
        DLSSolution result;
        std::string error;
        if (!instance.validate(&error)) { result.status = SolveStatus::Infeasible; return result; }
        if (!params_.validate(&error))  { result.status = SolveStatus::Failure;    return result; }

        const InstanceFeatures feats = computeFeatures(instance);
        chosen_ = SolverSelector::predict(feats);

        result = dispatch(chosen_, instance, config, feats);

        // Safety net: if the predicted solver fails, retry with best-rate which
        // handles any multi-installment case.
        if (!result.feasible() && chosen_ != "best-rate") {
            DLSSolution alt = bestRate(instance, config);
            if (alt.feasible()) { chosen_ += "→best-rate"; result = alt; }
        }
        return result;
    }

private:
    AutoMlParams params_;
    std::string  chosen_;   // which solver solve() delegated to (mutable run state)

    DLSSolution dispatch(const std::string& solver, const DLSInstance& inst,
                         const SolverConfig& cfg, const InstanceFeatures& feats) const {
        if (solver == "single-round") {
            return SingleRoundSolver().solve(inst, cfg);
        }
        if (solver == "ga") {
            GAParams p;
            p.installments       = std::max(static_cast<int>(feats.N), params_.maxInstallments);
            p.populationSize     = 12;
            p.maxGenerations     = 200;
            p.noImprovementLimit = 50;
            p.evaluatorBackend   = params_.evaluatorBackend;
            return GASolver(p).solve(inst, cfg);
        }
        if (solver == "best-rate") {
            return bestRate(inst, cfg);
        }
        if (solver == "online") {
            return OnlineSolver().solve(inst, cfg);
        }
        if (solver == "exact") {
            ExactParams p;
            p.maxInstallments  = params_.maxInstallments;
            p.allowRepeats     = true;
            p.evaluatorBackend = params_.evaluatorBackend;
            return ExactSolver(p).solve(inst, cfg);
        }
        if (solver == "exact-dual") {
            DualBisectionParams p;
            p.maxInstallments  = params_.maxInstallments;
            p.evaluatorBackend = params_.evaluatorBackend;
            return DualBisectionSolver(p).solve(inst, cfg);
        }
#ifdef DLS_WITH_HIGHS
        if (solver == "exact-milp") {
            MilpParams p; p.maxInstallments = params_.maxInstallments;
            return MilpSolver(p).solve(inst, cfg);
        }
        if (solver == "milp-multi") {
            MultiMilpParams p; p.maxInstallments = params_.maxInstallments;
            return MultiMilpSolver(p).solve(inst, cfg);
        }
#endif
        // Unknown prediction (e.g. label from a future training run on a newer build):
        // fall through to ga as a safe default.
        GAParams gp;
        gp.installments       = std::max(static_cast<int>(feats.N), params_.maxInstallments);
        gp.populationSize     = 12;
        gp.maxGenerations     = 200;
        gp.noImprovementLimit = 50;
        gp.evaluatorBackend   = params_.evaluatorBackend;
        return GASolver(gp).solve(inst, cfg);
    }

    DLSSolution bestRate(const DLSInstance& inst, const SolverConfig& cfg) const {
        BestRateParams bp; bp.evaluatorBackend = params_.evaluatorBackend;
        return BestRateSolver(bp).solve(inst, cfg);
    }
};

}  // namespace dls

#endif  // DLS_HEURISTICS_AUTO_ML_SOLVER_HPP
