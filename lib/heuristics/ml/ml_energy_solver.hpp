//---------------------------------------------------------------------------
// heuristics/ml/ml_energy_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// EnergyMlSolver: ML-based minimum-energy predictor + schedule constructor for
// single-load, single-installment (beta=0) DLS instances with an energy model.
// Registered in the portfolio as "ml-energy".
//
// Unlike ml-makespan, this is not a "closed form vs. slow oracle" gap: the
// ONLY existing way to minimize energy is exact-milp (minimizeCost=1), and its
// cost grows steeply with N (calibrated: N=7 ~3s, N=8 ~5s, N=9 already >20s;
// see tools/generate_energy_training_data.py) — there is no fast energy-aware
// heuristic oracle to fall back on for larger instances.
//
// Design: two-stage solve.
//   Stage 1 — GBM inference predicts log(minimum energy) in sub-microsecond
//             time (heuristics/ml/energy_predictor.hpp, test MAPE 16.8%).
//   Stage 2 — feed the prediction (with headroom for that error margin) as a
//             cost-limit hint to GASolver's existing bi-criteria mode
//             ("minimize Cmax subject to cost <= costLimit"), so the search
//             is steered toward the ML-estimated energy budget without paying
//             for exact-milp. If the tight budget is infeasible (the estimate
//             undershot), retry unconstrained rather than fail outright.
//
// predictedEnergy() is the Stage 1 estimate, reported alongside the Stage 2
// schedule's actual energy for comparison — the same honesty pattern as
// MlSolver::predictedMakespan(). The two are not expected to match exactly:
// Stage 2 still ultimately minimizes makespan (no fast energy-minimizing
// heuristic exists), so its actual energy is a bi-criteria compromise, not a
// re-derivation of the Stage 1 estimate.
//---------------------------------------------------------------------------

#ifndef DLS_HEURISTICS_ML_ML_ENERGY_SOLVER_HPP
#define DLS_HEURISTICS_ML_ML_ENERGY_SOLVER_HPP

#include <cmath>
#include <string>
#include <vector>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"
#include "heuristics/energy_features.hpp"
#include "heuristics/ga/ga_solver.hpp"
#include "heuristics/ml/energy_predictor.hpp"

namespace dls {

struct EnergyMlSolverParams {
    std::string evaluatorBackend = "simplex";
    int         maxInstallments  = 5;
    double      headroom         = 1.30;   // safety margin over the predicted
                                            // energy (test MAPE 16.8%) before
                                            // handing it to GA as a cost limit
};

class EnergyMlSolver : public DLSSolver {
public:
    explicit EnergyMlSolver(EnergyMlSolverParams params = {}) : params_(std::move(params)) {}

    std::string    name() const override     { return "ml-energy"; }
    SolverCategory category() const override  { return SolverCategory::Heuristic; }

    // Goal:   the minimum achievable energy predicted by the ML model, before
    //         any schedule is constructed. Compare against solution().energy.
    double predictedEnergy() const { return predictedEnergy_; }

    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override {
        DLSSolution result;
        std::string err;
        if (!instance.validate(&err)) { result.status = SolveStatus::Infeasible; return result; }

        // Stage 1: ML prediction (sub-microsecond).
        const EnergyFeatures f = computeEnergyFeatures(instance);
        float raw[15] = {
            f.N, f.loadPerProc, f.meanA, f.heteroA, f.meanC, f.heteroC,
            f.hasStartups, f.meanPowerIdle, f.meanPowerStartup, f.meanPowerNetwork,
            f.heteroPowerNetwork, f.meanEnergySlope, f.heteroEnergySlope,
            f.originatorPowerNetwork, f.originatorPowerIdle,
        };
        predictedEnergy_ = std::exp(predict_log_energy(raw));

        // Stage 2: steer GA toward that budget; fall back if it underestimated.
        GAParams p;
        p.installments     = params_.maxInstallments;
        p.evaluatorBackend = params_.evaluatorBackend;
        p.costLimit        = predictedEnergy_ * params_.headroom;
        result = GASolver(p).solve(instance, config);
        if (!result.feasible()) {
            GAParams unconstrained;
            unconstrained.installments     = params_.maxInstallments;
            unconstrained.evaluatorBackend = params_.evaluatorBackend;
            result = GASolver(unconstrained).solve(instance, config);
        }
        return result;
    }

private:
    EnergyMlSolverParams params_;
    double                predictedEnergy_ = 0.0;
};

}  // namespace dls

#endif  // DLS_HEURISTICS_ML_ML_ENERGY_SOLVER_HPP
