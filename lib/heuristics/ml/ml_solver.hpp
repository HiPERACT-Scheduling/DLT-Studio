//---------------------------------------------------------------------------
// heuristics/ml/ml_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// MlSolver: ML-based makespan predictor + schedule constructor for single-load
// DLS instances. Registered in the portfolio as "ml-makespan".
//
// Design: two-stage solve.
//   Stage 1 — sub-microsecond GBM inference predicts log(T*) → T_ml.
//   Stage 2 — best applicable heuristic constructs a valid schedule:
//     • Sᵢ=0, Cᵢ=0, no memory cap → BestRateSolver (closed-form, optimal here)
//     • Sᵢ>0, Cᵢ=0, no memory cap → FptasOptTSolver (ε=0.05 guaranteed approx)
//     • general (Cᵢ>0 or memory cap) → GASolver (genetic search + LP per candidate;
//                                       searches activation orders across all
//                                       processors, unlike BestRate's greedy pass)
//
// The predicted makespan T_ml is stored and reported separately from the actual
// schedule makespan, enabling fair benchmark comparison:
//   ml_predicted vs. heuristic_actual vs. exact_optimal
//
// The model ships as a placeholder (rough closed-form) and is overwritten by
// tools/train_makespan_predictor.py after the training pipeline runs.
//---------------------------------------------------------------------------

#ifndef DLS_HEURISTICS_ML_ML_SOLVER_HPP
#define DLS_HEURISTICS_ML_ML_SOLVER_HPP

#include <cmath>
#include <string>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"
#include "core/instance_features.hpp"
#include "heuristics/best_rate/best_rate_solver.hpp"
#include "heuristics/fptas/fptas_optt_solver.hpp"
#include "heuristics/ga/ga_solver.hpp"
#include "heuristics/ml/makespan_predictor.hpp"

namespace dls {

struct MlSolverParams {
    std::string evaluatorBackend = "simplex";
    int         maxInstallments  = 5;   // passed to GASolver for the general case
};

class MlSolver : public DLSSolver {
public:
    explicit MlSolver(MlSolverParams params = {}) : params_(std::move(params)) {}

    std::string    name() const override     { return "ml-makespan"; }
    SolverCategory category() const override  { return SolverCategory::Heuristic; }

    // Goal:   the makespan predicted by the ML model before scheduling.
    //         Compare against solution().makespan for prediction accuracy.
    double predictedMakespan() const { return predictedMakespan_; }

    // Goal:   predict T* then construct a valid schedule via the best
    //         applicable heuristic for the instance class.
    // Input:  instance - any valid single-load DLSInstance.
    // Output: a DLSSolution from the Stage 2 heuristic.
    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override {
        DLSSolution result;
        std::string err;
        if (!instance.validate(&err)) { result.status = SolveStatus::Infeasible; return result; }

        // Stage 1: ML prediction (sub-microsecond).
        const InstanceFeatures f = computeFeatures(instance);
        predictedMakespan_ = std::exp(MakespanPredictor::predict(f));

        // Stage 2: schedule construction — dispatch by instance class.
        if (f.hasCommCost < 0.5 && f.memoryRatio >= 1.0) {
            if (f.hasStartups > 0.5) {
                // Startup-only, infinite bandwidth: FPTAS OptT (ε=0.05 guarantee).
                FptasOptTParams p; p.epsilon = 0.05;
                result = FptasOptTSolver(p).solve(instance, config);
                if (!result.feasible()) result = ga(instance, config);
            } else {
                // Pure DLT (no S, no C, no memory cap): BestRate is LP-optimal here.
                result = bestRate(instance, config);
            }
        } else {
            // General case (comm costs and/or memory caps): BestRate is a single
            // greedy pass and often concentrates load on one processor. GASolver
            // searches activation orders and LP-optimises each split — much better
            // at distributing load across heterogeneous processors.
            result = ga(instance, config);
        }
        return result;
    }

private:
    MlSolverParams params_;
    double         predictedMakespan_ = 0.0;

    DLSSolution bestRate(const DLSInstance& inst, const SolverConfig& cfg) const {
        BestRateParams p; p.evaluatorBackend = params_.evaluatorBackend;
        return BestRateSolver(p).solve(inst, cfg);
    }

    DLSSolution ga(const DLSInstance& inst, const SolverConfig& cfg) const {
        GAParams p;
        p.installments     = params_.maxInstallments;
        p.evaluatorBackend = params_.evaluatorBackend;
        return GASolver(p).solve(inst, cfg);
    }
};

}  // namespace dls

#endif  // DLS_HEURISTICS_ML_ML_SOLVER_HPP
