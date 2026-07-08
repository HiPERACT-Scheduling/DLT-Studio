//---------------------------------------------------------------------------
// heuristics/ml/ml_mlsd_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// MlMlsdSolver: ML-based Cmax predictor + schedule constructor for MLSD
// instances. Registered in the portfolio as "ml-mlsd".
//
// Design: two-stage solve.
//   Stage 1 — sub-microsecond GBM inference predicts log(Cmax*) → Cmax_ml.
//   Stage 2 — MlsdGaSolver constructs a valid schedule, whose actual makespan
//              is reported separately from the ML prediction.
//
// The predicted Cmax is stored and exposed via predictedMakespan(), enabling
// fair benchmark comparison:
//   ml_predicted vs. GA_actual vs. exact_optimal
//
// The model ships as a placeholder (rough closed-form) and is overwritten by
// tools/train_mlsd_predictor.py after the training pipeline runs.
//---------------------------------------------------------------------------

#ifndef DLS_HEURISTICS_ML_ML_MLSD_SOLVER_HPP
#define DLS_HEURISTICS_ML_ML_MLSD_SOLVER_HPP

#include <cmath>

#include "mlsd/mlsd_ga_solver.hpp"
#include "mlsd/mlsd_instance.hpp"
#include "mlsd/mlsd_instance_features.hpp"
#include "heuristics/ml/mlsd_predictor.hpp"

namespace dls {

class MlMlsdSolver {
public:
    // Goal:   the Cmax predicted by the ML model before scheduling.
    //         Compare against solution().makespan for prediction accuracy.
    double predictedMakespan() const { return predictedMakespan_; }

    // Goal:   predict Cmax* then construct a valid schedule via MlsdGaSolver.
    // Input:  inst - any valid MlsdInstance.
    // Output: an MlsdSolution from the GA heuristic.
    MlsdSolution solve(const MlsdInstance& inst) {
        // Stage 1: ML prediction (sub-microsecond).
        predictedMakespan_ = std::exp(MlsdPredictor::predict(computeMlsdFeatures(inst)));

        // Stage 2: schedule construction via GA heuristic.
        MlsdGaSolver ga{MlsdGaSolver::Params{}};
        return ga.solve(inst);
    }

private:
    double predictedMakespan_ = 0.0;
};

}  // namespace dls

#endif  // DLS_HEURISTICS_ML_ML_MLSD_SOLVER_HPP
