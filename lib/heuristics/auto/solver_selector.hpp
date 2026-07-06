//---------------------------------------------------------------------------
// heuristics/auto/solver_selector.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// ML solver selector — PLACEHOLDER version (heuristic fallback).
// Replace with the trained model by running:
//
//   python tools/generate_training_data.py   # ~50 k instances, ~30 min
//   python tools/train_solver_selector.py    # trains GBM, overwrites this file
//
// The generated file has the same interface (SolverSelector::predict) but
// uses an embedded gradient-boosted tree instead of the hand-coded rules below.
//
// This placeholder mirrors the AutoSolver heuristic exactly so that the
// auto-ml solver behaves identically to auto before training is complete.
//---------------------------------------------------------------------------

#ifndef DLS_HEURISTICS_AUTO_SOLVER_SELECTOR_HPP
#define DLS_HEURISTICS_AUTO_SOLVER_SELECTOR_HPP

#include <string>

#include "core/instance_features.hpp"

namespace dls {

// Goal:   predict the portfolio solver most likely to achieve the best
//         makespan for the given instance features (placeholder: hand-coded
//         heuristic identical to AutoSolver; replaced by a trained GBM on
//         running tools/train_solver_selector.py).
// Input:  features - computed by computeFeatures(instance).
// Output: solver name string (matches registry keys).
class SolverSelector {
public:
    static std::string predict(const InstanceFeatures& f) {
        // Small N: exact B&B is tractable and provably optimal.
        if (f.N <= 6.0) return "exact";
        // Memory-limited: single-round cannot carry the full load.
        if (f.memoryRatio < 1.0) return "best-rate";
        // No startups: single-round LP has a closed-form optimum.
        if (f.hasStartups < 0.5) return "single-round";
        // General case: GA handles heterogeneous startups.
        return "ga";
    }
};

}  // namespace dls

#endif  // DLS_HEURISTICS_AUTO_SOLVER_SELECTOR_HPP
