//---------------------------------------------------------------------------
// heuristics/energy_features.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Feature vector for the ML minimum-energy predictor (heuristics/ml/
// ml_energy_solver.hpp), mirroring tools/generate_energy_training_data.py's
// compute_energy_features() exactly (same 15 features, same order, same CV
// formula) so the trained model sees the same inputs at inference time it saw
// during training.
//---------------------------------------------------------------------------

#ifndef DLS_HEURISTICS_ENERGY_FEATURES_HPP
#define DLS_HEURISTICS_ENERGY_FEATURES_HPP

#include <cmath>
#include <vector>

#include "core/dls_instance.hpp"

namespace dls {

struct EnergyFeatures {
    float N;
    float loadPerProc;
    float meanA;
    float heteroA;
    float meanC;
    float heteroC;
    float hasStartups;
    float meanPowerIdle;
    float meanPowerStartup;
    float meanPowerNetwork;
    float heteroPowerNetwork;
    float meanEnergySlope;
    float heteroEnergySlope;
    float originatorPowerNetwork;
    float originatorPowerIdle;
};

// Goal:   population coefficient of variation (std / mean), 0 when the mean is
//         ~0 — matches tools/generate_energy_training_data.py's _cv() exactly.
// Input:  a non-empty vector of values.
// Output: CV as a float.
inline float energyCv(const std::vector<double>& vals) {
    double mean = 0.0;
    for (double v : vals) mean += v;
    mean /= static_cast<double>(vals.size());
    if (mean < 1e-12) return 0.0f;
    double var = 0.0;
    for (double v : vals) var += (v - mean) * (v - mean);
    var /= static_cast<double>(vals.size());
    return static_cast<float>(std::sqrt(std::max(var, 0.0)) / mean);
}

// Goal:   extract the 15-feature vector an energy-model single-load instance
//         presents to the ML minimum-energy predictor.
// Input:  a validated DLSInstance (single-installment, energy model present).
// Output: EnergyFeatures, in the exact field order predict_log_energy expects.
inline EnergyFeatures computeEnergyFeatures(const DLSInstance& inst) {
    const auto& procs = inst.processors();
    const double N = static_cast<double>(procs.size());

    std::vector<double> A, C, pI, pS, pN, k;
    A.reserve(procs.size()); C.reserve(procs.size());
    pI.reserve(procs.size()); pS.reserve(procs.size());
    pN.reserve(procs.size()); k.reserve(procs.size());
    bool hasStartups = false;
    for (const auto& p : procs) {
        A.push_back(p.computeRate);
        C.push_back(p.commRate);
        pI.push_back(p.powerIdle);
        pS.push_back(p.powerStartup);
        pN.push_back(p.powerNetwork);
        k.push_back(p.effectiveEnergyPieces().front().slope);
        if (p.commStartup > 0.0) hasStartups = true;
    }
    const auto mean = [](const std::vector<double>& v) {
        double s = 0.0; for (double x : v) s += x; return s / static_cast<double>(v.size());
    };

    EnergyFeatures f;
    f.N                      = static_cast<float>(N);
    f.loadPerProc            = static_cast<float>(inst.totalLoad() / N);
    f.meanA                  = static_cast<float>(mean(A));
    f.heteroA                = energyCv(A);
    f.meanC                  = static_cast<float>(mean(C));
    f.heteroC                = energyCv(C);
    f.hasStartups            = hasStartups ? 1.0f : 0.0f;
    f.meanPowerIdle          = static_cast<float>(mean(pI));
    f.meanPowerStartup       = static_cast<float>(mean(pS));
    f.meanPowerNetwork       = static_cast<float>(mean(pN));
    f.heteroPowerNetwork     = energyCv(pN);
    f.meanEnergySlope        = static_cast<float>(mean(k));
    f.heteroEnergySlope      = energyCv(k);
    f.originatorPowerNetwork = static_cast<float>(inst.originatorPowerNetwork());
    f.originatorPowerIdle    = static_cast<float>(inst.originatorPowerIdle());
    return f;
}

}  // namespace dls

#endif  // DLS_HEURISTICS_ENERGY_FEATURES_HPP
