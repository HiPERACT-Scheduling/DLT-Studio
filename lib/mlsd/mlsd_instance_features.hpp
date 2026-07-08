//---------------------------------------------------------------------------
// mlsd/mlsd_instance_features.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Fixed-size feature vector extracted from an MlsdInstance in O(N + n_tasks).
// Mirrors the single-load InstanceFeatures pattern for the MLSD problem class.
//
// Used by the ML Cmax predictor (heuristics/ml/mlsd_predictor.hpp) and
// exposed as a utility for benchmarking and analysis.
//---------------------------------------------------------------------------

#ifndef DLS_MLSD_INSTANCE_FEATURES_HPP
#define DLS_MLSD_INSTANCE_FEATURES_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "mlsd/mlsd_instance.hpp"

namespace dls {

// Feature vector for an MlsdInstance. Never construct manually.
struct MlsdInstanceFeatures {
    // --- task distribution ---
    double nTasks;         // number of tasks n
    double nProcs;         // number of processors N
    double meanV;          // mean task size V̄
    double cvV;            // CV of task sizes std(Vⱼ) / mean(Vⱼ)
    double maxMinV;        // maxVⱼ / minVⱼ  — task imbalance
    double totalLoadPerProc; // ΣVⱼ / N  — workload density
    // --- memory ---
    double memoryRatio;    // ΣBᵢ / ΣVⱼ  — < 1 means multi-installment required
    // --- processor parameters ---
    double meanA;          // mean compute rate
    double heteroA;        // CV of Aᵢ
    double speedupA;       // maxAᵢ / minAᵢ
    double hasStartups;    // 1 if any Sᵢ > 0
    double hasCommCost;    // 1 if any Cᵢ > 0

    static constexpr int SIZE = 12;

    std::array<double, SIZE> toArray() const {
        return {nTasks, nProcs, meanV, cvV, maxMinV, totalLoadPerProc,
                memoryRatio, meanA, heteroA, speedupA, hasStartups, hasCommCost};
    }
};

// Goal:   extract the feature vector from an MlsdInstance in O(N + n_tasks).
// Input:  inst - any valid MLSD instance.
// Output: filled MlsdInstanceFeatures; all ratios clamped to [0, 1e6].
inline MlsdInstanceFeatures computeMlsdFeatures(const MlsdInstance& inst) {
    const auto& tasks = inst.tasks();
    const auto& procs = inst.processors();
    const int   n     = static_cast<int>(tasks.size());
    const int   m     = static_cast<int>(procs.size());

    // Task statistics
    double sumV = 0, sumV2 = 0, minV = std::numeric_limits<double>::infinity(), maxV = 0;
    for (const MlsdTask& t : tasks) {
        sumV  += t.size;
        sumV2 += t.size * t.size;
        minV   = std::min(minV, t.size);
        maxV   = std::max(maxV, t.size);
    }
    const double mV    = (n > 0) ? sumV / n : 0.0;
    const double varV  = (n > 0) ? std::max(0.0, sumV2 / n - mV * mV) : 0.0;
    const double cvV   = (mV > 1e-12) ? std::sqrt(varV) / mV : 0.0;
    const double mmV   = (minV > 1e-12) ? maxV / minV : 1.0;

    // Processor statistics
    double sumA = 0, sumA2 = 0, sumB = 0;
    double minA = std::numeric_limits<double>::infinity(), maxA = 0;
    bool anyS = false, anyC = false;
    for (const Processor& p : procs) {
        sumA  += p.computeRate;  sumA2 += p.computeRate * p.computeRate;
        minA   = std::min(minA, p.computeRate);
        maxA   = std::max(maxA, p.computeRate);
        if (p.commStartup > 0.0) anyS = true;
        if (p.commRate    > 0.0) anyC = true;
        sumB  += (p.memoryLimit > 0.0 && p.memoryLimit < 1e17) ? p.memoryLimit : 1e18;
    }
    const double invM = (m > 0) ? 1.0 / m : 1.0;
    const double mA   = sumA * invM;
    const double varA = (m > 0) ? std::max(0.0, sumA2 * invM - mA * mA) : 0.0;
    const double cvA  = (mA > 1e-12) ? std::sqrt(varA) / mA : 0.0;
    const double spA  = (minA > 1e-12) ? maxA / minA : 1.0;
    const double memR = (sumV > 1e-12) ? std::min(sumB / sumV, 1e6) : 1e6;

    return MlsdInstanceFeatures{
        .nTasks          = static_cast<double>(n),
        .nProcs          = static_cast<double>(m),
        .meanV           = mV,
        .cvV             = cvV,
        .maxMinV         = mmV,
        .totalLoadPerProc = (m > 0) ? sumV / m : 0.0,
        .memoryRatio     = memR,
        .meanA           = mA,
        .heteroA         = cvA,
        .speedupA        = spA,
        .hasStartups     = anyS ? 1.0 : 0.0,
        .hasCommCost     = anyC ? 1.0 : 0.0,
    };
}

}  // namespace dls

#endif  // DLS_MLSD_INSTANCE_FEATURES_HPP
