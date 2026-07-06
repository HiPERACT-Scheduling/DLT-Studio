//---------------------------------------------------------------------------
// core/instance_features.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Compact fixed-size feature vector extracted from a DLSInstance in O(N).
// Used by the ML-assisted solver selector (heuristics/auto/solver_selector.hpp)
// and exposed as a library utility for benchmarking and analysis.
//
// All 10 features are dimensionless ratios or counts so they are stable across
// different load scales and time units.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_INSTANCE_FEATURES_HPP
#define DLS_CORE_INSTANCE_FEATURES_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "core/dls_instance.hpp"

namespace dls {

// Feature vector for a single-load DLSInstance.
// All values are computed by computeFeatures() below; never construct manually.
struct InstanceFeatures {
    double N;              // number of processors (raw count)
    double memoryRatio;    // min(ΣBᵢ, 1e18) / V  — < 1 means multi-installment required
    double hasStartups;    // 1 if any Sᵢ > 0, else 0
    double hasCommCost;    // 1 if any Cᵢ > 0, else 0
    double heteroA;        // CV of Aᵢ = std(Aᵢ) / mean(Aᵢ)  — compute heterogeneity
    double heteroC;        // CV of Cᵢ (0 when all Cᵢ = 0)
    double heteroS;        // CV of Sᵢ (0 when all Sᵢ = 0)
    double startupFraction;// N·mean(Sᵢ) / (N·mean(Sᵢ) + V·mean(Aᵢ) + 1e-12)
    double hasBeta;        // 1 if β > 0, else 0
    double hasCost;        // 1 if any fᵢ > 0 or lᵢ > 0, else 0

    static constexpr int SIZE = 10;

    // Return features as a plain array in canonical order (matches training CSV columns).
    std::array<double, SIZE> toArray() const {
        return {N, memoryRatio, hasStartups, hasCommCost,
                heteroA, heteroC, heteroS, startupFraction,
                hasBeta, hasCost};
    }
};

// Goal:   extract the feature vector from a DLSInstance in O(N).
// Input:  inst - any valid single-load instance.
// Output: filled InstanceFeatures; all ratios are clamped to [0, 1e6].
inline InstanceFeatures computeFeatures(const DLSInstance& inst) {
    const auto& procs = inst.processors();
    const int   n     = static_cast<int>(procs.size());
    const double V    = inst.totalLoad();

    double sumA = 0, sumC = 0, sumS = 0, sumB = 0;
    double sumA2 = 0, sumC2 = 0, sumS2 = 0;
    bool anyS = false, anyC = false, anyB_finite = false;
    bool anyF = false;

    for (const Processor& p : procs) {
        sumA  += p.computeRate;   sumA2 += p.computeRate  * p.computeRate;
        sumC  += p.commRate;      sumC2 += p.commRate     * p.commRate;
        sumS  += p.commStartup;   sumS2 += p.commStartup  * p.commStartup;
        if (p.commStartup > 0.0) anyS = true;
        if (p.commRate    > 0.0) anyC = true;
        if (p.fixedCost   > 0.0 || p.linearCost > 0.0) anyF = true;
        if (p.memoryLimit > 0.0 && p.memoryLimit < 1e17) { sumB += p.memoryLimit; anyB_finite = true; }
        else sumB += 1e18;
    }

    const double invN = (n > 0) ? 1.0 / n : 1.0;
    const double mA   = sumA * invN;
    const double mC   = sumC * invN;
    const double mS   = sumS * invN;

    auto cv = [&](double sum, double sum2, double mean) -> double {
        if (mean < 1e-12) return 0.0;
        const double var = std::max(0.0, sum2 * invN - mean * mean);
        return std::sqrt(var) / mean;
    };

    const double memRatio   = (V > 1e-12) ? std::min(sumB / V, 1e6) : 1e6;
    const double startupFrac = (n * mS + V * mA + 1e-12 > 0)
                              ? (n * mS) / (n * mS + V * mA + 1e-12) : 0.0;

    return InstanceFeatures{
        .N               = static_cast<double>(n),
        .memoryRatio     = memRatio,
        .hasStartups     = anyS  ? 1.0 : 0.0,
        .hasCommCost     = anyC  ? 1.0 : 0.0,
        .heteroA         = cv(sumA, sumA2, mA),
        .heteroC         = cv(sumC, sumC2, mC),
        .heteroS         = cv(sumS, sumS2, mS),
        .startupFraction = startupFrac,
        .hasBeta         = (inst.resultFraction() > 0.0) ? 1.0 : 0.0,
        .hasCost         = anyF ? 1.0 : 0.0,
    };
}

}  // namespace dls

#endif  // DLS_CORE_INSTANCE_FEATURES_HPP
