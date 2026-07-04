//---------------------------------------------------------------------------
// core/bounds.hpp
//
// Cheap, closed-form bounds on the optimum makespan of a DLS instance. These
// are O(N) to compute (no LP, no search) and are useful as a quality reference
// for heuristics and as a warm bound for the exact solvers.
//
// The lower bound below is the one Berlińska uses throughout her thesis
// (eq. 3.23) as the reference for measuring heuristic quality:
//
//   τ₁  = n_min·S_min + V·C_min          (fastest possible communication of V)
//   LB  = τ₁ + max{0, V/(Σ 1/Aᵢ) − τ₁ + S_min}
//
// where n_min = ⌈V / B_max⌉ is the least number of messages able to carry V,
// and Σ 1/Aᵢ is the aggregate computing speed. It assumes (optimistically, and
// hence validly for a lower bound) that the smallest startup coincides with the
// largest buffer and that communication and computation overlap maximally.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_BOUNDS_HPP
#define DLS_CORE_BOUNDS_HPP

#include <algorithm>
#include <cmath>
#include <limits>

#include "dls_instance.hpp"

namespace dls {

// Goal:   the per-processor compute rate used by the bounds — the *fastest*
//         (smallest) piece slope, which keeps the bound a valid underestimate.
// Input:  a processor.
// Output: its smallest effective piece slope (= computeRate for the default).
inline double boundComputeRate(const Processor& p) {
    double a = std::numeric_limits<double>::infinity();
    for (const ComputePiece& pc : p.effectivePieces()) a = std::min(a, pc.slope);
    return a;
}

// Goal:   compute Berlińska's lower bound (eq. 3.23) on the optimum makespan.
// Input:  a (validated) DLS instance.
// Output: a value LB with LB ≤ Cmax* for every feasible schedule. Returns 0 for
//         a degenerate instance (no load / no processors).
inline double divisibleLoadLowerBound(const DLSInstance& instance) {
    const auto& procs = instance.processors();
    if (procs.empty()) return 0.0;
    const double V = instance.totalLoad();
    if (V <= 0.0) return 0.0;

    double sMin = std::numeric_limits<double>::infinity();   // smallest startup
    double cMin = std::numeric_limits<double>::infinity();   // smallest comm rate
    double bMax = 0.0;                                        // largest buffer
    double sumInvA = 0.0;                                     // aggregate compute speed
    bool   anyInstantCompute = false;                        // some Aᵢ == 0
    for (const Processor& p : procs) {
        sMin = std::min(sMin, p.commStartup);
        cMin = std::min(cMin, p.commRate);
        bMax = std::max(bMax, p.memoryLimit);
        const double A = boundComputeRate(p);
        if (A <= 0.0) anyInstantCompute = true; else sumInvA += 1.0 / A;
    }

    // Least number of messages that can carry V (1 if memory is unbounded).
    const double nMin = (bMax > 0.0) ? std::ceil(V / bMax) : 1.0;
    const double tau1 = nMin * sMin + V * cMin;              // fastest communication
    // Minimum time to compute V on the aggregate speed (0 if any Aᵢ == 0).
    const double fluid = (anyInstantCompute || sumInvA <= 0.0) ? 0.0 : V / sumInvA;

    return tau1 + std::max(0.0, fluid - tau1 + sMin);
}

// Goal:   the fluid (no-communication, full-parallel) lower bound V / Σ(1/Aᵢ).
// Input:  a DLS instance.
// Output: the bound, or 0 if any processor computes instantly (Aᵢ == 0).
inline double fluidLowerBound(const DLSInstance& instance) {
    double sumInvA = 0.0;
    for (const Processor& p : instance.processors()) {
        const double A = boundComputeRate(p);
        if (A <= 0.0) return 0.0;
        sumInvA += 1.0 / A;
    }
    return (sumInvA > 0.0) ? instance.totalLoad() / sumInvA : 0.0;
}

// Goal:   a lower bound that drops the single-port serialization: every
//         processor may receive from time 0 with no contention, so by time T it
//         can process at most (T − Sᵢ)/(Cᵢ + Aᵢ) load. The smallest T whose
//         aggregate capacity reaches V is a valid lower bound (relaxing the port
//         constraint only lowers the optimum). Respects each Sᵢ, Cᵢ, Aᵢ, so it
//         is usually far tighter than eq. 3.23 on heterogeneous instances.
// Input:  a DLS instance.
// Output: the relaxed lower bound (0 if no processor has a positive Cᵢ + Aᵢ).
inline double portRelaxedLowerBound(const DLSInstance& instance) {
    const double V = instance.totalLoad();
    if (V <= 0.0) return 0.0;

    // Per-processor (startup Sᵢ, slope dᵢ = Cᵢ + fastest Aᵢ). Only dᵢ > 0 count.
    double hi = std::numeric_limits<double>::infinity();    // an upper bound on T_opt
    bool any = false;
    for (const Processor& p : instance.processors()) {
        const double d = p.commRate + boundComputeRate(p);
        if (d > 0.0) { any = true; hi = std::min(hi, p.commStartup + d * V); }  // one proc does all V
    }
    if (!any) return 0.0;

    // Max load processable by time T with no port contention (monotone in T).
    auto capacity = [&](double T) {
        double load = 0.0;
        for (const Processor& p : instance.processors()) {
            const double d = p.commRate + boundComputeRate(p);
            if (d > 0.0 && T > p.commStartup) load += (T - p.commStartup) / d;
        }
        return load;
    };

    double lo = 0.0;
    for (int it = 0; it < 100; ++it) {                      // bisect for the smallest feasible T
        const double mid = 0.5 * (lo + hi);
        if (capacity(mid) >= V) hi = mid; else lo = mid;
    }
    return hi;
}

// Goal:   the tightest of the available cheap valid lower bounds.
// Input:  a DLS instance.
// Output: max(eq. 3.23, fluid, port-relaxed) — still O(N) and still a valid
//         lower bound on the optimum makespan.
inline double divisibleLoadLowerBoundTight(const DLSInstance& instance) {
    return std::max({ divisibleLoadLowerBound(instance),
                      fluidLowerBound(instance),
                      portRelaxedLowerBound(instance) });
}

}  // namespace dls

#endif  // DLS_CORE_BOUNDS_HPP
