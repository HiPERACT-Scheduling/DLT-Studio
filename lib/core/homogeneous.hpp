//---------------------------------------------------------------------------
// core/homogeneous.hpp
//
// Closed-form analysis of a HOMOGENEOUS single-port star: m identical
// processors, each with the same startup S, communication rate C, computation
// rate A, distributing a single load V in one round (all participating
// processors finish at the same instant T — the optimal single-round structure).
//
// For k identical processors the equal-completion load recurrence solves in
// closed form. With q = A/(C+A) (so 0 < q < 1) and the geometric sum
// G(k) = Σ_{i=0}^{k-1} qⁱ = (1 − qᵏ)/(1 − q):
//
//   α₁ = (V + k·S/C)/G(k) − S/C,     T = S + (C+A)·α₁,
//   αᵢ = −S/C + q^{i−1}·(α₁ + S/C)   (decreasing; αₖ is the smallest).
//
// Using more processors lowers T until the startup chain drives the last
// processor's load negative; the optimal count k* is the largest k ≤ m with all
// αᵢ ≥ 0. (Sending a processor several chunks does NOT help under the library's
// no-overlap carried-load model — each repeat just adds a startup S — so for a
// homogeneous system the only lever is how many processors to use.)
//
// The C = 0 case (infinite bandwidth) is the q → 1 limit, handled separately.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_HOMOGENEOUS_HPP
#define DLS_CORE_HOMOGENEOUS_HPP

#include <cmath>

namespace dls {

// Result of the homogeneous single-round closed form for a fixed processor count.
struct HomogeneousRound {
    double makespan = 0.0;   // T (all used processors finish together)
    double minLoad  = 0.0;   // αₖ, the smallest per-processor load (< 0 ⇒ k too large)
};

// Goal:   closed-form makespan and smallest load for k identical processors in
//         one round (equal completion).
// Input:  S,C,A - identical processor parameters (A > 0, C+A > 0); V - load;
//         k - number of processors used (>= 1).
// Output: { makespan T, minLoad αₖ }. αₖ < 0 means k is too large to be feasible.
inline HomogeneousRound homogeneousSingleRound(double S, double C, double A, double V, int k) {
    HomogeneousRound r;
    if (k < 1) return r;
    const double d = C + A;                       // C + A
    double alpha1, alphaK;
    if (C > 0.0) {
        const double q  = A / d;                  // 0 < q < 1
        const double sc = S / C;                  // the fixed point is −S/C
        const double G  = (std::abs(1.0 - q) < 1e-15) ? k : (1.0 - std::pow(q, k)) / (1.0 - q);
        alpha1 = (V + k * sc) / G - sc;
        alphaK = -sc + std::pow(q, k - 1) * (alpha1 + sc);
    } else {
        // C = 0 (q → 1): αᵢ = α₁ − (i−1)·S/A, Σαᵢ = k·α₁ − (S/A)·k(k−1)/2 = V.
        const double sa = (A > 0.0) ? S / A : 0.0;
        alpha1 = (V + sa * (k * (k - 1) / 2.0)) / k;
        alphaK = alpha1 - (k - 1) * sa;
    }
    r.makespan = S + d * alpha1;
    r.minLoad  = alphaK;
    return r;
}

// Optimal homogeneous single-round outcome.
struct HomogeneousOptimum {
    int    kStar    = 0;     // best number of processors to use (0 ⇒ none feasible)
    double makespan = 0.0;   // its makespan T
};

// Goal:   choose how many of m identical processors to use to minimize the
//         single-round makespan (the diminishing-returns count k*).
// Input:  S,C,A - identical parameters; V - load; m - processors available.
// Output: { k*, makespan }. Scans k = 1..m using the O(1) closed form and keeps
//         the smallest feasible (all αᵢ ≥ 0) makespan.
inline HomogeneousOptimum homogeneousOptimalProcessors(double S, double C, double A, double V, int m) {
    HomogeneousOptimum best;
    double bestT = 0.0;
    for (int k = 1; k <= m; ++k) {
        const HomogeneousRound r = homogeneousSingleRound(S, C, A, V, k);
        if (r.minLoad < -1e-9) break;            // αₖ went negative ⇒ k (and larger) infeasible
        if (best.kStar == 0 || r.makespan < bestT - 1e-12) { best.kStar = k; bestT = r.makespan; }
    }
    best.makespan = bestT;
    return best;
}

}  // namespace dls

#endif  // DLS_CORE_HOMOGENEOUS_HPP
