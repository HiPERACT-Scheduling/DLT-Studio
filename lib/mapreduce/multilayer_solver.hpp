//---------------------------------------------------------------------------
// mapreduce/multilayer_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Multilayer divisible applications (Berlińska thesis, chapter 5): a generalized
// MapReduce with one mapper layer feeding R >= 1 reducer layers in a pipeline.
// Each reducer layer reads the whole output of the previous layer, sorts and
// reduces it, and forwards γ_p times that to the next layer.
//
// This module provides two things:
//
//  1. xLogXConvexPieces(a, maxX) — the thesis's piecewise-linear convex
//     approximation (§5.2.1) of the reducer cost τ(x) = a·x·log2(x). On each
//     interval [2^y, 2^{y+1}) it returns the secant line a_y·x + b_y with
//     a_y = (τ(2^{y+1}) − τ(2^y)) / (2^{y+1} − 2^y),  b_y = τ(2^y) − a_y·2^y.
//     Because x·log2(x) is convex, these chords are convex, exact at the
//     breakpoints 2^y, and an over-estimate elsewhere (τ' >= τ). The result is a
//     drop-in `std::vector<ComputePiece>` for the piecewise-convex processing
//     feature (see core/dls_instance.hpp), so any DLS solver can model a sorting
//     (x·log x) cost.
//
//  2. MultilayerSolver — a closed-form schedule length for the *homogeneous*
//     pipeline (identical mappers of rate A, and within each layer p identical
//     reducers). With identical reducers the optimal split is equal (δ = 1/r_p),
//     so all reducers in a layer finish together. The schedule is modeled as
//     sequential, non-overlapping phases (a feasible schedule / upper bound on
//     the thesis's overlapped optimum):
//        T = T_map + Σ_{p=1..R} ( read_p + compute_p ),
//        T_map      = m·S + A·(V/m)                       (last mapper finishes),
//        L_p        = V·γ0·Π_{q=1..p-1} γ_q                (input bytes to layer p),
//        read_p     = C·L_p / min(l, r_p)                  (l = bisection width),
//        compute_p  = s_red_p + a_red_p·(L_p/r_p)·log2(L_p/r_p).
//---------------------------------------------------------------------------

#ifndef DLS_MAPREDUCE_MULTILAYER_SOLVER_HPP
#define DLS_MAPREDUCE_MULTILAYER_SOLVER_HPP

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/dls_instance.hpp"    // ComputePiece
#include "core/dls_solution.hpp"    // SolveStatus

namespace dls {

// Goal:   piecewise-linear convex pieces approximating τ(x) = a·x·log2(x) on
//         [1, maxX], with breakpoints at powers of two (Berlińska §5.2.1).
// Input:  a - the reducer rate a_red; maxX - the largest load to cover (>= 2).
// Output: convex ComputePieces {a_y, b_y}; exact at each 2^y, an over-estimate
//         between. (For x < 1 the cost is taken as 0; the pieces start at 2^0=1.)
inline std::vector<ComputePiece> xLogXConvexPieces(double a, double maxX) {
    auto tau = [&](double x) { return (x > 0.0) ? a * x * std::log2(x) : 0.0; };
    std::vector<ComputePiece> pieces;
    const int yMax = std::max(1, static_cast<int>(std::ceil(std::log2(std::max(maxX, 2.0)))));
    for (int y = 0; y < yMax; ++y) {
        const double lo = std::ldexp(1.0, y);       // 2^y
        const double hi = std::ldexp(1.0, y + 1);   // 2^{y+1}
        const double slope     = (tau(hi) - tau(lo)) / (hi - lo);   // a_y (secant)
        const double intercept = tau(lo) - slope * lo;              // b_y
        pieces.push_back({intercept, slope});
    }
    return pieces;
}

// One reducer layer of a multilayer application (identical reducers).
struct ReducerLayer {
    int    count          = 1;     // r_p reducers in this layer
    double startup        = 0.0;   // s_red_p
    double rate           = 0.0;   // a_red_p  (τ_p(x) = rate · x · log2 x)
    double resultFraction = 1.0;   // γ_p: output / input ratio of this layer
};

class MultilayerInstance {
public:
    double totalLoad     = 0.0;    // V
    int    numMappers    = 1;      // m
    double mapperRate    = 0.0;    // A (equal for all mappers)
    double startup       = 0.0;    // S (per-mapper code-load, sequential)
    double readRate      = 0.0;    // C (inter-layer read rate, time/byte)
    double mapperFraction = 1.0;   // γ0: mapper output / input ratio
    int    bisectionWidth = 1;     // l: concurrent communication channels
    std::vector<ReducerLayer> layers;   // R reducer layers (layer 1..R)

    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (totalLoad <= 0.0)  return fail("totalLoad (V) must be positive");
        if (numMappers < 1)    return fail("numMappers (m) must be >= 1");
        if (mapperRate < 0.0 || startup < 0.0 || readRate < 0.0 || mapperFraction < 0.0)
            return fail("a mapper-layer parameter is negative");
        if (bisectionWidth < 1) return fail("bisectionWidth (l) must be >= 1");
        if (layers.empty())    return fail("at least one reducer layer is required");
        for (std::size_t p = 0; p < layers.size(); ++p) {
            const ReducerLayer& L = layers[p];
            if (L.count < 1) return fail("layer " + std::to_string(p) + " has < 1 reducer");
            if (L.startup < 0.0 || L.rate < 0.0 || L.resultFraction < 0.0)
                return fail("layer " + std::to_string(p) + " has a negative parameter");
        }
        if (error) error->clear();
        return true;
    }
};

// Result of scheduling a multilayer application.
struct MultilayerSolution {
    SolveStatus         status   = SolveStatus::NotSolved;
    double              makespan = 0.0;    // total schedule length T
    double              mapperTime = 0.0;  // T_map
    std::vector<double> layerInput;        // L_p, input bytes to each layer
    std::vector<double> layerRead;         // read_p
    std::vector<double> layerCompute;      // compute_p (= s_red_p + τ_p(L_p/r_p))

    bool feasible() const { return status == SolveStatus::Optimal || status == SolveStatus::Feasible; }
};

class MultilayerSolver {
public:
    std::string name() const { return "multilayer"; }

    // Goal:   compute the homogeneous multilayer schedule length (sequential
    //         phases, equal per-layer split).
    // Input:  the instance.
    // Output: a MultilayerSolution; Feasible with the per-layer breakdown, or
    //         Infeasible when the instance is invalid.
    MultilayerSolution solve(const MultilayerInstance& inst) const {
        MultilayerSolution sol;
        std::string error;
        if (!inst.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }

        // Mapper phase: sequential startups + compute of the equal V/m share.
        sol.mapperTime = inst.numMappers * inst.startup
                       + inst.mapperRate * (inst.totalLoad / inst.numMappers);
        double T = sol.mapperTime;

        // Reducer layers in pipeline. Input to layer 1 is the mapper output γ0·V;
        // layer p forwards γ_p times its input to layer p+1.
        double input = inst.mapperFraction * inst.totalLoad;     // L_1
        for (const ReducerLayer& L : inst.layers) {
            const double perReducer = input / L.count;           // equal split share
            const int    channels   = std::min(inst.bisectionWidth, L.count);
            const double read    = inst.readRate * input / channels;
            const double compute = L.startup
                                 + L.rate * perReducer * std::log2(std::max(perReducer, 1.0));
            sol.layerInput.push_back(input);
            sol.layerRead.push_back(read);
            sol.layerCompute.push_back(compute);
            T += read + compute;
            input *= L.resultFraction;                           // L_{p+1}
        }
        sol.makespan = T;
        sol.status   = SolveStatus::Feasible;    // a valid (non-overlapped) schedule
        return sol;
    }
};

}  // namespace dls

#endif  // DLS_MAPREDUCE_MULTILAYER_SOLVER_HPP
