//---------------------------------------------------------------------------
// mapreduce/mapreduce_solver.hpp
//
// Exact closed-form scheduler for MapReduce divisible loads (Berlińska thesis,
// section 4.4.1 single reducer / 4.4.2 "first method" for many reducers).
//
// Optimal schedule structure (Proposition 4.1): activate mappers in order of
// non-decreasing rate Aᵢ, read mapper outputs sequentially in FIFO order. The
// completion of computing-plus-reading on Pi coincides with startup-plus-
// computing on Pi₊₁, giving the linear system
//
//     (Aᵢ + γ₀C/r)·αᵢ = S + Aᵢ₊₁·αᵢ₊₁     (i = 1..m−1)
//     Σ αᵢ = V
//
// solved in O(m) by the affine reduction αᵢ = lᵢ + kᵢ·αₘ (eqs. 4.46–4.48). If
// αₘ < 0 the system is using too many processors, so the slowest mapper is
// dropped and the system re-solved. The schedule length is
//
//     T = mS + αₘ(Aₘ + γ₀C/r) + t_red ,   t_red = s_red + a_red·(z·log₂z),
//
// with z = γ₀V/r. The "many reducers, first method" case is exactly the single-
// reducer algorithm with γ₀C replaced by γ₀C/r.
//---------------------------------------------------------------------------

#ifndef DLS_MAPREDUCE_SOLVER_HPP
#define DLS_MAPREDUCE_SOLVER_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "core/dls_solution.hpp"   // SolveStatus (shared status enum)
#include "mapreduce/mapreduce_instance.hpp"

namespace dls {

// Result of scheduling a MapReduce instance.
struct MapReduceSolution {
    SolveStatus         status   = SolveStatus::NotSolved;
    double              makespan = 0.0;   // total schedule length T
    std::vector<double> mapperLoads;      // αᵢ for the USED mappers, in activation order
    std::vector<int>    mapperOrder;      // original indices of the used mappers (increasing Aᵢ)
    double              reducerTime = 0.0;// t_red, the (equal) reducer execution time

    bool feasible() const {
        return status == SolveStatus::Optimal || status == SolveStatus::Feasible;
    }
    double totalAssignedLoad() const {
        double s = 0.0; for (double a : mapperLoads) s += a; return s;
    }
};

class MapReduceSolver {
public:
    // Goal:   stable identifier (registry / reporting).
    std::string name() const { return "mapreduce"; }

    // Goal:   compute the optimal MapReduce schedule.
    // Input:  instance - the problem (mapper rates, V, global params).
    // Output: a MapReduceSolution; Optimal with loads/order/makespan, or
    //         Infeasible/Failure when the instance is invalid.
    MapReduceSolution solve(const MapReduceInstance& instance) const {
        MapReduceSolution sol;
        std::string error;
        if (!instance.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }

        const double V = instance.totalLoad();
        const double g = instance.resultFraction() * instance.readRate()
                         / static_cast<double>(instance.numReducers());   // γ₀C/r
        const double S = instance.startup();

        // Mapper indices sorted by non-decreasing rate Aᵢ (Proposition 4.1).
        std::vector<int> order(instance.numMappers());
        std::iota(order.begin(), order.end(), 0);
        const auto& A = instance.mapperRates();
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return A[a] < A[b]; });

        // Use the fastest m' mappers; drop the slowest while αₘ < 0.
        std::vector<double> rates;                 // sorted rates of the used mappers
        rates.reserve(order.size());
        for (int idx : order) rates.push_back(A[idx]);

        std::vector<double> alpha;
        int used = static_cast<int>(rates.size());
        while (used >= 1) {
            std::vector<double> r(rates.begin(), rates.begin() + used);
            if (solvePartition(r, V, g, S, alpha)) break;   // all αᵢ >= 0
            --used;                                          // drop the slowest mapper
        }
        if (used < 1) { sol.status = SolveStatus::Failure; return sol; }  // should not happen

        // Reducer execution time t_red = s_red + a_red·(z·log₂z), z = γ₀V/r.
        const double z = instance.resultFraction() * V / static_cast<double>(instance.numReducers());
        const double tRed = instance.reducerStartup()
                            + instance.reducerRate() * std::max(0.0, z * std::log2(std::max(z, 1.0)));

        sol.status      = SolveStatus::Optimal;
        sol.reducerTime = tRed;
        sol.mapperLoads = alpha;
        sol.mapperOrder.assign(order.begin(), order.begin() + used);
        // T = mS + αₘ(Aₘ + g) + t_red.
        sol.makespan = used * S + alpha.back() * (rates[used - 1] + g) + tRed;
        return sol;
    }

private:
    // Goal:   solve the O(m) load-partitioning system for the m fastest mappers.
    // Input:  A - sorted (increasing) rates of the m used mappers; V - load;
    //         g - γ₀C/r; S - startup; alpha - out-param for the loads.
    // Output: true with alpha filled if every αᵢ >= 0; false if αₘ < 0 (signal
    //         to drop the slowest mapper).
    static bool solvePartition(const std::vector<double>& A, double V, double g,
                               double S, std::vector<double>& alpha) {
        const int m = static_cast<int>(A.size());
        std::vector<double> l(m), k(m);            // αᵢ = lᵢ + kᵢ·αₘ
        l[m - 1] = 0.0; k[m - 1] = 1.0;
        for (int i = m - 2; i >= 0; --i) {
            const double d = A[i] + g;             // Aᵢ + γ₀C/r  (> 0 since Aᵢ > 0)
            l[i] = S / d + A[i + 1] * l[i + 1] / d;
            k[i] = A[i + 1] * k[i + 1] / d;
        }
        double sumL = 0.0, sumK = 0.0;
        for (int i = 0; i < m; ++i) { sumL += l[i]; sumK += k[i]; }
        const double am = (V - sumL) / sumK;       // eq. 4.48
        if (am < 0.0) return false;                // too many processors
        alpha.assign(m, 0.0);
        alpha[m - 1] = am;
        for (int i = m - 2; i >= 0; --i) alpha[i] = l[i] + k[i] * am;
        return true;
    }
};

}  // namespace dls

#endif  // DLS_MAPREDUCE_SOLVER_HPP
