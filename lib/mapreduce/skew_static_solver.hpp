//---------------------------------------------------------------------------
// mapreduce/skew_static_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Static partitioning-skew mitigation ("fine partitioning", Berlińska &
// Drozdowski MISTA 2013, §3.1). Instead of the plain r-way key partition,
// the space of intermediate keys is divided into k*r parts (k>1); after the
// mappers finish, each reports its part sizes to the master, which solves the
// resulting NP-hard bin-packing assignment with an LPT-style heuristic (sort
// parts descending, greedily assign each to the currently least-loaded
// reducer — the paper's own kr·log(kr) + kr·log(r) master-cost formula is
// exactly this sort-then-heap-assign algorithm's complexity, confirming the
// intended algorithm), then sends the assignment back before reducers start
// reading.
//
// Dependency-free (a pure greedy heuristic, no LP/HiGHS needed) — the paper
// explicitly frames both its algorithms as "easy to implement in practice".
//---------------------------------------------------------------------------

#ifndef DLS_MAPREDUCE_SKEW_STATIC_SOLVER_HPP
#define DLS_MAPREDUCE_SKEW_STATIC_SOLVER_HPP

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "core/dls_solution.hpp"
#include "mapreduce/mapreduce_skew_instance.hpp"

namespace dls {

struct StaticSkewSolution {
    SolveStatus status   = SolveStatus::NotSolved;
    double      makespan = 0.0;
    std::vector<double>          reducerLoads;   // per reducer j: its assigned total (bytes)
    std::vector<std::vector<int>> assignment;    // assignment[j] = original partition indices given to reducer j
    double masterTime = 0.0;   // t^master, the LPT assignment's own compute cost

    bool feasible() const { return status == SolveStatus::Optimal || status == SolveStatus::Feasible; }
};

class StaticSkewSolver {
public:
    std::string name() const { return "skew-static"; }

    // Goal:   schedule length under fine-partitioning + LPT reassignment.
    // Input:  instance - k*r partition sizes, k>=1 (k=1 just re-derives the
    //         plain unmitigated schedule, since LPT over r items onto r
    //         identical reducers cannot improve on the given sizes).
    // Output: a StaticSkewSolution; Feasible (LPT is a heuristic for an
    //         NP-hard assignment, not a proven optimum) or Infeasible if the
    //         instance is invalid.
    StaticSkewSolution solve(const MapReduceSkewInstance& instance) const {
        StaticSkewSolution sol;
        std::string error;
        if (!instance.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }

        const int m = instance.numMappers, r = instance.numReducers, k = instance.k;
        const int kr = k * r;
        const double C = instance.readRate, eps = instance.epsilon;
        const int minMR = std::min(m, r);
        const double perUnitRead = std::max(C, C * minMR / instance.bisectionWidth) / minMR;

        // Mapper phase (unaffected by skew mitigation) + control-message
        // rounds (mappers -> master reports kr sizes; master -> reducers
        // sends the assignment), both at kr granularity per Section 3.1.
        const double tMap     = instance.mapperRate * instance.totalLoad / m;
        const double tReport  = C * kr * m * eps;
        const double tMaster  = instance.masterRate *
            (kr * m + kr * std::log2(std::max(1.0, static_cast<double>(kr)))
                    + kr * std::log2(std::max(1.0, static_cast<double>(r))));
        const double tAssign  = C * kr * m * eps;

        // LPT: sort the kr parts descending, greedily hand each to whichever
        // reducer currently has the least assigned load (min-heap).
        std::vector<int> order(kr);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            return instance.partitionSizes[a] > instance.partitionSizes[b];
        });
        std::vector<double> load(r, 0.0);
        std::vector<std::vector<int>> assign(r);
        using PQItem = std::pair<double, int>;   // (currentLoad, reducerIdx)
        std::priority_queue<PQItem, std::vector<PQItem>, std::greater<PQItem>> pq;
        for (int j = 0; j < r; ++j) pq.push({0.0, j});
        for (int idx : order) {
            auto [ld, j] = pq.top(); pq.pop();
            load[j] += instance.partitionSizes[idx];
            assign[j].push_back(idx);
            pq.push({load[j], j});
        }

        double makespan = 0.0;
        for (int j = 0; j < r; ++j) {
            const double p = load[j];
            const double read_j   = perUnitRead * p;
            const double sort_j   = instance.sortRate * std::max(0.0, p * std::log2(std::max(p, 1.0)));
            const double reduce_j = instance.reduceRate * p;
            const double finish   = tMap + tReport + tMaster + tAssign + read_j + sort_j + reduce_j;
            makespan = std::max(makespan, finish);
        }

        sol.reducerLoads = std::move(load);
        sol.assignment    = std::move(assign);
        sol.masterTime    = tMaster;
        sol.makespan      = makespan;
        sol.status        = SolveStatus::Feasible;
        return sol;
    }
};

}  // namespace dls

#endif  // DLS_MAPREDUCE_SKEW_STATIC_SOLVER_HPP
