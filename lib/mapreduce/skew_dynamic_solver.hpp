//---------------------------------------------------------------------------
// mapreduce/skew_dynamic_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Dynamic partitioning-skew mitigation (a simplified SkewTune, Berlińska &
// Drozdowski MISTA 2013, §3.2). Unlike the static method, the plain r-way
// partition is left alone; mapping, reading and sorting proceed exactly as
// in the unmitigated schedule (Section 2 of the paper). Once EVERY reducer
// has finished sorting AND at least one has also finished reducing, the
// master halts the r1 still-busy reducers, learns their remaining (linear,
// already-sorted) load, and rebalances it onto the already-finished
// ("idle") reducers before resuming — a single rebalancing event, not
// SkewTune's continual one.
//
// The paper describes the rebalancing decision only narratively ("the
// master checks the scenario where the most loaded reducer sends part of
// its data to the least loaded one... continues to more processors as long
// as profitable") without a closed form beyond the 2-node case. This solver
// derives the natural n-sender/n-receiver generalization via the same
// equal-finish-time DLT principle the paper invokes: for n_s busy senders
// (only their remaining-data SUM S matters) each free to keep some load
// locally (rate a^red) or hand it to one of n_r idle receivers (rate
// C + a^red, transfer then reduce), the equalized finish time is
//     τ(n_s, n_r, S) = S / (n_s/a^red + n_r/(C + a^red))
// (set n_s=n_r=1 and this is exactly the paper's 2-node case). Using every
// available idle reducer as a receiver is always weakly better (more
// receiver capacity strictly lowers τ), so the only real choice is how many
// of the busiest senders to fold in — a 1-D search over n_s=0..r1 that this
// solver evaluates exhaustively (cheap, O(r1)) rather than the paper's own
// greedy incremental search, so it is guaranteed to find the true best
// n_s under this model rather than possibly stopping early.
//
// Requires k=1 (the base r-way partition) — repartitioning is the static
// method's job; this one only ever moves already-computed remainders.
// Dependency-free (closed form + a small search), no LP/HiGHS needed.
//---------------------------------------------------------------------------

#ifndef DLS_MAPREDUCE_SKEW_DYNAMIC_SOLVER_HPP
#define DLS_MAPREDUCE_SKEW_DYNAMIC_SOLVER_HPP

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "core/dls_solution.hpp"
#include "mapreduce/mapreduce_skew_instance.hpp"

namespace dls {

struct DynamicSkewSolution {
    SolveStatus status   = SolveStatus::NotSolved;
    double      makespan = 0.0;
    double      unmitigatedMakespan = 0.0;   // same instance, no rebalancing at all
    double      triggerTime         = 0.0;   // when the master intervenes
    int         numBusyAtTrigger    = 0;     // r1
    int         sendersUsed         = 0;     // best n_s found (0 = rebalancing didn't help)

    bool feasible() const { return status == SolveStatus::Optimal || status == SolveStatus::Feasible; }
};

class DynamicSkewSolver {
public:
    std::string name() const { return "skew-dynamic"; }

    // Goal:   schedule length under single-shot dynamic rebalancing.
    // Input:  instance - a k=1 (plain r-way) partition-skew instance.
    // Output: a DynamicSkewSolution; Feasible, or Infeasible if the instance
    //         is invalid or k != 1.
    DynamicSkewSolution solve(const MapReduceSkewInstance& instance) const {
        DynamicSkewSolution sol;
        std::string error;
        if (!instance.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }
        if (instance.k != 1) { sol.status = SolveStatus::Infeasible; return sol; }

        const int m = instance.numMappers, r = instance.numReducers;
        const double C = instance.readRate;
        const int minMR = std::min(m, r);
        const double perUnitRead = std::max(C, C * minMR / instance.bisectionWidth) / minMR;
        const double tMap = instance.mapperRate * instance.totalLoad / m;
        const double tLoc = 2.0 * C * m * r * instance.epsilon;

        // Unmitigated schedule (Section 2): this IS the plain baseline, and
        // doubles as the phase-1 computation the rebalancing decision needs.
        std::vector<double> sortFinish(r), reduceTime(r), rawFinish(r);
        for (int j = 0; j < r; ++j) {
            const double p = instance.partitionSizes[j];
            const double read_j = perUnitRead * p;
            const double sort_j = instance.sortRate * std::max(0.0, p * std::log2(std::max(p, 1.0)));
            sortFinish[j] = tMap + tLoc + read_j + sort_j;
            reduceTime[j] = instance.reduceRate * p;
            rawFinish[j]  = sortFinish[j] + reduceTime[j];
        }
        sol.unmitigatedMakespan = *std::max_element(rawFinish.begin(), rawFinish.end());

        // Trigger: all reducers done sorting, AND at least one fully done.
        const double tAllSortDone   = *std::max_element(sortFinish.begin(), sortFinish.end());
        const double tFirstDoneFull = *std::min_element(rawFinish.begin(), rawFinish.end());
        const double tTrigger = std::max(tAllSortDone, tFirstDoneFull);
        sol.triggerTime = tTrigger;

        std::vector<double> busyRemaining;   // remaining DATA (not time) for still-busy reducers
        int nIdle = 0;
        for (int j = 0; j < r; ++j) {
            const double remTime = std::max(0.0, reduceTime[j] - (tTrigger - sortFinish[j]));
            if (remTime > 1e-12) busyRemaining.push_back(remTime / instance.reduceRate);
            else ++nIdle;
        }
        const int r1 = static_cast<int>(busyRemaining.size());
        sol.numBusyAtTrigger = r1;

        if (r1 == 0) {
            // Every reducer reached this instant simultaneously — nothing to
            // rebalance, no halt overhead incurred.
            sol.makespan   = tTrigger;
            sol.sendersUsed = 0;
            sol.status = SolveStatus::Feasible;
            return sol;
        }

        // Halt the r1 busy reducers, learn their remaining load, and (once
        // decided) notify them to resume — three C·r1·ε control rounds. The
        // paper gives explicit formulas for the first two; the third
        // ("notifies the reducers to resume... and informs them about the
        // load parts") is described only narratively, so this solver
        // extends the same established per-round cost to it.
        const double tAfterHalt = tTrigger + 3.0 * C * r1 * instance.epsilon;

        std::sort(busyRemaining.begin(), busyRemaining.end(), std::greater<double>());
        double best = instance.reduceRate * busyRemaining[0];   // n_s = 0: do nothing
        int bestNs = 0;
        double sumS = 0.0;
        for (int ns = 1; ns <= r1; ++ns) {
            sumS += busyRemaining[ns - 1];
            const double denom = ns / instance.reduceRate + static_cast<double>(nIdle) / (C + instance.reduceRate);
            const double tau = sumS / denom;
            const double leftover = (ns < r1) ? instance.reduceRate * busyRemaining[ns] : 0.0;
            const double candidate = std::max(tau, leftover);
            if (candidate < best) { best = candidate; bestNs = ns; }
        }

        sol.sendersUsed = bestNs;
        sol.makespan     = tAfterHalt + best;
        sol.status       = SolveStatus::Feasible;
        return sol;
    }
};

}  // namespace dls

#endif  // DLS_MAPREDUCE_SKEW_DYNAMIC_SOLVER_HPP
