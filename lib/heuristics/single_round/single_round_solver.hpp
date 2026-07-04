//---------------------------------------------------------------------------
// heuristics/single_round/single_round_solver.hpp
//
// SingleRoundSolver: the classical closed-form divisible-load algorithm for a
// single distribution round on a single-port star. Every participating
// processor receives exactly one chunk and they all finish computing at the
// same instant T (the optimal-schedule property for one round); the chunk sizes
// follow an O(m) affine recurrence.
//
// Activation order. Berlińska's thesis (§2.4) recalls the special cases of the
// optimal single-round order: by non-decreasing Cᵢ when all Sᵢ = 0 (and when the
// load V is large). This solver therefore activates processors in non-decreasing
// Cᵢ. For Sᵢ = 0 the result is the exact optimum (and matches the exact B&B);
// with startups it is a fast, high-quality heuristic (exact ordering is hard for
// general Cᵢ>0). Processors whose load would turn negative are dropped (too many
// for the load), mirroring the over-provisioning rule of the other closed forms.
//
// Recurrence (equal completion, single port): a chunk to processor i starts
// being sent at tᵢ, takes Sᵢ + Cᵢαᵢ, then computes Aᵢαᵢ, finishing at
//   tᵢ + Sᵢ + (Cᵢ+Aᵢ)αᵢ = T,  with  tᵢ₊₁ = tᵢ + Sᵢ + Cᵢαᵢ,  t₁ = 0.
// Eliminating tᵢ gives  Aᵢαᵢ = Sᵢ₊₁ + (Cᵢ₊₁+Aᵢ₊₁)αᵢ₊₁, and writing αᵢ = lᵢ + kᵢT:
//   l₁ = −S₁/(C₁+A₁), k₁ = 1/(C₁+A₁);
//   lᵢ₊₁ = (Aᵢlᵢ − Sᵢ₊₁)/(Cᵢ₊₁+Aᵢ₊₁), kᵢ₊₁ = Aᵢkᵢ/(Cᵢ₊₁+Aᵢ₊₁);
//   Σαᵢ = V  ⇒  T = (V − Σlᵢ)/Σkᵢ.
//---------------------------------------------------------------------------

#ifndef DLS_HEURISTICS_SINGLE_ROUND_SOLVER_HPP
#define DLS_HEURISTICS_SINGLE_ROUND_SOLVER_HPP

#include <algorithm>
#include <numeric>
#include <string>
#include <vector>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"

namespace dls {

class SingleRoundSolver : public DLSSolver {
public:
    SingleRoundSolver() = default;

    std::string    name() const override     { return "single-round"; }
    SolverCategory category() const override  { return SolverCategory::Heuristic; }  // exact when Sᵢ=0

    // Goal:   compute the single-round closed-form schedule (all used processors
    //         finish together), activating in non-decreasing Cᵢ order.
    // Input:  instance - the problem; config - runtime (unused; deterministic).
    // Output: a DLSSolution; Feasible with the schedule (one chunk per used
    //         processor, loads summing to V), or Infeasible/Failure.
    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override {
        (void)config;
        DLSSolution sol;
        std::string error;
        if (!instance.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }

        const auto& procs = instance.processors();
        const double V = instance.totalLoad();

        // Activation order: non-decreasing Cᵢ (Berlińska §2.4 special case).
        std::vector<int> order(procs.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return procs[a].commRate < procs[b].commRate; });

        // Use the fastest-communicating prefix; drop the last while a load < 0.
        std::vector<double> alpha;
        double T = 0.0;
        int used = static_cast<int>(order.size());
        for (; used >= 1; --used) {
            if (solveRound(procs, order, used, V, alpha, T)) break;   // all αᵢ >= 0
        }
        if (used < 1) { sol.status = SolveStatus::Failure; return sol; }

        // Build the schedule: cumulative send times, each used processor finishes at T.
        double t = 0.0;
        for (int k = 0; k < used; ++k) {
            const Processor& p = procs[order[k]];
            LoadFragment f;
            f.processorId   = order[k];
            f.loadSize      = alpha[k];
            f.commStart     = t;
            f.commFinish    = t + p.commStartup + p.commRate * alpha[k];
            f.computeStart  = f.commFinish;
            f.computeFinish = f.commFinish + p.computeRate * alpha[k];   // == T
            sol.fragments.push_back(f);
            sol.sequence.push_back(order[k]);
            t = f.commFinish;                                            // single port: next send after this
        }
        sol.status   = SolveStatus::Feasible;
        sol.makespan = T;
        return sol;
    }

private:
    // Goal:   solve the affine recurrence for the first `used` processors in
    //         `order`; report the loads and completion time T.
    // Input:  procs, order, used, V; alpha/T out-params.
    // Output: true with alpha (size used) and T set if every αᵢ >= 0, else false
    //         (signal to drop the slowest-communicating processor).
    static bool solveRound(const std::vector<Processor>& procs, const std::vector<int>& order,
                           int used, double V, std::vector<double>& alpha, double& T) {
        std::vector<double> l(used), k(used);              // αᵢ = lᵢ + kᵢ·T
        const Processor& p0 = procs[order[0]];
        const double d0 = p0.commRate + p0.computeRate;
        l[0] = -p0.commStartup / d0;
        k[0] = 1.0 / d0;
        for (int i = 1; i < used; ++i) {
            const Processor& p = procs[order[i]];
            const double d = p.commRate + p.computeRate;   // Cᵢ + Aᵢ (> 0 by hasCost/validate)
            const double aPrev = procs[order[i - 1]].computeRate;
            l[i] = (aPrev * l[i - 1] - p.commStartup) / d;
            k[i] = (aPrev * k[i - 1]) / d;
        }
        double sumL = 0.0, sumK = 0.0;
        for (int i = 0; i < used; ++i) { sumL += l[i]; sumK += k[i]; }
        if (sumK <= 0.0) return false;
        T = (V - sumL) / sumK;
        alpha.assign(used, 0.0);
        for (int i = 0; i < used; ++i) {
            alpha[i] = l[i] + k[i] * T;
            if (alpha[i] < 0.0) return false;              // too many processors
        }
        return true;
    }
};

}  // namespace dls

#endif  // DLS_HEURISTICS_SINGLE_ROUND_SOLVER_HPP
