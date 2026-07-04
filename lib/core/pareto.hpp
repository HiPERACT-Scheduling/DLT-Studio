//---------------------------------------------------------------------------
// core/pareto.hpp
//
// Time–energy trade-off front (Marszałkowski 2020): for a fixed activation
// order, how low can the energy go if the makespan is capped at T̄? Sweeping the
// cap from the makespan-optimal point up to the energy-optimal point traces the
// Pareto frontier the GUI's "time–energy explorer" plots.
//
// It reuses the per-sequence evaluator's bicriteria modes directly (no LP code
// here): MinMakespan gives the fast/high-energy end, MinCost gives the
// slow/low-energy end, and MinCost subject to makespanLimit = T̄ gives every
// interior point. Energy is reported on every solution by the evaluator
// (scheduleEnergy), so a MinMakespan solve already carries its energy.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_PARETO_HPP
#define DLS_CORE_PARETO_HPP

#include <algorithm>
#include <vector>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/schedule_evaluator.hpp"

namespace dls {

// One point on the time–energy front.
struct ParetoPoint {
    double makespan = 0.0;
    double energy   = 0.0;
};

// Goal:   compute the time–energy Pareto front for a fixed activation sequence.
// Input:  instance - the problem (an energy model makes the front non-trivial);
//         ev - a schedule evaluator; sequence - the activation order to hold
//         fixed; nPoints - number of makespan caps to sample (>=2); base -
//         evaluator knobs (iteration cap, seed) reused for every solve.
// Output: points sorted by increasing makespan, energy non-increasing — from the
//         makespan-optimal schedule (fast, energy-costly) to the energy-optimal
//         one (slow, energy-cheap). Empty if the sequence is infeasible.
inline std::vector<ParetoPoint> timeEnergyFront(const DLSInstance&      instance,
                                                const ScheduleEvaluator& ev,
                                                const std::vector<int>&  sequence,
                                                int                      nPoints,
                                                EvaluatorConfig          base = {}) {
    std::vector<ParetoPoint> front;

    EvaluatorConfig tcfg = base; tcfg.objective = EvalObjective::MinMakespan;
    DLSSolution sT = ev.evaluate(instance, sequence, tcfg);
    EvaluatorConfig ecfg = base; ecfg.objective = EvalObjective::MinCost;
    DLSSolution sE = ev.evaluate(instance, sequence, ecfg);
    if (!sT.feasible() || !sE.feasible()) return front;

    const double tLo = sT.makespan;             // smallest achievable makespan
    const double tHi = sE.makespan;             // makespan at the energy optimum (>= tLo)
    const int n = std::max(2, nPoints);
    const double eps = 1e-9;

    // The makespan-optimal schedule is always the first (highest-energy) point.
    front.push_back({sT.makespan, sT.energy});
    if (tHi - tLo > eps) {
        // Interior + endpoint caps: min energy subject to Cmax <= T̄.
        for (int i = 1; i < n; ++i) {
            const double cap = tLo + (tHi - tLo) * i / (n - 1);
            EvaluatorConfig c = base;
            c.objective     = EvalObjective::MinCost;
            c.makespanLimit = cap;
            DLSSolution s = ev.evaluate(instance, sequence, c);
            if (s.feasible()) front.push_back({s.makespan, s.energy});
        }
    }

    // Sort by makespan and drop points that don't improve energy (keep the
    // lower envelope: as the cap grows, energy must strictly fall to matter).
    std::sort(front.begin(), front.end(),
              [](const ParetoPoint& a, const ParetoPoint& b) { return a.makespan < b.makespan; });
    std::vector<ParetoPoint> out;
    for (const ParetoPoint& p : front) {
        if (out.empty() || p.energy < out.back().energy - eps ||
            p.makespan > out.back().makespan + eps)
            out.push_back(p);
    }
    return out;
}

}  // namespace dls

#endif  // DLS_CORE_PARETO_HPP
