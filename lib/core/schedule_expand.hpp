//---------------------------------------------------------------------------
// core/schedule_expand.hpp
//
// Fill in the full per-installment timing of a solved schedule so it can be
// drawn (e.g. a Gantt chart) or inspected by a front-end.
//
// Most solvers populate only what their method needs: the LP evaluator and the
// MILP fill each fragment's communication start (and load), but leave the
// communication-finish and the compute interval at 0; the online heuristic fills
// all four. expandSchedule() reconstructs the missing fields uniformly from the
// instance coefficients and the (already chosen) commStart + load, using the same
// timing the makespan model assumes:
//
//   commFinish    = commStart + S + C·α                 (single-port transfer)
//   computeStart  = max(commFinish, releaseTime)        (compute begins on receipt)
//   computeFinish = computeStart + τ(α + y)             (τ = convex piecewise time)
//
// This matches the makespan model's per-installment finish term, so for the
// single-installment / no-result-return case the reconstructed makespan equals
// the solver's reported makespan exactly (asserted in the tests). With result
// return (β > 0) the reported makespan additionally covers the collection phase,
// which this compute-only reconstruction does not draw.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_SCHEDULE_EXPAND_HPP
#define DLS_CORE_SCHEDULE_EXPAND_HPP

#include <algorithm>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/energy_model.hpp"   // processingTime (convex piecewise τ)

namespace dls {

// Goal:   fill commFinish / computeStart / computeFinish on every fragment.
// Input:  instance - the problem (timing coefficients); sol - a solved schedule
//         whose fragments carry processorId, loadSize, carriedLoad and commStart.
// Output: the reconstructed makespan (max computeFinish); sol.fragments are
//         updated in place. Fragments with an invalid processorId are skipped.
inline double expandSchedule(const DLSInstance& instance, DLSSolution& sol) {
    const int P = static_cast<int>(instance.numProcessors());
    double makespan = 0.0;
    for (LoadFragment& f : sol.fragments) {
        if (f.processorId < 0 || f.processorId >= P) continue;
        const Processor& p = instance.processors()[f.processorId];
        f.commFinish    = f.commStart + p.commStartup + p.commRate * f.loadSize;
        f.computeStart  = std::max(f.commFinish, p.releaseTime);
        f.computeFinish = f.computeStart + processingTime(p, f.loadSize + f.carriedLoad);
        makespan = std::max(makespan, f.computeFinish);
    }
    return makespan;
}

}  // namespace dls

#endif  // DLS_CORE_SCHEDULE_EXPAND_HPP
