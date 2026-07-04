//---------------------------------------------------------------------------
// core/schedule_evaluator.hpp
//
// Shared machinery: given a DLS instance and a FIXED activation sequence,
// compute the optimal load fragmentation and the resulting makespan.
//
// This is the piece currently buried inside the GA's EvaluatePopulation():
// for a fixed order of installments the problem of choosing fragment sizes
// is a linear program, and Cmax is its optimum. Lifting it out behind this
// interface is what lets exact and heuristic solvers share one contract.
//
//   - A heuristic (e.g. the GA) searches over sequences and calls
//     evaluate() to score each candidate.
//   - An exact branch-and-bound enumerates sequences and calls the same
//     evaluate() at the leaves.
//   - A monolithic exact MILP solver may bypass this and model order as
//     decision variables; it simply does not use a ScheduleEvaluator.
//
// The interface is backend-agnostic: the legacy CSimplex becomes one
// concrete implementation (e.g. SimplexScheduleEvaluator); a third-party LP
// library could be dropped in without touching any solver.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_SCHEDULE_EVALUATOR_HPP
#define DLS_CORE_SCHEDULE_EVALUATOR_HPP

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "dls_instance.hpp"
#include "dls_solution.hpp"

namespace dls {

// Knobs for the underlying LP solve. These are *solver/runtime* parameters
// (shared infrastructure), kept separate from both the problem and any
// search-algorithm parameters.
// Backend-neutral runtime knobs for a ScheduleEvaluator. Each backend
// interprets what it can and ignores the rest.
// Which quantity the evaluator minimizes for a fixed sequence.
enum class EvalObjective {
    MinMakespan,   // minimize Cmax (optionally subject to cost G <= costLimit)
    MinCost,       // minimize cost G  (optionally subject to Cmax <= makespanLimit)
    MaxLoad        // maximize the total load Σαᵢ subject to Cmax <= makespanLimit
                   // (the OptV dual: how much load fits within a deadline T).
                   // Requires a finite makespanLimit; the instance's totalLoad is
                   // ignored (load conservation is dropped and Σαᵢ is maximized).
};

struct EvaluatorConfig {
    int           maxIterations = 1000;   // LP iteration cap (legacy iSolMaxIter)
    double        epsilon       = 1e-5;   // numerical tolerance
    std::uint64_t seed          = 0;      // anti-cycling RNG seed; used by backends
                                          // that need it (e.g. CSimplex), ignored by others
    EvalObjective objective     = EvalObjective::MinMakespan;  // which criterion to minimize
    double        costLimit =              // G-bar: cost limit (MinMakespan mode)
        std::numeric_limits<double>::infinity();  // (inf = none)
    double        makespanLimit =          // C-bar: makespan limit (MinCost mode)
        std::numeric_limits<double>::infinity();  // (inf = none)
};

class ScheduleEvaluator {
public:
    virtual ~ScheduleEvaluator() = default;

    // Compute the best load distribution for `sequence`, an ordered list of
    // processor indices into instance.processors(). The returned solution
    // carries the same `sequence`, the per-installment fragments with sizes
    // and timing, the makespan, and a status:
    //   Optimal    - LP solved; makespan and fragments are filled in
    //   Infeasible - no feasible distribution for this sequence
    //   Unbounded  - modelling error
    //   Failure    - iteration limit / numerical failure
    virtual DLSSolution evaluate(const DLSInstance&       instance,
                                 const std::vector<int>&  sequence,
                                 const EvaluatorConfig&   config) const = 0;

    // Identifies the backend (e.g. "simplex"), for logging/reporting.
    virtual std::string name() const = 0;
};

}  // namespace dls

#endif  // DLS_CORE_SCHEDULE_EVALUATOR_HPP
