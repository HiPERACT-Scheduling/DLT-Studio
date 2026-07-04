//---------------------------------------------------------------------------
// core/dls_solution.hpp
//
// Uniform result type returned by every DLS solver and by the schedule
// evaluator. It describes *what* schedule was found, not *how*.
//
// A solution is an ordered list of load fragments. Each fragment is one
// installment: a processor receives `loadSize` units of work, with the
// communication and computation timing the evaluator computed for it.
// The makespan is the time the last fragment finishes computing.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_DLS_SOLUTION_HPP
#define DLS_CORE_DLS_SOLUTION_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dls {

enum class SolveStatus {
    NotSolved,    // solver has not run / produced nothing
    Optimal,      // proven optimal (exact solvers, or LP for a fixed sequence)
    Feasible,     // a valid schedule, optimality not proven (heuristics)
    Infeasible,   // no schedule satisfies the constraints
    Unbounded,    // model is unbounded (indicates a modelling error)
    Failure       // numerical failure / iteration limit hit
};

// One installment of the schedule.
struct LoadFragment {
    int    processorId   = -1;   // index into DLSInstance::processors()
    double loadSize      = 0.0;  // units of load assigned to this installment
    double carriedLoad   = 0.0;  // y: load carried in from this proc's previous
                                 // installment (0 on first appearance / single-round)

    // Timing, as filled in by the evaluator (0 if not computed).
    double commStart     = 0.0;  // master begins sending this fragment
    double commFinish    = 0.0;  // fragment fully received by the worker
    double computeStart  = 0.0;  // worker begins computing
    double computeFinish = 0.0;  // worker finishes (contributes to makespan)
};

class DLSSolution {
public:
    SolveStatus               status   = SolveStatus::NotSolved;
    double                    makespan = 0.0;   // Cmax
    double                    cost     = 0.0;   // G = Σ(fᵢ + αᵢ lᵢ) over used processors
    double                    energy   = 0.0;   // E: total four-state energy (0 if no energy model)
    std::vector<int>          sequence;         // processor activation order
    std::vector<LoadFragment> fragments;        // schedule, in dispatch order

    // Optional run metadata, filled in by solvers that track it.
    long          iterations = 0;     // generations / simplex iters / nodes
    double        wallTimeSec = 0.0;  // measured solve time
    std::uint64_t usedSeed   = 0;     // RNG seed actually used (for reproducibility)

    bool feasible() const {
        return status == SolveStatus::Optimal || status == SolveStatus::Feasible;
    }

    double totalAssignedLoad() const {
        double sum = 0.0;
        for (const LoadFragment& f : fragments) sum += f.loadSize;
        return sum;
    }

    // Sanity check that the schedule distributes the expected total load.
    bool conservesLoad(double totalLoad, double tol = 1e-4) const {
        return std::fabs(totalAssignedLoad() - totalLoad) <= tol;
    }
};

}  // namespace dls

#endif  // DLS_CORE_DLS_SOLUTION_HPP
