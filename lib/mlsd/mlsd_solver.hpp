//---------------------------------------------------------------------------
// mlsd/mlsd_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Exact MLSD solver: enumerate permutation-schedule structures (task execution
// order x per-task ordered processor subsets), evaluate each with the MLSD LP,
// and keep the minimum makespan. MLSD is NP-hard, so this is for small
// instances; it is the reference oracle the polynomial special cases validate
// against.
//---------------------------------------------------------------------------

#ifndef DLS_MLSD_SOLVER_HPP
#define DLS_MLSD_SOLVER_HPP

#include <algorithm>
#include <functional>
#include <limits>
#include <numeric>
#include <vector>

#include "mlsd/mlsd_evaluator.hpp"
#include "mlsd/mlsd_instance.hpp"

namespace dls {

// Goal:   all non-empty ordered subsets (activation sequences) of {0..m-1}.
// Input:  m - processor count.
// Output: every sequence a task could use as its processor activation order.
inline std::vector<std::vector<int>> orderedProcessorSequences(int m) {
    std::vector<std::vector<int>> out;
    for (int mask = 1; mask < (1 << m); ++mask) {
        std::vector<int> elems;
        for (int i = 0; i < m; ++i) if (mask & (1 << i)) elems.push_back(i);
        std::sort(elems.begin(), elems.end());
        do { out.push_back(elems); } while (std::next_permutation(elems.begin(), elems.end()));
    }
    return out;
}

class MlsdSolver {
public:
    // Goal:   best schedule for a FIXED task order (searches per-task processor
    //         sequences only).
    // Input:  inst, taskOrder.
    // Output: the optimal MlsdSolution for that order (Infeasible if none).
    MlsdSolution solveFixedOrder(const MlsdInstance& inst,
                                 const std::vector<int>& taskOrder) const {
        const int n = static_cast<int>(taskOrder.size());
        const int m = static_cast<int>(inst.numProcessors());
        const std::vector<std::vector<int>> seqs = orderedProcessorSequences(m);

        MlsdScheduleEvaluator ev;
        double best = std::numeric_limits<double>::infinity();
        MlsdSolution bestSol;                         // Infeasible until something is found
        MlsdStructure st; st.taskOrder = taskOrder; st.procSeq.resize(n);

        std::function<void(int)> assign = [&](int l) {
            if (l == n) {
                MlsdSolution s = ev.evaluate(inst, st);
                if (s.status == SolveStatus::Optimal && s.makespan < best) {
                    best = s.makespan; bestSol = s; bestSol.taskOrder = taskOrder;
                }
                return;
            }
            for (const std::vector<int>& seq : seqs) { st.procSeq[l] = seq; assign(l + 1); }
        };
        assign(0);
        return bestSol;
    }

    // Goal:   globally optimal schedule (searches all task orders).
    // Input:  inst.
    // Output: the optimal MlsdSolution (Optimal), or Infeasible/Failure.
    MlsdSolution solve(const MlsdInstance& inst) const {
        MlsdSolution result;
        std::string error;
        if (!inst.validate(&error)) { result.status = SolveStatus::Infeasible; return result; }

        const int n = static_cast<int>(inst.numTasks());
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);     // start sorted (deterministic)

        double best = std::numeric_limits<double>::infinity();
        do {
            MlsdSolution s = solveFixedOrder(inst, order);
            if (s.status == SolveStatus::Optimal && s.makespan < best) {
                best = s.makespan; result = s;
            }
        } while (std::next_permutation(order.begin(), order.end()));

        if (result.status != SolveStatus::Optimal) result.status = SolveStatus::Infeasible;
        return result;
    }
};

}  // namespace dls

#endif  // DLS_MLSD_SOLVER_HPP
