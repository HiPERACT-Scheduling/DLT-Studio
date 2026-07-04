//---------------------------------------------------------------------------
// mlsd/mlsd_instance.hpp
//
// Types for the MLSD problem (Multiple Loads, Single Distribution): n divisible
// tasks of sizes V_j scheduled on m processors over a single-port star, in a
// permutation schedule. Mirrors the single-load library's split: a fixed
// structure (task order + per-task processor sequences) reduces to an LP for
// the load fractions; a solver searches structures on top.
//
// This first version is the "related" case: a processor's parameters (S, C, A)
// are the same for every task (reusing dls::Processor). Per-task ("unrelated")
// parameters are a later generalization.
//---------------------------------------------------------------------------

#ifndef DLS_MLSD_INSTANCE_HPP
#define DLS_MLSD_INSTANCE_HPP

#include <string>
#include <vector>

#include "core/dls_instance.hpp"   // dls::Processor
#include "core/dls_solution.hpp"   // dls::SolveStatus

namespace dls {

// One divisible task.
struct MlsdTask {
    double size           = 0.0;   // V_j : task load size
    double resultFraction = 0.0;   // β_j : fraction returned as results (Phase 4; 0 = none)
};

// The combinatorial choice for a permutation schedule.
struct MlsdStructure {
    std::vector<int> taskOrder;             // taskOrder[l] = task id executed l-th
    std::vector<std::vector<int>> procSeq;  // procSeq[l] = processor activation sequence
                                            // for the l-th task (its set P_j + order)
};

// Forward declaration (defined above).
struct MlsdStructure;

// A solved MLSD schedule.
struct MlsdSolution {
    SolveStatus status = SolveStatus::NotSolved;
    double      makespan = 0.0;                  // Cmax
    std::vector<std::vector<double>> loads;      // loads[l][k] = load of l-th task to k-th proc
    std::vector<int> taskOrder;                  // the chosen task execution order (solver fills it)
};

class MlsdInstance {
public:
    MlsdInstance() = default;
    MlsdInstance(std::vector<MlsdTask> tasks, std::vector<Processor> processors)
        : tasks_(std::move(tasks)), processors_(std::move(processors)) {}

    std::vector<MlsdTask>&        tasks()       { return tasks_; }
    const std::vector<MlsdTask>&  tasks() const { return tasks_; }
    std::vector<Processor>&       processors()       { return processors_; }
    const std::vector<Processor>& processors() const { return processors_; }

    std::size_t numTasks() const      { return tasks_.size(); }
    std::size_t numProcessors() const { return processors_.size(); }

    // Goal:   check the instance is well-formed.
    // Input:  error - optional out-param set to a reason on failure.
    // Output: true if valid.
    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (tasks_.empty())      return fail("MLSD instance has no tasks");
        if (processors_.empty()) return fail("MLSD instance has no processors");
        for (std::size_t j = 0; j < tasks_.size(); ++j)
            if (tasks_[j].size <= 0.0)
                return fail("task " + std::to_string(j) + " has non-positive size");
        for (std::size_t i = 0; i < processors_.size(); ++i)
            if (!processors_[i].hasCost())
                return fail("processor " + std::to_string(i) + " has no per-unit cost");
        if (error) error->clear();
        return true;
    }

private:
    std::vector<MlsdTask>  tasks_;
    std::vector<Processor> processors_;
};

}  // namespace dls

#endif  // DLS_MLSD_INSTANCE_HPP
