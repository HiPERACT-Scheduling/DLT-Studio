//---------------------------------------------------------------------------
// exact/optv/optv_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// OptVSolver: the OptV ("how much load fits in a deadline?") direction of DLS,
// the dual of the usual makespan minimization (Berlińska thesis, ch. 2). Given
// a deadline T, it finds the activation sequence and load split that maximize
// the total processed load V subject to Cmax <= T.
//
// For a FIXED sequence the maximum load is a linear program — the shared model
// with the conservation rows dropped and Σαᵢ maximized (EvalObjective::MaxLoad,
// makespanLimit = T). This solver searches sequences (length 1..L, repeats
// optional) and keeps the one whose LP yields the most load, so it is exact
// over that sequence space — exactly the space the makespan ExactSolver covers,
// which makes the two duals consistent: OptV(T = OptT-makespan(V)) = V.
//
// The instance's totalLoad (V) is ignored here; the deadline T is the query and
// lives in OptVParams. The result's makespan is the achieved finish time (<= T)
// and totalAssignedLoad() is the maximized load.
//---------------------------------------------------------------------------

#ifndef DLS_EXACT_OPTV_SOLVER_HPP
#define DLS_EXACT_OPTV_SOLVER_HPP

#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"
#include "core/evaluator_factory.hpp"

namespace dls {

// Algorithm-specific parameters for the OptV solver.
struct OptVParams {
    double deadline = std::numeric_limits<double>::infinity();  // T (required, finite, > 0)
    int    maxInstallments = 5;     // search sequences of length <= this
    bool   allowRepeats    = true;  // true: a processor may appear several times
    int    lpMaxIterations = 1000;  // iteration cap for the per-sequence LP
    std::string evaluatorBackend = "simplex";   // "simplex" (default) or "highs"

    // Goal:   check the parameters before a run.
    // Input:  error - optional out-param; set to a reason when invalid.
    // Output: true if valid, false otherwise.
    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (!(deadline > 0.0 && deadline < std::numeric_limits<double>::infinity()))
            return fail("deadline (T) must be finite and positive");
        if (maxInstallments < 1) return fail("maxInstallments must be >= 1");
        if (lpMaxIterations < 1) return fail("lpMaxIterations must be >= 1");
        if (error) error->clear();
        return true;
    }
};

class OptVSolver : public DLSSolver {
public:
    // Goal:   build an OptV solver with a fixed configuration (incl. deadline T).
    explicit OptVSolver(OptVParams params) : params_(std::move(params)) {}

    std::string    name() const override     { return "optv"; }
    SolverCategory category() const override  { return SolverCategory::Exact; }

    // Goal:   maximize the load processable within the deadline T.
    // Input:  instance - processor costs (totalLoad is ignored); config - runtime.
    // Output: a DLSSolution; status Optimal with the max-load schedule
    //         (makespan <= T, totalAssignedLoad() = max V), or Infeasible/Failure.
    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override {
        DLSSolution best;
        std::string error;
        if (!instance.validate(&error)) { best.status = SolveStatus::Infeasible; return best; }
        if (!params_.validate(&error))  { best.status = SolveStatus::Failure;    return best; }

        backend_ = makeScheduleEvaluator(params_.evaluatorBackend);
        if (!backend_) { best.status = SolveStatus::Failure; return best; }   // unknown backend

        cfg_ = EvaluatorConfig{};
        cfg_.objective     = EvalObjective::MaxLoad;
        cfg_.makespanLimit = params_.deadline;       // Cmax <= T
        cfg_.maxIterations = params_.lpMaxIterations;
        cfg_.seed          = config.seed.value_or(0);

        const int N = static_cast<int>(instance.numProcessors());
        bestLoad_ = -1.0;
        evaluated_ = 0;
        std::vector<int> seq;
        std::vector<bool> used(N, false);
        enumerate(instance, N, seq, used, best);

        if (bestLoad_ < 0.0) { best.status = SolveStatus::Failure; return best; }  // nothing evaluated
        best.status     = SolveStatus::Optimal;      // exact over the searched space
        best.iterations = evaluated_;
        return best;
    }

private:
    OptVParams                          params_;
    std::unique_ptr<ScheduleEvaluator>  backend_;
    EvaluatorConfig                     cfg_;
    double                              bestLoad_  = -1.0;
    long                                evaluated_ = 0;

    // Goal:   depth-first enumeration of activation sequences (length 1..L),
    //         evaluating each MaxLoad LP and keeping the best.
    void enumerate(const DLSInstance& instance, int N, std::vector<int>& seq,
                   std::vector<bool>& used, DLSSolution& best) {
        if (!seq.empty()) {
            DLSSolution s = backend_->evaluate(instance, seq, cfg_);
            ++evaluated_;
            if (s.status == SolveStatus::Optimal && s.totalAssignedLoad() > bestLoad_) {
                bestLoad_ = s.totalAssignedLoad();
                best = s;
            }
        }
        if (static_cast<int>(seq.size()) >= params_.maxInstallments) return;
        for (int p = 0; p < N; ++p) {
            if (!params_.allowRepeats && used[p]) continue;
            seq.push_back(p); used[p] = true;
            enumerate(instance, N, seq, used, best);
            used[p] = false; seq.pop_back();
        }
    }
};

}  // namespace dls

#endif  // DLS_EXACT_OPTV_SOLVER_HPP
