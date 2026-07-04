//---------------------------------------------------------------------------
// exact/dual/dual_bisection_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// A genuinely different exact method for the single-load makespan, used mainly
// as an INDEPENDENT cross-check on the primal solvers (B&B, MILP). Instead of
// minimizing the makespan directly, it solves the OptV *decision* problem —
// "what is the most load processable within a deadline T?" — and bisects on T:
//
//   minimize_σ  makespan(σ)  =  min { T : max_σ OptV(σ, T) >= V }.
//
// The equality holds because for a fixed sequence the min makespan for load V
// and the max load within deadline T are duals, and the min/max over sequences
// commute (min_σ min{T:V(σ,T)>=V} = min{T : exists σ with V(σ,T)>=V}). So this
// reaches the same optimum as ExactSolver over the same sequence space — by an
// entirely different route (a sequence of feasibility decisions + bisection),
// reusing the exact OptVSolver.
//
// Scope / cost: it enumerates sequences inside each OptV call (no pruning) once
// per bisection step, so it is for small instances / validation, not speed. It
// needs a feasible in-space single-installment schedule to seed the upper bound
// (true when memory is ample); otherwise it reports Failure.
//---------------------------------------------------------------------------

#ifndef DLS_EXACT_DUAL_BISECTION_SOLVER_HPP
#define DLS_EXACT_DUAL_BISECTION_SOLVER_HPP

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "core/bounds.hpp"
#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"
#include "core/evaluator_factory.hpp"
#include "exact/optv/optv_solver.hpp"

namespace dls {

struct DualBisectionParams {
    int    maxInstallments = 5;      // search length (shared with the OptV enumeration)
    bool   allowRepeats    = true;   // multi-installment vs each processor once
    int    bisectionSteps  = 60;     // deadline bisection iterations
    std::string evaluatorBackend = "simplex";

    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (maxInstallments < 1) return fail("maxInstallments must be >= 1");
        if (bisectionSteps   < 1) return fail("bisectionSteps must be >= 1");
        if (error) error->clear();
        return true;
    }
};

class DualBisectionSolver : public DLSSolver {
public:
    explicit DualBisectionSolver(DualBisectionParams params = {}) : params_(std::move(params)) {}

    std::string    name() const override     { return "exact-dual"; }
    SolverCategory category() const override  { return SolverCategory::Exact; }

    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override {
        DLSSolution result;
        std::string error;
        if (!instance.validate(&error)) { result.status = SolveStatus::Infeasible; return result; }
        if (!params_.validate(&error))  { result.status = SolveStatus::Failure;    return result; }

        std::unique_ptr<ScheduleEvaluator> backend = makeScheduleEvaluator(params_.evaluatorBackend);
        if (!backend) { result.status = SolveStatus::Failure; return result; }

        const int N = static_cast<int>(instance.numProcessors());
        const int L = params_.maxInstallments;
        const double V = instance.totalLoad();
        EvaluatorConfig ecfg; ecfg.maxIterations = 1000; ecfg.seed = config.seed.value_or(0);

        // Upper bound: a feasible in-space single-installment schedule (the first
        // min(N, L) processors, each once). Its makespan bounds the optimum.
        std::vector<int> seedSeq;
        for (int i = 0; i < std::min(N, L); ++i) seedSeq.push_back(i);
        DLSSolution seed = backend->evaluate(instance, seedSeq, ecfg);
        if (!seed.feasible()) { result.status = SolveStatus::Failure; return result; }  // needs ample memory

        double lo = divisibleLoadLowerBoundTight(instance);   // valid lower bound on T*
        double hi = seed.makespan;                            // valid upper bound on T*
        std::vector<int> bestSeq = seedSeq;                   // best in-space sequence found

        // Bisect: feasible(T) iff some sequence processes >= V within deadline T.
        for (int it = 0; it < params_.bisectionSteps && hi - lo > 1e-9 * (1.0 + hi); ++it) {
            const double T = 0.5 * (lo + hi);
            OptVParams op;
            op.deadline = T; op.maxInstallments = L; op.allowRepeats = params_.allowRepeats;
            op.evaluatorBackend = params_.evaluatorBackend;
            DLSSolution ov = OptVSolver(op).solve(instance, config);
            if (ov.status == SolveStatus::Optimal && ov.totalAssignedLoad() >= V - 1e-6 * (1.0 + V)) {
                hi = T; bestSeq = ov.sequence;                // T is achievable for load V
            } else {
                lo = T;                                       // T too tight
            }
        }

        // Exact makespan = the best sequence evaluated for the true load V.
        result = backend->evaluate(instance, bestSeq, ecfg);
        if (result.feasible()) result.status = SolveStatus::Optimal;   // proven over the searched space
        result.usedSeed   = ecfg.seed;
        result.iterations = params_.bisectionSteps;
        return result;
    }

private:
    DualBisectionParams params_;
};

}  // namespace dls

#endif  // DLS_EXACT_DUAL_BISECTION_SOLVER_HPP
