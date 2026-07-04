//---------------------------------------------------------------------------
// exact/enumerative/exact_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// ExactSolver: a provably-optimal DLS solver exposed through the uniform
// dls::DLSSolver contract. It branch-and-bounds over activation SEQUENCES and
// reuses the core ScheduleEvaluator for the exact per-sequence load split, so
// it shares all the inner machinery with the GA.
//
// Search space (selectable via ExactParams):
//   - multi-installment: every sequence of length 1..L over the N processors
//     (a processor may appear several times),
//   - single-installment: each processor used at most once (allowRepeats=false).
// The result is optimal over that space ("optimal with at most L installments").
//
// Branch-and-bound bounds (both provably <= the makespan of any completion of
// the current prefix, so pruning never discards the true optimum):
//   - fluid bound:    Cmax >= V / sum(1/A_i) over all processors (using every
//                     processor is the most capacity, hence the loosest bound;
//                     if any A_i == 0 the bound is 0).
//   - startup bound:  single-port communication is sequential and never
//                     overlaps, so Cmax >= sum of the installment startups S;
//                     a prefix's startup sum only grows as it is extended.
// A child is pruned when max(fluid, prefixStartup + S_child) >= incumbent.
//
// Note: with zero startup costs (S = 0) the startup bound does not prune, so
// the search approaches full enumeration; pruning is effective when S > 0.
// Tighter sequence-dependent bounds are possible future work.
//---------------------------------------------------------------------------

#ifndef DLS_EXACT_EXACT_SOLVER_HPP
#define DLS_EXACT_EXACT_SOLVER_HPP

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"
#include "core/evaluator_factory.hpp"
#include "core/caching_schedule_evaluator.hpp"
#include "heuristics/single_round/single_round_solver.hpp"   // warm-start incumbent
#include "heuristics/best_rate/best_rate_solver.hpp"          // warm-start incumbent

namespace dls {

// Algorithm-specific parameters for the exact solver.
struct ExactParams {
    int  maxInstallments = 5;     // L: maximum activation-sequence length
    bool allowRepeats    = true;  // true: multi-installment; false: each processor once
    long nodeBudget      = 0;     // 0 = exhaustive; else cap explored nodes (best-so-far, Feasible)
    int  lpMaxIterations = 500;   // per-sequence LP iteration cap
    bool warmStart       = true;  // true: seed the incumbent from fast heuristics (prunes harder,
                                  // same optimum); false: the original B&B (no warm start)
    double costLimit =            // G-bar: minimize Cmax subject to cost G <= this
        std::numeric_limits<double>::infinity();   // (inf = no cost limit)
    std::string evaluatorBackend = "simplex";  // LP backend: "simplex" (default) or "highs"

    // Goal:   check the parameters are usable.
    // Input:  error - optional out-param set to a reason when invalid.
    // Output: true if valid, false otherwise.
    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (maxInstallments < 1) return fail("maxInstallments must be >= 1");
        if (lpMaxIterations < 1) return fail("lpMaxIterations must be >= 1");
        if (nodeBudget      < 0) return fail("nodeBudget must be >= 0");
        if (error) error->clear();
        return true;
    }
};

class ExactSolver : public DLSSolver {
public:
    // Goal:   build an exact solver with a fixed configuration.
    // Input:  params - exact-search parameters (see ExactParams).
    // Output: a solver ready to accept solve() calls.
    explicit ExactSolver(ExactParams params) : params_(std::move(params)) {}

    // Goal:   stable identifier for registry / CLI selection.
    // Output: "exact".
    std::string name() const override { return "exact"; }

    // Goal:   report this solver's category.
    // Output: SolverCategory::Exact.
    SolverCategory category() const override { return SolverCategory::Exact; }

    // Goal:   find a provably optimal schedule (over the configured space).
    // Input:  instance - the problem; config - shared runtime config (seed is
    //         only used for the LP's deterministic anti-cycling path; the
    //         optimum itself is seed-independent).
    // Output: a DLSSolution. Status Optimal when the search completes,
    //         Feasible if a node budget was hit before completion, Infeasible
    //         if the instance is invalid or no feasible sequence exists,
    //         Failure if the parameters are invalid.
    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override {
        DLSSolution result;

        std::string error;                       // validation message, if any
        if (!instance.validate(&error)) { result.status = SolveStatus::Infeasible; return result; }
        if (!params_.validate(&error))  { result.status = SolveStatus::Failure;    return result; }

        // The exact optimum must not depend on RNG, so the LP uses a FIXED seed
        // (the config seed is irrelevant here). This makes the solver fully
        // deterministic: identical results for any SolverConfig.
        (void)config;
        const std::uint64_t lpSeed = 0;
        result.usedSeed = lpSeed;

        std::unique_ptr<ScheduleEvaluator> backend = makeScheduleEvaluator(params_.evaluatorBackend);
        if (!backend) { result.status = SolveStatus::Failure; return result; }  // unknown backend
        CachingScheduleEvaluator evaluator(*backend); // memoize (harmless; reusable)
        EvaluatorConfig ecfg;
        ecfg.maxIterations = params_.lpMaxIterations;
        ecfg.seed          = lpSeed;
        ecfg.costLimit     = params_.costLimit;   // bi-criteria: min Cmax s.t. G <= G-bar

        const int N = static_cast<int>(instance.numProcessors());   // processor count
        int L = params_.maxInstallments;                            // effective max length
        if (!params_.allowRepeats && L > N) L = N;                  // can't exceed N distinct

        // Fluid lower bound: V / sum(1/A_i) over all processors (0 if any A_i == 0).
        const double fluidLB = computeFluidBound(instance);

        // Ideal-processor lower bound (from the thesis): a hypothetical best
        // processor with the smallest S, C, A over all real ones. Completing a
        // partial sequence with copies of the ideal processor and solving the LP
        // gives a valid lower bound on every real completion's makespan — it
        // prunes effectively even when startups are zero.
        double minS = std::numeric_limits<double>::infinity();
        double minC = std::numeric_limits<double>::infinity();
        double minA = std::numeric_limits<double>::infinity();
        for (const Processor& p : instance.processors()) {
            minS = std::min(minS, p.commStartup);
            minC = std::min(minC, p.commRate);
            minA = std::min(minA, p.computeRate);
        }
        // The relaxation completes a prefix with up to L DISTINCT ideal
        // processors (each doing one chunk, in parallel) — not one ideal repeated
        // (which the model would serialise). The fill uses commStartup = 0 (NOT
        // minS): appended to the single port, an ideal with a positive startup
        // would inject that startup into the makespan even at zero load, making
        // the "lower" bound exceed a real completion that uses fewer processors
        // (invalid — it could prune the optimum). With startup 0 the fill is
        // optimistic in every dimension (0<=realS, minC<=realC, minA<=realA), so
        // it stays a valid lower bound at the single completion length L.
        (void)minS;
        DLSInstance augmented = instance;                  // real processors + L ideals
        for (int k = 0; k < L; ++k) {
            Processor ideal;                               // best C/A, zero startup, no limits
            ideal.commStartup = 0.0; ideal.commRate = minC;
            ideal.computeRate = minA; ideal.memoryLimit = 1e18;
            ideal.name = "ideal";
            augmented.processors().push_back(ideal);
        }

        // Lower bound on every completion of `prefix`: solve the LP for prefix
        // followed by (L - l) distinct ideal processors. Returns 0 (no prune) if
        // that relaxation is not optimal.
        auto idealLowerBound = [&](const std::vector<int>& prefix) -> double {
            std::vector<int> seq = prefix;
            int k = 0;
            for (int i = static_cast<int>(prefix.size()); i < L; ++i) seq.push_back(N + k++);
            DLSSolution s = evaluator.evaluate(augmented, seq, ecfg);
            return (s.status == SolveStatus::Optimal) ? s.makespan : 0.0;
        };

        // Branch in a deterministic order, fastest compute (smallest A) first,
        // to drive the incumbent down quickly (and keep results reproducible).
        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            double ca = instance.processors()[a].computeRate;
            double cb = instance.processors()[b].computeRate;
            if (ca != cb) return ca < cb;
            return a < b;                            // stable tie-break for determinism
        });

        // ---- search state ------------------------------------------------
        double      incumbent = std::numeric_limits<double>::infinity(); // best makespan
        bool        haveSol   = false;              // any feasible sequence found
        DLSSolution bestEval;                        // evaluator result for the best sequence
        long        nodes     = 0;                   // sequences evaluated
        bool        budgetHit = false;               // node budget exhausted
        std::vector<int>  prefix;                    // current activation sequence
        std::vector<bool> used(N, false);            // processor usage (for !allowRepeats)
        prefix.reserve(L);

        // ---- warm start (optional): seed the incumbent with fast heuristics --
        // A near-optimal initial upper bound lets the bound prune from the very
        // first node. Each candidate sequence is re-scored by THIS solver's
        // evaluator + ecfg (so cost / availability limits are honoured), and only
        // sequences within the search space are used: length <= L, and (when
        // allowRepeats is false) no repeated processor. Seeding only lowers the
        // incumbent, so it never affects the proven optimum — purely a speedup.
        // Disable it (params_.warmStart = false) for the original B&B behaviour.
        if (params_.warmStart) {
            auto trySeed = [&](const std::vector<int>& seq) {
                if (seq.empty() || static_cast<int>(seq.size()) > L) return;
                if (!params_.allowRepeats) {              // must stay in the single-installment space
                    std::vector<bool> seen(N, false);
                    for (int p : seq) { if (p < 0 || p >= N || seen[p]) return; seen[p] = true; }
                }
                DLSSolution s = evaluator.evaluate(instance, seq, ecfg);
                if (s.status == SolveStatus::Optimal && s.makespan < incumbent) {
                    incumbent = s.makespan; bestEval = s; haveSol = true;
                }
            };
            trySeed(SingleRoundSolver().solve(instance, config).sequence);
            BestRateParams bp;
            bp.evaluatorBackend = params_.evaluatorBackend;
            bp.costLimit        = params_.costLimit;
            trySeed(BestRateSolver(bp).solve(instance, config).sequence);
        }

        // DFS: evaluate the current prefix, then extend (depth-first).
        std::function<void(double)> search = [&](double prefixStartup) {
            if (params_.nodeBudget > 0 && nodes >= params_.nodeBudget) { budgetHit = true; return; }

            if (!prefix.empty()) {                   // every prefix (length >= 1) is a candidate
                // Prune the whole subtree if even the ideal completion of this
                // prefix cannot beat the incumbent (valid: ideal <= any real).
                if (static_cast<int>(prefix.size()) < L && idealLowerBound(prefix) >= incumbent) return;

                ++nodes;
                DLSSolution s = evaluator.evaluate(instance, prefix, ecfg);
                if (s.status == SolveStatus::Optimal && s.makespan < incumbent) {
                    incumbent = s.makespan;
                    bestEval  = s;
                    haveSol   = true;
                }
            }

            if (static_cast<int>(prefix.size()) >= L) return;   // cannot extend further

            for (int c : order) {
                if (!params_.allowRepeats && used[c]) continue;
                double childStartup = prefixStartup + instance.processors()[c].commStartup;
                // Prune: no completion through this child can beat the incumbent.
                if (std::max(fluidLB, childStartup) >= incumbent) continue;

                prefix.push_back(c); used[c] = true;
                search(childStartup);
                prefix.pop_back();  used[c] = false;
                if (budgetHit) return;
            }
        };
        search(0.0);

        if (!haveSol) { result.status = SolveStatus::Infeasible; result.usedSeed = lpSeed; return result; }

        result            = bestEval;               // makespan, sequence, fragments
        result.status     = budgetHit ? SolveStatus::Feasible : SolveStatus::Optimal;
        result.iterations = nodes;
        result.usedSeed   = lpSeed;
        return result;
    }

private:
    ExactParams params_;   // immutable configuration

    // Goal:   fluid (no-communication, all-parallel) lower bound on makespan.
    // Input:  instance - the problem.
    // Output: V / sum(1/A_i) over all processors; 0 if any A_i == 0 (an
    //         infinitely fast computer makes the bound trivial but still valid).
    static double computeFluidBound(const DLSInstance& instance) {
        double invSum = 0.0;                        // sum of compute rates (1/A_i)
        for (const Processor& p : instance.processors()) {
            if (p.computeRate <= 0.0) return 0.0;   // infinite capacity -> bound is 0
            invSum += 1.0 / p.computeRate;
        }
        return (invSum > 0.0) ? instance.totalLoad() / invSum : 0.0;
    }
};

}  // namespace dls

#endif  // DLS_EXACT_EXACT_SOLVER_HPP
