//---------------------------------------------------------------------------
// heuristics/best_rate/best_rate_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// BestRateSolver: the "best rate" constructive heuristic from Berlińska's
// thesis (section 3.5.4), exposed through the uniform dls::DLSSolver contract.
// It is fast (no search), deterministic, and — per the thesis experiments —
// near-Pareto-optimal: best quality per unit of run time across most instances.
//
// How it works. The solver builds the activation sequence greedily. It keeps a
// discrete-event simulation of the single-port master and the per-processor
// memory buffers (block allocation: a chunk of size α occupies α bytes from the
// moment it starts arriving until its computation finishes). At each step it
// considers sending the next chunk (of phantom size βᵢ = Bᵢ/x) to every free
// processor and picks the one with the best processing *rate* Tᵢ/βᵢ, where Tᵢ
// is the marginal time to get that chunk computed (eq. 3.24, generalized to the
// piecewise processing time). It repeats until the whole load V is placed.
//
// The greedy fixes only the *order* (the combinatorial part). The chunk sizes
// are then recomputed optimally by the shared ScheduleEvaluator LP — this is
// the "BRxLP" variant, which the thesis found essentially as good as the raw
// greedy and which makes the result consistent with every other solver and
// model feature (memory, p/r/d, cost, result return, either LP backend).
//
// Parameter split:
//   - PROBLEM   params come in via the DLSInstance argument of solve().
//   - RUNTIME   params (seed) come in via the SolverConfig argument.
//   - ALGORITHM params live in BestRateParams, supplied to the constructor.
//---------------------------------------------------------------------------

#ifndef DLS_HEURISTICS_BEST_RATE_SOLVER_HPP
#define DLS_HEURISTICS_BEST_RATE_SOLVER_HPP

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"
#include "core/evaluator_factory.hpp"   // backend selection by name

namespace dls {

// Algorithm-specific parameters for the best-rate heuristic.
struct BestRateParams {
    int chunkDivisor = 1;                  // the x in BRx: chunk size = Bᵢ/x (>=1).
                                           // x=1 => no chunk overlap (BR1, recommended);
                                           // larger x => smaller chunks, finer overlap.
    int maxCommunications = 1000;          // cap on sequence length (matches the thesis)
    int lpMaxIterations   = 1000;          // iteration cap for the chunk-sizing LP
    double costLimit =                     // G-bar forwarded to the LP (bi-criteria)
        std::numeric_limits<double>::infinity();
    std::string evaluatorBackend = "simplex";  // LP backend: "simplex" (default) or "highs"

    // Goal:   check the parameters are usable before a run.
    // Input:  error - optional out-param; set to a reason when invalid.
    // Output: true if valid, false otherwise.
    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (chunkDivisor      < 1) return fail("chunkDivisor (x) must be >= 1");
        if (maxCommunications < 1) return fail("maxCommunications must be >= 1");
        if (lpMaxIterations   < 1) return fail("lpMaxIterations must be >= 1");
        if (error) error->clear();
        return true;
    }
};

class BestRateSolver : public DLSSolver {
public:
    // Goal:   build a best-rate solver with a fixed algorithm configuration.
    // Input:  params - the heuristic parameters (see BestRateParams).
    // Output: a solver ready to accept solve() calls.
    explicit BestRateSolver(BestRateParams params = {}) : params_(std::move(params)) {}

    // Goal:   stable identifier for registry / CLI selection.
    std::string name() const override { return "best-rate"; }

    // Goal:   report this solver's category (no optimality guarantee).
    SolverCategory category() const override { return SolverCategory::Heuristic; }

    // Goal:   run the best-rate heuristic on a DLS instance.
    // Input:  instance - the problem; config - shared runtime config.
    // Output: a DLSSolution; Feasible with the greedy sequence's optimal load
    //         split, or Infeasible/Failure on a bad instance / backend.
    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override {
        DLSSolution result;

        std::string error;
        if (!instance.validate(&error)) { result.status = SolveStatus::Infeasible; return result; }
        if (!params_.validate(&error))  { result.status = SolveStatus::Failure;    return result; }

        // ---- greedy: build the activation sequence ------------------------
        std::vector<int> sequence = buildSequence(instance);
        if (sequence.empty()) { result.status = SolveStatus::Failure; return result; }

        // ---- optimal chunk sizes for that sequence via the shared LP ------
        std::unique_ptr<ScheduleEvaluator> backend = makeScheduleEvaluator(params_.evaluatorBackend);
        if (!backend) { result.status = SolveStatus::Failure; return result; }   // unknown backend

        const std::uint64_t seed = config.seed.value_or(0);   // deterministic; LP anti-cycling only
        EvaluatorConfig ec;
        ec.maxIterations = params_.lpMaxIterations;
        ec.seed          = seed;
        ec.costLimit     = params_.costLimit;

        result = backend->evaluate(instance, sequence, ec);
        // The LP proves optimality only for this fixed order; as a heuristic
        // over orders, the overall result carries no global optimality proof.
        if (result.status == SolveStatus::Optimal) result.status = SolveStatus::Feasible;
        result.usedSeed   = seed;
        result.iterations = static_cast<long>(sequence.size());
        return result;
    }

private:
    BestRateParams params_;

    // Goal:   processing time of `load` on processor p (max over convex pieces).
    static double procTime(const Processor& p, double load) {
        double t = 0.0;
        for (const ComputePiece& pc : p.effectivePieces())
            t = std::max(t, pc.intercept + pc.slope * load);
        return t;
    }

    // Goal:   earliest time >= t0 at which processor `i` has `chunk` free memory.
    // Input:  held - (computeFinish, size) of chunks still occupying the buffer
    //         (all already received by t0); B - buffer size; chunk - needed size.
    // Output: t0 if it already fits, else the finish time of just enough chunks.
    static double earliestMemoryFree(std::vector<std::pair<double,double>> held,
                                     double B, double chunk, double t0) {
        if (B <= 0.0) return t0;                       // unbounded memory
        const double eps = 1e-9;
        double occupied = 0.0;
        for (const auto& h : held) if (h.first > t0) occupied += h.second;
        if (B - occupied >= chunk - eps) return t0;
        std::sort(held.begin(), held.end());           // by finish time ascending
        for (const auto& h : held) {
            if (h.first <= t0) continue;
            occupied -= h.second;                      // this chunk's memory is released
            if (B - occupied >= chunk - eps) return h.first;
        }
        return t0;                                     // chunk <= B guarantees a fit
    }

    // Goal:   build the activation sequence by the best-rate greedy.
    // Input:  the instance.
    // Output: an ordered list of processor indices (one per chunk sent).
    std::vector<int> buildSequence(const DLSInstance& instance) const {
        const auto& procs = instance.processors();
        const int N = static_cast<int>(procs.size());
        const double V = instance.totalLoad();
        const double eps = 1e-9;
        const double x = static_cast<double>(params_.chunkDivisor);

        std::vector<double> compFree(N, 0.0);                       // tᵢ: end of compute backlog
        std::vector<std::vector<std::pair<double,double>>> held(N); // memory occupancy per processor
        double t0 = 0.0;                                            // originator free time
        double remaining = V;
        std::vector<int> sequence;

        while (remaining > eps && static_cast<int>(sequence.size()) < params_.maxCommunications) {
            int    bestI = -1;
            double bestRate = std::numeric_limits<double>::infinity();
            double bestReceiveEnd = 0.0, bestComputeEnd = 0.0, bestChunk = 0.0;

            for (int i = 0; i < N; ++i) {
                const Processor& p = procs[i];
                const double cap   = (p.memoryLimit > 0.0) ? p.memoryLimit / x : V;
                const double chunk = std::min(cap, remaining);
                if (chunk <= eps) continue;

                const double tauI         = earliestMemoryFree(held[i], p.memoryLimit, chunk, t0);
                const double receiveStart = std::max(t0, tauI);
                const double receiveEnd    = receiveStart + p.commStartup + chunk * p.commRate;
                const double computeStart  = std::max(receiveEnd, compFree[i]);
                const double computeEnd     = computeStart + procTime(p, chunk);
                const double rate           = (computeEnd - t0) / chunk;   // Tᵢ / βᵢ

                if (rate < bestRate) {
                    bestRate = rate; bestI = i;
                    bestReceiveEnd = receiveEnd; bestComputeEnd = computeEnd; bestChunk = chunk;
                }
            }
            if (bestI < 0) break;                       // no processor can accept load

            sequence.push_back(bestI);
            t0             = bestReceiveEnd;            // single port busy until the send completes
            compFree[bestI] = bestComputeEnd;
            held[bestI].push_back({bestComputeEnd, bestChunk});
            remaining -= bestChunk;
        }
        return sequence;
    }
};

}  // namespace dls

#endif  // DLS_HEURISTICS_BEST_RATE_SOLVER_HPP
