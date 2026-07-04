//---------------------------------------------------------------------------
// heuristics/online/online_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// OnlineSolver: the online, simulation-based heuristics of Marszałkowski (2020),
// "Scheduling divisible computations with energy constraints", §6.2. Unlike every
// other single-load solver in the library, this one uses NO linear program: it
// runs a discrete-event simulation of the single-port master dispatching load
// chunks to heterogeneous, memory-limited workers, and reads the makespan (and
// energy) straight off the simulated schedule. That makes it very fast — O(V/ρ ·
// log m) chunks — and a natural fit for large, memory-bound instances.
//
// Two orthogonal knobs define a concrete heuristic (thesis §6.2):
//
//   * Processor Sorting Rule (PSR) — when several workers compete for the next
//     chunk, the master serves the one ranked first by a simple key (compute
//     rate a₁, comm rate C, startup S, RAM size, or in-core energy k₁). "super"
//     (Psr::All) runs every rule and keeps the best schedule (thesis: combining
//     simple rules is a cheap way to improve quality).
//
//   * Chunk sizing — SSC (Simple Static Chunk: αᵢ = ρ, the worker's RAM, never
//     going out of core) or GSS (Guided Self-Scheduling: αᵢ = min{Vˡ, max{minChunk,
//     min{Vˡ/m, ρ}}}, chunks shrinking as the remaining load Vˡ falls so the
//     workers' completion times even out at the end).
//
// Dispatch loop (thesis §6.2): start idle machines (by PSR) to fill the pipeline;
// once any worker has finished computing its chunk it becomes "ready" and is
// preferred for the next chunk. The single port serialises all transfers. The
// result is a feasible multi-installment schedule whose load split is fixed by
// the rule (not LP-optimised), so its makespan upper-bounds the LP optimum for
// the same activation order — the cross-check used in the tests.
//---------------------------------------------------------------------------

#ifndef DLS_HEURISTICS_ONLINE_SOLVER_HPP
#define DLS_HEURISTICS_ONLINE_SOLVER_HPP

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "core/dls_instance.hpp"
#include "core/dls_lp_model.hpp"   // scheduleCost
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"
#include "core/energy_model.hpp"   // scheduleEnergy, processingTime

namespace dls {

// Load-chunk sizing rule.
enum class ChunkRule {
    SSC,   // Simple Static Chunk: αᵢ = ρ (never out of core)
    GSS    // Guided Self-Scheduling: shrinking chunks that equalise finish times
};

// Processor Sorting Rule: the key by which competing workers are ranked
// (non-decreasing key = served first), plus the "super" meta-rule.
enum class Psr {
    ComputeRate,   // a₁: fastest in-core compute first
    CommRate,      // C : fastest link first
    CommStartup,   // S : lowest startup first
    MemoryDesc,    // largest RAM first (fewer, bigger chunks)
    EnergySlope,   // k₁: lowest in-core energy/load first
    All            // "super": try every concrete rule, keep the best makespan
};

// Algorithm-specific parameters for the online heuristic.
struct OnlineParams {
    ChunkRule chunk    = ChunkRule::GSS;   // chunk sizing rule
    Psr       psr      = Psr::All;         // sorting rule (default: super)
    double    minChunk = 1.0;              // GSS minimum chunk (thesis: 1 MB)
    int       maxChunks = 2000000;         // safety cap on the number of installments

    // Goal:   check the parameters are usable before a run.
    // Input:  error - optional out-param; set to a reason when invalid.
    // Output: true if valid, false otherwise.
    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (minChunk  <= 0.0) return fail("minChunk must be > 0");
        if (maxChunks < 1)    return fail("maxChunks must be >= 1");
        if (error) error->clear();
        return true;
    }
};

class OnlineSolver : public DLSSolver {
public:
    // Goal:   build an online heuristic solver with a fixed configuration.
    // Input:  params - the heuristic parameters (see OnlineParams).
    explicit OnlineSolver(OnlineParams params = {}) : params_(std::move(params)) {}

    // Goal:   stable identifier for registry / CLI selection.
    std::string name() const override { return "online"; }

    // Goal:   report this solver's category (no optimality guarantee).
    SolverCategory category() const override { return SolverCategory::Heuristic; }

    // Goal:   run the online heuristic on a DLS instance.
    // Input:  instance - the problem; config - shared runtime config (unused: the
    //         simulation is deterministic).
    // Output: a Feasible DLSSolution with the simulated schedule (fragments,
    //         makespan, cost, energy), or Infeasible/Failure on a bad instance.
    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override {
        (void)config;                                  // deterministic; no RNG
        DLSSolution result;
        std::string error;
        if (!instance.validate(&error)) { result.status = SolveStatus::Infeasible; return result; }
        if (!params_.validate(&error))  { result.status = SolveStatus::Failure;    return result; }

        if (params_.psr != Psr::All)
            return simulate(instance, params_.psr);

        // "super": run every concrete PSR, keep the shortest-makespan schedule.
        const Psr rules[] = { Psr::ComputeRate, Psr::CommRate, Psr::CommStartup,
                              Psr::MemoryDesc, Psr::EnergySlope };
        DLSSolution best;
        best.makespan = std::numeric_limits<double>::infinity();
        for (Psr r : rules) {
            DLSSolution s = simulate(instance, r);
            if (s.feasible() && s.makespan < best.makespan) best = std::move(s);
        }
        return best.feasible() ? best : simulate(instance, Psr::ComputeRate);
    }

private:
    OnlineParams params_;

    // Goal:   the PSR sort key of processor i (non-decreasing => higher priority).
    // Input:  p - the processor; rule - which concrete PSR.
    // Output: the key value (smaller is preferred).
    static double psrKey(const Processor& p, Psr rule) {
        switch (rule) {
            case Psr::ComputeRate: return p.effectivePieces().front().slope;       // a₁
            case Psr::CommRate:    return p.commRate;                              // C
            case Psr::CommStartup: return p.commStartup;                          // S
            case Psr::MemoryDesc:  return p.memoryLimit > 0.0 ? -p.memoryLimit     // larger RAM first
                                                              : -std::numeric_limits<double>::infinity();
            case Psr::EnergySlope: return p.effectiveEnergyPieces().front().slope; // k₁
            case Psr::All:         break;
        }
        return 0.0;
    }

    // Goal:   simulate the single-port online dispatch under one concrete PSR.
    // Input:  instance - the problem; rule - the (concrete) processor sorting rule.
    // Output: a Feasible DLSSolution, or Failure if no chunk can be placed.
    DLSSolution simulate(const DLSInstance& instance, Psr rule) const {
        const auto& procs = instance.processors();
        const int   N     = static_cast<int>(procs.size());
        const double V    = instance.totalLoad();
        const double eps  = 1e-9;

        std::vector<double> computeFinish(N, 0.0);   // when worker i finishes its backlog
        std::vector<bool>   started(N, false);       // has worker i ever received a chunk?
        double masterFree = 0.0;                     // when the single port is next free
        double remaining  = V;

        DLSSolution sol;
        int guard = 0;
        while (remaining > eps) {
            if (++guard > params_.maxChunks) { sol.status = SolveStatus::Failure; return sol; }
            double t = masterFree;

            // Candidates: workers ready to receive at time t. Prefer those already
            // started and done computing; otherwise start an idle (unstarted) one.
            int chosen = pick(procs, rule, started, computeFinish, t, /*wantReady=*/true);
            if (chosen < 0) chosen = pick(procs, rule, started, computeFinish, t, /*wantReady=*/false);
            if (chosen < 0) {
                // All started but none ready yet: advance the port to the next ready.
                double next = std::numeric_limits<double>::infinity();
                for (int i = 0; i < N; ++i) if (started[i]) next = std::min(next, computeFinish[i]);
                if (next == std::numeric_limits<double>::infinity()) { sol.status = SolveStatus::Failure; return sol; }
                masterFree = next;
                continue;
            }

            const Processor& p = procs[chosen];
            const double rho   = (p.memoryLimit > 0.0) ? p.memoryLimit : V;   // effective RAM cap
            double alpha;
            if (params_.chunk == ChunkRule::SSC) {
                alpha = std::min(remaining, rho);                            // αᵢ = ρ (clamped to remaining)
            } else { // GSS
                alpha = std::min(remaining, std::max(params_.minChunk,
                                                     std::min(remaining / N, rho)));
            }
            if (alpha <= eps) { sol.status = SolveStatus::Failure; return sol; }

            LoadFragment f;
            f.processorId   = chosen;
            f.loadSize      = alpha;
            f.commStart     = t;
            f.commFinish    = t + p.commStartup + p.commRate * alpha;
            f.computeStart  = std::max(f.commFinish, computeFinish[chosen]);
            f.computeFinish = f.computeStart + processingTime(p, alpha);

            masterFree           = f.commFinish;       // port busy through this transfer
            computeFinish[chosen] = f.computeFinish;
            started[chosen]       = true;
            sol.sequence.push_back(chosen);
            sol.fragments.push_back(f);
            remaining -= alpha;
        }

        double makespan = 0.0;
        for (int i = 0; i < N; ++i) makespan = std::max(makespan, computeFinish[i]);
        sol.status     = SolveStatus::Feasible;
        sol.makespan   = makespan;
        sol.iterations = static_cast<long>(sol.fragments.size());
        sol.cost       = scheduleCost(instance, sol.fragments);
        sol.energy     = scheduleEnergy(instance, sol.fragments, makespan);
        return sol;
    }

    // Goal:   choose the best worker for the next chunk at time t.
    // Input:  rule - the PSR; started/computeFinish - simulation state; t - the
    //         master's free time; wantReady - true => only already-started workers
    //         that have finished computing (ready) by t; false => only idle
    //         (never-started) workers.
    // Output: the chosen processor index, or -1 if no candidate qualifies.
    static int pick(const std::vector<Processor>& procs, Psr rule,
                    const std::vector<bool>& started,
                    const std::vector<double>& computeFinish,
                    double t, bool wantReady) {
        const double eps = 1e-9;
        int    best = -1;
        double bestKey = 0.0;
        for (int i = 0; i < static_cast<int>(procs.size()); ++i) {
            const bool ready = started[i] && computeFinish[i] <= t + eps;
            if (wantReady ? !ready : started[i]) continue;   // wrong category
            const double key = psrKey(procs[i], rule);
            if (best < 0 || key < bestKey) { best = i; bestKey = key; }  // ties: lower index
        }
        return best;
    }
};

}  // namespace dls

#endif  // DLS_HEURISTICS_ONLINE_SOLVER_HPP
