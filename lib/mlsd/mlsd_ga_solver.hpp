//---------------------------------------------------------------------------
// mlsd/mlsd_ga_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Genetic-algorithm heuristic for MLSD. A genome IS an MlsdStructure (task order
// + per-task processor sequences); fitness is the LP makespan from the
// MlsdScheduleEvaluator. For the large structure space where the exact solver is
// intractable, the GA searches good structures. Deterministic from a seed.
//---------------------------------------------------------------------------

#ifndef DLS_MLSD_GA_SOLVER_HPP
#define DLS_MLSD_GA_SOLVER_HPP

#include <algorithm>
#include <limits>
#include <numeric>
#include <vector>

#include "mlsd/mlsd_evaluator.hpp"
#include "mlsd/mlsd_instance.hpp"
#include "util/random.hpp"

namespace dls {

class MlsdGaSolver {
public:
    struct Params {
        int populationSize  = 24;
        int maxGenerations  = 150;
        int tournamentSize  = 3;
        double mutationProb = 0.3;   // per-genome probability of each mutation kind

        [[nodiscard]] bool validate(std::string* error = nullptr) const {
            auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
            if (populationSize < 1) return fail("populationSize must be >= 1");
            if (maxGenerations < 1) return fail("maxGenerations must be >= 1");
            if (tournamentSize < 1) return fail("tournamentSize must be >= 1");
            if (!(mutationProb >= 0.0 && mutationProb <= 1.0)) return fail("mutationProb must be in [0, 1]");
            if (error) error->clear();
            return true;
        }
    };

    explicit MlsdGaSolver(Params p) : params_(p) {}

    // Goal:   search MLSD structures heuristically; return the best schedule.
    // Input:  inst - the problem; seed - RNG seed (reproducible).
    // Output: MlsdSolution (Feasible with the best structure, or Infeasible).
    MlsdSolution solve(const MlsdInstance& inst, std::uint64_t seed = 0) const {
        MlsdSolution result;
        std::string error;
        if (!inst.validate(&error))    { result.status = SolveStatus::Infeasible; return result; }
        if (!params_.validate(&error)) { result.status = SolveStatus::Infeasible; return result; }

        util::Rng rng(seed);
        const int n = static_cast<int>(inst.numTasks());
        const int m = static_cast<int>(inst.numProcessors());
        MlsdScheduleEvaluator ev;
        const double INF = std::numeric_limits<double>::infinity();

        auto randProcSeq = [&]() {
            std::vector<int> all(m);
            std::iota(all.begin(), all.end(), 0);
            for (int i = m - 1; i > 0; --i) std::swap(all[i], all[rng.index(i + 1)]);  // shuffle
            all.resize(1 + rng.index(m));                                              // random size 1..m
            return all;
        };
        auto randStruct = [&]() {
            MlsdStructure s;
            s.taskOrder.resize(n);
            std::iota(s.taskOrder.begin(), s.taskOrder.end(), 0);
            for (int i = n - 1; i > 0; --i) std::swap(s.taskOrder[i], s.taskOrder[rng.index(i + 1)]);
            s.procSeq.resize(n);
            for (int l = 0; l < n; ++l) s.procSeq[l] = randProcSeq();
            return s;
        };
        auto fitness = [&](const MlsdStructure& s) {
            MlsdSolution r = ev.evaluate(inst, s);
            return r.status == SolveStatus::Optimal ? r.makespan : INF;
        };

        // ---- initial population -----------------------------------------
        const int Pn = params_.populationSize;
        std::vector<MlsdStructure> pop(Pn);
        std::vector<double> fit(Pn);
        MlsdStructure best; double bestFit = INF;
        for (int i = 0; i < Pn; ++i) {
            pop[i] = randStruct(); fit[i] = fitness(pop[i]);
            if (fit[i] < bestFit) { bestFit = fit[i]; best = pop[i]; }
        }

        auto tournament = [&]() {                     // index of a tournament winner
            int b = rng.index(Pn);
            for (int t = 1; t < params_.tournamentSize; ++t) {
                int c = rng.index(Pn);
                if (fit[c] < fit[b]) b = c;
            }
            return b;
        };

        // ---- evolve ------------------------------------------------------
        for (int gen = 0; gen < params_.maxGenerations; ++gen) {
            std::vector<MlsdStructure> next(Pn);
            next[0] = best;                            // elitism
            for (int i = 1; i < Pn; ++i) {
                const MlsdStructure& pa = pop[tournament()];
                const MlsdStructure& pb = pop[tournament()];
                MlsdStructure child;
                child.taskOrder = pa.taskOrder;        // order from one parent
                child.procSeq.resize(n);
                for (int l = 0; l < n; ++l)            // each task's procSeq from either parent
                    child.procSeq[l] = (rng.uniform01() < 0.5) ? pa.procSeq[l] : pb.procSeq[l];
                // mutation: swap two positions in the task order
                if (n >= 2 && rng.uniform01() < params_.mutationProb) {
                    int a = rng.index(n), b = rng.index(n);
                    std::swap(child.taskOrder[a], child.taskOrder[b]);
                }
                // mutation: reassign a random task's processor sequence
                if (rng.uniform01() < params_.mutationProb)
                    child.procSeq[rng.index(n)] = randProcSeq();
                next[i] = child;
            }
            pop.swap(next);
            for (int i = 0; i < Pn; ++i) {
                fit[i] = fitness(pop[i]);
                if (fit[i] < bestFit) { bestFit = fit[i]; best = pop[i]; }
            }
        }

        if (bestFit == INF) { result.status = SolveStatus::Infeasible; return result; }
        result = ev.evaluate(inst, best);             // recompute loads for the best structure
        result.status = SolveStatus::Feasible;        // heuristic: no optimality guarantee
        result.taskOrder = best.taskOrder;
        return result;
    }

private:
    Params params_;
};

}  // namespace dls

#endif  // DLS_MLSD_GA_SOLVER_HPP
