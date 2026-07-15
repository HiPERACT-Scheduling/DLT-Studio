//---------------------------------------------------------------------------
// heuristics/ga/ga_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// GASolver: the genetic-algorithm heuristic exposed through the uniform
// dls::DLSSolver contract. A caller hands it a DLSInstance plus shared
// SolverConfig and receives a DLSSolution, exactly like any other solver.
//
// Parameter split:
//   - PROBLEM  params come in via the DLSInstance argument of solve().
//   - RUNTIME  params (seed, ...) come in via the SolverConfig argument.
//   - ALGORITHM params live in GAParams, supplied to the constructor.
//
// This file is the orchestration layer over the legacy GA machinery in
// ga.hpp (Population/Chromosome/GenotypesList and the *Parameters structs).
// It converts to/from those internal structures so the GA's evolutionary
// loop stays unchanged.
//
// STAGE 1: fitness is still evaluated by the GA's inline-LP
// Population::EvaluatePopulation(). Stage 2 replaces that with a
// dls::ScheduleEvaluator call; the public contract here does not change.
//---------------------------------------------------------------------------

#ifndef DLS_HEURISTICS_GA_SOLVER_HPP
#define DLS_HEURISTICS_GA_SOLVER_HPP

#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"
#include "core/energy_model.hpp"                // scheduleEnergy
#include "core/evaluator_factory.hpp"           // backend selection by name
#include "core/caching_schedule_evaluator.hpp"  // memoizes solves by sequence
#include "util/random.hpp"                       // injectable RNG
#include "ga.hpp"   // legacy GA machinery (global ::Population, ::Gene, ...)

namespace dls {

// Genetic operator strategies (each is dispatched in the GA machinery).
enum class Selection { Roulette,    // fitness-proportionate, with elitism (default)
                       Tournament   // best of N random contenders per slot
                     };
enum class Crossover { SinglePoint, // swap tails at one random cut (default)
                       TwoPoint     // swap the middle gene segment between parents
                     };
enum class Mutation  { PerGene,     // reassign a gene to a random processor (default)
                       Swap         // exchange two genes' processors (reorders installments)
                     };

// Algorithm-specific parameters for the genetic algorithm. Defaults mirror
// the legacy ga_par.txt values.
struct GAParams {
    int    populationSize     = 10;    // number of genotypes kept per generation
    int    maxGenerations     = 1000;  // hard cap on generations (legacy maxIterNum)
    int    noImprovementLimit = 100;   // stop after this many non-improving gens
    int    installments       = 5;     // genes per chromosome (legacy chromosomeMaxSize)
    double crossoverProb      = 3.0;   // expected crossovers per population (legacy scale)
    double mutationProb       = 1.0;   // expected mutations per population*genes (legacy scale)
    int    lpMaxIterations    = 500;   // iteration cap for the per-genotype LP solve
    double costLimit =                 // G-bar: minimize Cmax subject to cost G <= this
        std::numeric_limits<double>::infinity();   // (inf = no cost limit)
    std::string evaluatorBackend = "simplex";  // LP backend: "simplex" (default) or "highs"

    Selection selection      = Selection::Roulette;     // parent-selection strategy
    int       tournamentSize = 3;                        // contenders when selection == Tournament
    Crossover crossover      = Crossover::SinglePoint;   // recombination strategy
    Mutation  mutation       = Mutation::PerGene;        // mutation strategy

    // Goal:   check the parameters are usable before a run.
    // Input:  error - optional out-param; set to a reason when invalid.
    // Output: true if valid, false otherwise.
    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (populationSize  < 1) return fail("populationSize must be >= 1");
        if (installments    < 1) return fail("installments must be >= 1");
        if (maxGenerations  < 0) return fail("maxGenerations must be >= 0");
        if (noImprovementLimit < 0) return fail("noImprovementLimit must be >= 0");
        if (crossoverProb   < 0.0) return fail("crossoverProb must be >= 0");
        if (mutationProb    < 0.0) return fail("mutationProb must be >= 0");
        if (lpMaxIterations < 1) return fail("lpMaxIterations must be >= 1");
        if (selection == Selection::Tournament && tournamentSize < 1)
            return fail("tournamentSize must be >= 1 for tournament selection");
        if (error) error->clear();
        return true;
    }
};

class GASolver : public DLSSolver {
public:
    // Goal:   build a GA solver with a fixed algorithm configuration.
    // Input:  params - the GA-specific parameters (see GAParams).
    // Output: a solver ready to accept solve() calls.
    explicit GASolver(GAParams params) : params_(std::move(params)) {}

    // Goal:   stable identifier for registry / CLI selection.
    // Output: the solver name, "ga".
    std::string name() const override { return "ga"; }

    // Goal:   report this solver's category.
    // Output: SolverCategory::Heuristic.
    SolverCategory category() const override { return SolverCategory::Heuristic; }

    // Goal:   run the genetic algorithm on a DLS instance.
    // Input:  instance - the problem (processor costs, total load V);
    //         config   - shared runtime config (seed is honoured here).
    // Output: a DLSSolution; status Feasible with the best schedule found,
    //         or Infeasible/Failure if validation fails or nothing was found.
    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override {
        DLSSolution result;  // value returned to the caller

        std::string error;   // validation message, if any
        if (!instance.validate(&error)) {
            result.status = SolveStatus::Infeasible;   // bad problem
            return result;
        }
        if (!params_.validate(&error)) {
            result.status = SolveStatus::Failure;       // bad algorithm params
            return result;
        }

        // Resolve the seed: explicit if given, else drawn once from the OS.
        // It is recorded in the result so an unseeded run is reproducible.
        std::uint64_t seed = config.seed.has_value()
                                 ? *config.seed
                                 : std::random_device{}();
        result.usedSeed = seed;
        util::Rng rng(seed);                 // this run's generator (thread-local to the call)

        // ---- translate problem + params into the legacy GA structures ----
        ::GeneticAlgorithmParameters gaPar;  // GA control knobs
        fillGaParams(gaPar);

        ::ProcessorsParameters procData;     // processor cost table (1-based rows)
        fillProcessors(procData, instance);

        ::ProgrammeParameters progPar;       // runtime knobs the GA reads
        progPar.solverMaxIter = params_.lpMaxIterations;

        // ---- wire in the core schedule evaluator (fitness backend) -------
        // Pick the LP backend by name (CSimplex default; HiGHS if built in),
        // then memoize solves by sequence (deterministic LP -> a cache hit
        // equals a re-solve). All local to this call, so the cache is private.
        std::unique_ptr<ScheduleEvaluator> backend = makeScheduleEvaluator(params_.evaluatorBackend);
        if (!backend) { result.status = SolveStatus::Failure; return result; }  // unknown backend
        CachingScheduleEvaluator evaluator(*backend);   // memoizes by sequence
        EvaluatorConfig evalCfg;                        // LP runtime knobs
        evalCfg.maxIterations = params_.lpMaxIterations;
        evalCfg.seed          = seed;                   // deterministic LP anti-cycling
        evalCfg.costLimit     = params_.costLimit;      // bi-criteria cost limit

        // ---- run the evolutionary loop (mirrors the legacy main()) -------
        ::Population pop;                    // owns old/new/best populations
        pop.dlsInstance       = &instance;   // problem fed to the evaluator
        pop.scheduleEvaluator = &evaluator;  // fitness via cached core evaluator
        pop.evaluatorConfig   = evalCfg;
        pop.rng               = &rng;        // injected RNG (no global rand)
        pop.GenerateNewPopulation(&procData, &gaPar);
        pop.EvaluatePopulation(&progPar, &procData, &gaPar);

        int generationsLeft = params_.maxGenerations;     // remaining generations
        int improvementLeft = params_.noImprovementLimit; // gens left without improvement
        while (generationsLeft && improvementLeft) {
            pop.NewToOldCopy();
            pop.Select(&procData, &gaPar);   // roulette or tournament per GAParams
            pop.CrossOver(&procData, &gaPar);
            pop.Mutation(&procData, &gaPar);
            pop.EvaluatePopulation(&progPar, &procData, &gaPar);

            if (pop.CheckImprovement())
                improvementLeft = params_.noImprovementLimit - 1;
            else
                improvementLeft--;
            generationsLeft--;
        }

        // ---- translate the best genotype into a DLSSolution --------------
        result.iterations = params_.maxGenerations - generationsLeft;
        extractSolution(instance, pop, result);
        return result;
    }

private:
    GAParams params_;  // immutable algorithm configuration

    // Goal:   copy GAParams into the legacy GA parameter struct.
    // Input:  gaPar - the legacy struct to populate.
    // Output: none (gaPar is filled in place).
    void fillGaParams(::GeneticAlgorithmParameters& gaPar) const {
        gaPar.maxIterNum           = params_.maxGenerations;
        gaPar.improvementNum       = params_.noImprovementLimit;
        gaPar.populationSize       = params_.populationSize;
        gaPar.chromosomeGeneration = 0;                       // constant-length chromosomes
        gaPar.chromosomeMinSize    = params_.installments;
        gaPar.chromosomeMaxSize    = params_.installments;
        gaPar.selectionMethod      = (params_.selection == Selection::Tournament) ? 1 : 0;
        gaPar.tourGroupSize        = params_.tournamentSize;
        gaPar.crossoverPoint       = (params_.crossover == Crossover::TwoPoint) ? 1 : 0;
        gaPar.crossoverProb        = params_.crossoverProb;
        gaPar.mutationMethod       = (params_.mutation  == Mutation::Swap)       ? 1 : 0;
        gaPar.mutationProb         = params_.mutationProb;
    }

    // Goal:   build the GA's 1-based processor table from a DLSInstance.
    // Input:  procData - the legacy struct to populate (takes ownership of
    //                    the allocated rows; its destructor frees them);
    //         instance - the source problem.
    // Output: none (procData is filled in place).
    static void fillProcessors(::ProcessorsParameters& procData, const DLSInstance& instance) {
        const int n = static_cast<int>(instance.numProcessors());  // worker count
        procData.volume    = static_cast<long>(instance.totalLoad());
        procData.procNum   = n;
        procData.allocRows = n + 1;   // row 0 is unused: the GA indexes processors 1..n
        procData.procParams = new double*[procData.allocRows];
        for (int r = 0; r < procData.allocRows; ++r) {
            procData.procParams[r] = new double[4];   // 4 columns: S, C, A, B
            for (int col = 0; col < 4; ++col) procData.procParams[r][col] = 0.0;
        }
        for (int k = 0; k < n; ++k) {
            const Processor& p = instance.processors()[k];
            procData.procParams[k + 1][PROC_COMM_STARTUP] = p.commStartup;
            procData.procParams[k + 1][PROC_COMM_RATE]    = p.commRate;
            procData.procParams[k + 1][PROC_COMPUTE_RATE] = p.computeRate;
            procData.procParams[k + 1][PROC_MEMORY_LIMIT] = p.memoryLimit;
        }
    }

    // Goal:   read the GA's best genotype into the uniform DLSSolution.
    // Input:  pop    - the finished population (bestCmax / bestSolution set);
    //         result - the solution to fill (status, makespan, fragments).
    // Output: none (result is filled in place).
    static void extractSolution(const DLSInstance& instanceForCost, ::Population& pop, DLSSolution& result) {
        if (!pop.haveBest) {            // no feasible solution was ever recorded
            result.status = SolveStatus::Failure;
            return;
        }
        result.status   = SolveStatus::Feasible;   // GA gives no optimality proof
        result.makespan = pop.bestCmax;
        double busTime = 0.0;                       // single-port bus: transfers are sequential
        for (const ::Gene& gene : pop.best.genes) {
            const int proc0 = gene.processorNum - 1;    // GA is 1-based; DLS is 0-based
            result.sequence.push_back(proc0);
            LoadFragment frag;                          // one installment of the schedule
            frag.processorId = proc0;
            frag.loadSize    = gene.dataSize;
            frag.commStart   = busTime;                 // earliest start on the shared bus
            const Processor& p = instanceForCost.processors()[proc0];
            busTime += p.commStartup + p.commRate * gene.dataSize;  // t_{i+1} = t_i + S + C·α_i
            result.fragments.push_back(frag);
        }
        result.cost   = scheduleCost(instanceForCost, result.fragments);
        result.energy = scheduleEnergy(instanceForCost, result.fragments, result.makespan);
    }
};

}  // namespace dls

#endif  // DLS_HEURISTICS_GA_SOLVER_HPP
