//---------------------------------------------------------------------------
// core/solver_registry.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// A by-name registry for the single-load DLS solver portfolio. It ties the
// growing set of dls::DLSSolver implementations together behind one factory, so
// a CLI, benchmark harness, or test can select any solver at run time by its
// stable name() and drive them all through the uniform solve() contract.
//
// Algorithm-specific parameters are carried by a single SolverOptions struct;
// each factory reads the fields relevant to it and ignores the rest. This keeps
// the registry uniform without collapsing the solvers' distinct configurations.
//
// The available set depends on the build: the MILP solver is registered only in
// the HiGHS build (it links the HiGHS MILP backend).
//---------------------------------------------------------------------------

#ifndef DLS_CORE_SOLVER_REGISTRY_HPP
#define DLS_CORE_SOLVER_REGISTRY_HPP

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include "core/dls_solver.hpp"
#include "heuristics/ga/ga_solver.hpp"
#include "heuristics/best_rate/best_rate_solver.hpp"
#include "heuristics/online/online_solver.hpp"
#include "heuristics/single_round/single_round_solver.hpp"
#include "heuristics/auto/auto_solver.hpp"
#include "heuristics/auto/auto_ml_solver.hpp"
#include "heuristics/ml/ml_solver.hpp"
#include "heuristics/fptas/fptas_optv_solver.hpp"
#include "heuristics/fptas/fptas_optt_solver.hpp"
#include "exact/enumerative/exact_solver.hpp"
#include "exact/optv/optv_solver.hpp"
#include "exact/dual/dual_bisection_solver.hpp"
#ifdef DLS_WITH_HIGHS
#include "exact/milp/milp_solver.hpp"
#include "exact/milp/multi_milp_solver.hpp"
#endif

namespace dls {

// Union of the algorithm parameters the registered solvers accept. Defaults are
// sensible for a quick run; each factory uses only what applies to it.
struct SolverOptions {
    int    maxInstallments   = 5;       // GA installments / exact & optv search length
    bool   allowRepeats      = true;    // exact, optv: may a processor repeat?
    bool   warmStart         = true;    // exact: seed the incumbent from heuristics (false = original B&B)
    std::string evaluatorBackend = "simplex";   // LP backend: "simplex" | "highs"
    long   nodeBudget        = 0;       // exact: 0 = exhaustive
    int    chunkDivisor      = 1;       // best-rate: chunk = Bᵢ/x
    ChunkRule onlineChunk    = ChunkRule::GSS;  // online: SSC | GSS
    Psr       onlinePsr      = Psr::All;        // online: sorting rule (default super)
    double deadline          = std::numeric_limits<double>::infinity();  // optv: T (required)
    double epsilon           = 0.1;     // fptas: approximation precision
    bool   autoEpsilon       = false;   // fptas: derive ε from instance features
    double costLimit         = std::numeric_limits<double>::infinity();  // bi-criteria G-bar
    int    populationSize    = 12;      // GA
    int    maxGenerations    = 200;     // GA
    int    noImprovementLimit = 60;     // GA
};

// Goal:   list the solver names registered in this build.
// Output: stable names usable with makeSolver (MILP only in the HiGHS build).
inline std::vector<std::string> availableSolvers() {
    std::vector<std::string> v = {
        "auto", "auto-ml", "ml-makespan", "ga", "best-rate", "online", "single-round", "exact", "exact-dual", "optv", "fptas-optv", "fptas-optt"
    };
#ifdef DLS_WITH_HIGHS
    v.push_back("exact-milp");
    v.push_back("milp-multi");
#endif
    return v;
}

// Goal:   construct a solver by its stable name, configured from `opt`.
// Input:  name - one of availableSolvers(); opt - algorithm parameters.
// Output: an owning solver pointer, or nullptr if the name is unknown/unavailable.
inline std::unique_ptr<DLSSolver> makeSolver(const std::string& name,
                                             const SolverOptions& opt = {}) {
    if (name == "auto") {
        AutoParams p;
        p.maxInstallments  = opt.maxInstallments;
        p.evaluatorBackend = opt.evaluatorBackend;
        return std::make_unique<AutoSolver>(p);
    }
    if (name == "auto-ml") {
        AutoMlParams p;
        p.maxInstallments  = opt.maxInstallments;
        p.evaluatorBackend = opt.evaluatorBackend;
        return std::make_unique<AutoMlSolver>(p);
    }
    if (name == "ml-makespan") {
        MlSolverParams p;
        p.evaluatorBackend = opt.evaluatorBackend;
        p.maxInstallments  = opt.maxInstallments;
        return std::make_unique<MlSolver>(p);
    }
    if (name == "ga") {
        GAParams p;
        p.populationSize     = opt.populationSize;
        p.maxGenerations     = opt.maxGenerations;
        p.noImprovementLimit = opt.noImprovementLimit;
        p.installments       = opt.maxInstallments;
        p.costLimit          = opt.costLimit;
        p.evaluatorBackend   = opt.evaluatorBackend;
        return std::make_unique<GASolver>(p);
    }
    if (name == "best-rate") {
        BestRateParams p;
        p.chunkDivisor     = opt.chunkDivisor;
        p.costLimit        = opt.costLimit;
        p.evaluatorBackend = opt.evaluatorBackend;
        return std::make_unique<BestRateSolver>(p);
    }
    if (name == "online") {
        OnlineParams p;
        p.chunk = opt.onlineChunk;
        p.psr   = opt.onlinePsr;
        return std::make_unique<OnlineSolver>(p);
    }
    if (name == "single-round") return std::make_unique<SingleRoundSolver>();
    if (name == "exact") {
        ExactParams p;
        p.maxInstallments  = opt.maxInstallments;
        p.allowRepeats     = opt.allowRepeats;
        p.nodeBudget       = opt.nodeBudget;
        p.costLimit        = opt.costLimit;
        p.evaluatorBackend = opt.evaluatorBackend;
        p.warmStart        = opt.warmStart;
        return std::make_unique<ExactSolver>(p);
    }
    if (name == "exact-dual") {
        DualBisectionParams p;
        p.maxInstallments  = opt.maxInstallments;
        p.allowRepeats     = opt.allowRepeats;
        p.evaluatorBackend = opt.evaluatorBackend;
        return std::make_unique<DualBisectionSolver>(p);
    }
    if (name == "optv") {
        OptVParams p;
        p.deadline         = opt.deadline;
        p.maxInstallments  = opt.maxInstallments;
        p.allowRepeats     = opt.allowRepeats;
        p.evaluatorBackend = opt.evaluatorBackend;
        return std::make_unique<OptVSolver>(p);
    }
    if (name == "fptas-optv") {
        FptasOptVParams p;
        p.deadline    = opt.deadline;
        p.epsilon     = opt.epsilon;
        p.autoEpsilon = opt.autoEpsilon;
        return std::make_unique<FptasOptVSolver>(p);
    }
    if (name == "fptas-optt") {
        FptasOptTParams p;
        p.epsilon     = opt.epsilon;
        p.autoEpsilon = opt.autoEpsilon;
        return std::make_unique<FptasOptTSolver>(p);
    }
#ifdef DLS_WITH_HIGHS
    if (name == "exact-milp") {
        MilpParams p; p.maxInstallments = opt.maxInstallments; p.costLimit = opt.costLimit;
        return std::make_unique<MilpSolver>(p);
    }
    if (name == "milp-multi") {
        MultiMilpParams p; p.maxInstallments = opt.maxInstallments;
        return std::make_unique<MultiMilpSolver>(p);
    }
#endif
    return nullptr;   // unknown or unavailable in this build
}

}  // namespace dls

#endif  // DLS_CORE_SOLVER_REGISTRY_HPP
