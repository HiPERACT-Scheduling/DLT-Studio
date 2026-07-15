//---------------------------------------------------------------------------
// cli/dls_main.cpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Uniform command-line front-end for the single-load DLS solver portfolio. It
// selects any registered solver by name (core/solver_registry.hpp) and runs it
// on an instance through the shared dls::DLSSolver contract.
//
// Usage:
//   dls list
//   dls solve --solver=NAME [options] INSTANCE_FILE
//   dls show INSTANCE_FILE                 (re-emit the parsed instance)
//
// Instance file (see core/instance_io.hpp): a "V <value>" line sets the load,
// an optional "beta <value>" line the result-return fraction, and each processor
// is a row "S C A B [p r d f l]" (4 required scalars plus optional compute
// startup / release / deadline / fixed & linear cost). A "pieces a,b a,b ..."
// line attaches convex compute pieces to the previous processor. '#' comments.
//
// Options (each used by the solvers it applies to):
//   --load=V            override the total load
//   --installments=N    search length / GA installments        (default 5)
//   --repeats=0|1       allow a processor to repeat (exact/optv) (default 1)
//   --backend=NAME      LP backend: simplex | highs            (default simplex)
//   --deadline=T        deadline for optv / fptas-optv         (required there)
//   --epsilon=E         approximation precision for the FPTAS  (default 0.1)
//   --nodes=N           exact node budget (0 = exhaustive)
//   --cost=G            bi-criteria cost limit (min Cmax s.t. G<=this)
//   --seed=S            RNG seed (GA reproducibility)
//---------------------------------------------------------------------------

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "core/bounds.hpp"
#include "core/instance_io.hpp"
#include "core/json_io.hpp"
#include "core/schedule_expand.hpp"
#include "core/solver_registry.hpp"
#include "heuristics/auto/auto_solver.hpp"
#include "cli/class_io.hpp"
#include "mlsd/mlsd_solver.hpp"
#include "mlsd/mlsd_ga_solver.hpp"
#include "heuristics/ml/ml_mlsd_solver.hpp"
#ifdef DLS_WITH_HIGHS
#include "exact/milp/mlsd_milp_solver.hpp"
#endif
#include "mapreduce/mapreduce_solver.hpp"
#include "mapreduce/multilayer_solver.hpp"
#include "mapreduce/reducer_read_instance.hpp"
#include "mapreduce/mapreduce_skew_instance.hpp"
#include "mapreduce/skew_static_solver.hpp"
#include "mapreduce/skew_dynamic_solver.hpp"
#include "mapreduce/multisource_instance.hpp"
#ifdef DLS_WITH_HIGHS
#include "exact/branch_and_price/reducer_read_bp_solver.hpp"
#include "mapreduce/mapreduce_bwidth_solver.hpp"
#include "mapreduce/multisource_solver.hpp"
#endif
#include "topology/linear_chain.hpp"
#include "topology/tree.hpp"
#include "topology/graph.hpp"
#include "util/time.hpp"

using namespace dls;

namespace {

// Parse "--key=value" flags and a single positional file. Returns false on a
// malformed flag, with `error` set.
struct Args {
    std::string subcommand;
    std::string solver;
    std::string problemClass = "dls";   // dls | mlsd | mapreduce | multilayer
    std::string file;
    SolverOptions opt;
    std::optional<double> loadOverride;
    std::optional<double> betaOverride;
    std::optional<std::uint64_t> seed;
    bool json = false;                  // emit machine-readable JSON (for a GUI/front-end)
};

bool parseDouble(const std::string& s, double& out) {
    try { std::size_t n; out = std::stod(s, &n); return n == s.size(); } catch (...) { return false; }
}

const char* toStr(SolveStatus s) {
    switch (s) {
        case SolveStatus::Optimal:    return "Optimal";
        case SolveStatus::Feasible:   return "Feasible";
        case SolveStatus::Infeasible: return "Infeasible";
        case SolveStatus::Unbounded:  return "Unbounded";
        case SolveStatus::Failure:    return "Failure";
        default:                      return "NotSolved";
    }
}

void printUsage(const char* prog) {
    std::cerr << "Usage:\n"
              << "  " << prog << " list\n"
              << "  " << prog << " solve [--class=dls|mlsd|mapreduce|mapreduce-bwidth|mapreduce-skew-static|mapreduce-skew-dynamic|multilayer|chain|tree|graph|reducer-read|multisource] [--solver=NAME] [options] FILE\n"
              << "  " << prog << " show INSTANCE_FILE   (single-load only)\n\n"
              << "Options: --class --solver --load --beta --installments --repeats --warm --backend"
              << " --deadline --epsilon --nodes --cost --seed --chunk --psr --json\n";
}

int doList() {
    std::cout << "Single-load solvers (--class=dls, default):\n";
    for (const std::string& name : availableSolvers()) {
        std::unique_ptr<DLSSolver> s = makeSolver(name);
        std::cout << "  --solver=" << name << "  (" << (s ? toString(s->category()) : "?") << ")\n";
    }
    std::cout << "Other problem classes:\n"
              << "  --class=mlsd        --solver=mlsd-exact (default) | mlsd-ga | mlsd-milp | ml-mlsd\n"
              << "  --class=mapreduce   (closed-form scheduler)\n"
              << "  --class=multilayer  (closed-form scheduler)\n"
              << "  --class=chain       (linear daisy chain, allocation LP)\n"
              << "  --class=tree        (multi-level tree, allocation LP)\n"
              << "  --class=graph       (general graph / mesh, best spanning arborescence)\n"
              << "  --class=mapreduce-skew-static  (reducer partitioning-skew mitigation,\n"
              << "                       fine partitioning + LPT bin-packing)\n"
              << "  --class=mapreduce-skew-dynamic (reducer partitioning-skew mitigation,\n"
              << "                       single-shot post-sort rebalancing)\n"
#ifdef DLS_WITH_HIGHS
              << "  --class=mapreduce-bwidth (MapReduce with a bisection-width read-channel\n"
              << "                       limit, \"third method\" LP; HiGHS builds only)\n"
              << "  --class=reducer-read (heterogeneous multi-channel MapReduce reducer read\n"
              << "                       scheduling, branch-and-price; HiGHS builds only)\n"
              << "  --class=multisource  (multi-source map-phase scheduling, bipartite\n"
              << "                       storage-nodes-to-mappers LP; HiGHS builds only)\n"
#endif
              ;
    return 0;
}

int doSolveSingleLoad(const Args& a) {
    if (a.solver.empty()) { std::cerr << "Error: --solver=NAME is required.\n"; return 1; }
    if (a.file.empty())   { std::cerr << "Error: an instance file is required.\n"; return 1; }

    std::unique_ptr<DLSSolver> solver = makeSolver(a.solver, a.opt);
    if (!solver) {
        std::cerr << "Error: unknown/unavailable solver '" << a.solver << "'.\n";
        std::cerr << "Available: "; for (auto& n : availableSolvers()) std::cerr << n << " "; std::cerr << "\n";
        return 1;
    }

    DLSInstance inst;
    std::string err;
    if (!readInstanceFile(a.file, inst, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    if (a.loadOverride) inst.setTotalLoad(*a.loadOverride);
    if (a.betaOverride) inst.setResultFraction(*a.betaOverride);
    if (!inst.validate(&err)) { std::cerr << "Error: invalid instance: " << err << "\n"; return 1; }

    SolverConfig cfg;
    if (a.seed) cfg.seed = *a.seed;

    Time timer; timer.SetStartTime();
    DLSSolution sol = solver->solve(inst, cfg);
    timer.SetStopTime();
    sol.wallTimeSec = timer.CalcWallTime() / 1e6;

    // Machine-readable output for a front-end: full schedule timing + JSON.
    if (a.json) {
        expandSchedule(inst, sol);
        std::cout << "{\"instance\":"; writeInstanceJson(std::cout, inst);
        std::cout << ",\"lowerBound\":" << json::num(divisibleLoadLowerBoundTight(inst));
        std::cout << ",\"solution\":"; writeSolutionJson(std::cout, sol);
        std::cout << "}\n";
        return sol.feasible() ? 0 : 1;
    }

    std::cout << "solver     : " << solver->name() << " (" << toString(solver->category()) << ")\n";
    if (auto* meta = dynamic_cast<AutoSolver*>(solver.get()))
        std::cout << "  chose    : " << meta->chosenSolver() << "\n";
    std::cout << "status     : " << toStr(sol.status) << "\n";
    if (sol.feasible()) {
        std::cout << "makespan   : " << sol.makespan << "\n";
        std::cout << "load (Σα)  : " << sol.totalAssignedLoad() << "\n";
        if (sol.cost != 0.0) std::cout << "cost (G)   : " << sol.cost << "\n";
        if (sol.energy != 0.0) std::cout << "energy (E) : " << sol.energy << "\n";
        std::cout << "lower bound: " << divisibleLoadLowerBoundTight(inst) << "\n";
        std::cout << "schedule   :\n";
        for (const LoadFragment& f : sol.fragments)
            std::cout << "  P" << f.processorId << "  load=" << f.loadSize << "\n";
    }
    std::cout << "time (s)   : " << timer.CalcWallTime() / 1e6 << "\n";
    return sol.feasible() ? 0 : 1;
}

// ---- MLSD ----------------------------------------------------------------
int doSolveMlsd(const Args& a) {
    if (a.file.empty()) { std::cerr << "Error: an instance file is required.\n"; return 1; }
    std::ifstream in(a.file);
    if (!in) { std::cerr << "Error: cannot open " << a.file << "\n"; return 1; }
    MlsdInstance inst; std::string err;
    if (!readMlsdInstance(in, inst, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    if (!inst.validate(&err)) { std::cerr << "Error: invalid instance: " << err << "\n"; return 1; }

    const std::string which = a.solver.empty() ? "mlsd-exact" : a.solver;
    Time timer; timer.SetStartTime();
    MlsdSolution sol;
    double predictedMakespan = 0.0;
    bool havePrediction = false;
    if (which == "mlsd-ga")       sol = MlsdGaSolver(MlsdGaSolver::Params{}).solve(inst, a.seed.value_or(0));
#ifdef DLS_WITH_HIGHS
    else if (which == "mlsd-milp") sol = MlsdMilpSolver().solve(inst);
#endif
    else if (which == "ml-mlsd") {
        MlMlsdSolver ml;
        sol = ml.solve(inst);
        predictedMakespan = ml.predictedMakespan();
        havePrediction = true;
    }
    else                          sol = MlsdSolver().solve(inst);
    timer.SetStopTime();

    const bool ok = sol.status == SolveStatus::Optimal || sol.status == SolveStatus::Feasible;
    std::cout << "class      : mlsd\nsolver     : " << which << "\n"
              << "status     : " << toStr(sol.status) << "\n";
    if (havePrediction)
        std::cout << "predicted  : " << predictedMakespan << "  (ML estimate, vs. GA-actual below)\n";
    if (ok) {
        std::cout << "makespan   : " << sol.makespan << "\ntask order : ";
        for (int t : sol.taskOrder) std::cout << t << " ";
        std::cout << "\nloads      :\n";
        for (std::size_t l = 0; l < sol.loads.size(); ++l) {
            std::cout << "  task " << (l < sol.taskOrder.size() ? sol.taskOrder[l] : (int)l) << ":";
            for (double v : sol.loads[l]) std::cout << " " << v;
            std::cout << "\n";
        }
    }
    std::cout << "time (s)   : " << timer.CalcWallTime() / 1e6 << "\n";
    return ok ? 0 : 1;
}

// ---- MapReduce -----------------------------------------------------------
int doSolveMapReduce(const Args& a) {
    if (a.file.empty()) { std::cerr << "Error: an instance file is required.\n"; return 1; }
    std::ifstream in(a.file);
    if (!in) { std::cerr << "Error: cannot open " << a.file << "\n"; return 1; }
    MapReduceInstance inst; std::string err;
    if (!readMapReduceInstance(in, inst, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    if (a.loadOverride) inst.setTotalLoad(*a.loadOverride);
    if (!inst.validate(&err)) { std::cerr << "Error: invalid instance: " << err << "\n"; return 1; }

    Time timer; timer.SetStartTime();
    MapReduceSolution sol = MapReduceSolver().solve(inst);
    timer.SetStopTime();

    std::cout << "class      : mapreduce\nsolver     : mapreduce\n"
              << "status     : " << toStr(sol.status) << "\n";
    if (sol.feasible()) {
        std::cout << "makespan   : " << sol.makespan << "\n"
                  << "reducerTime: " << sol.reducerTime << "\n"
                  << "mappers    : " << sol.mapperOrder.size() << " used\n"
                  << "loads      :\n";
        for (std::size_t k = 0; k < sol.mapperOrder.size(); ++k)
            std::cout << "  mapper " << sol.mapperOrder[k] << "  load=" << sol.mapperLoads[k] << "\n";
    }
    std::cout << "time (s)   : " << timer.CalcWallTime() / 1e6 << "\n";
    return sol.feasible() ? 0 : 1;
}

// ---- MapReduce, bisection-width-limited ("third method") -----------------
#ifdef DLS_WITH_HIGHS
int doSolveMapReduceBwidth(const Args& a) {
    if (a.file.empty()) { std::cerr << "Error: an instance file is required.\n"; return 1; }
    std::ifstream in(a.file);
    if (!in) { std::cerr << "Error: cannot open " << a.file << "\n"; return 1; }
    MapReduceInstance inst; std::string err;
    if (!readMapReduceInstance(in, inst, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    if (a.loadOverride) inst.setTotalLoad(*a.loadOverride);
    if (!inst.validate(&err)) { std::cerr << "Error: invalid instance: " << err << "\n"; return 1; }

    Time timer; timer.SetStartTime();
    MapReduceBwidthSolution sol = MapReduceBwidthSolver().solve(inst);
    timer.SetStopTime();

    std::cout << "class      : mapreduce-bwidth\nsolver     : mapreduce-bwidth (third method LP)\n"
              << "status     : " << toStr(sol.status) << "\n";
    if (sol.feasible()) {
        std::cout << "makespan   : " << sol.makespan << "\n"
                  << "reducerTime: " << sol.reducerTime << "\n"
                  << "mappers    : " << sol.mapperOrder.size() << " used\n"
                  << "loads      :\n";
        for (std::size_t k = 0; k < sol.mapperOrder.size(); ++k)
            std::cout << "  mapper " << sol.mapperOrder[k] << "  load=" << sol.mapperLoads[k] << "\n";
    }
    std::cout << "time (s)   : " << timer.CalcWallTime() / 1e6 << "\n";
    return sol.feasible() ? 0 : 1;
}
#endif

// ---- Multi-source map-phase scheduling ------------------------------------
#ifdef DLS_WITH_HIGHS
int doSolveMultiSource(const Args& a) {
    if (a.file.empty()) { std::cerr << "Error: an instance file is required.\n"; return 1; }
    std::ifstream in(a.file);
    if (!in) { std::cerr << "Error: cannot open " << a.file << "\n"; return 1; }
    MultiSourceInstance inst; std::string err;
    if (!readMultiSourceInstance(in, inst, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    if (a.loadOverride) inst.totalLoad = *a.loadOverride;
    if (!inst.validate(&err)) { std::cerr << "Error: invalid instance: " << err << "\n"; return 1; }

    Time timer; timer.SetStartTime();
    MultiSourceSolution sol = MultiSourceSolver().solve(inst);
    timer.SetStopTime();

    std::cout << "class      : multisource\nsolver     : multisource (LP, "
              << (inst.isFixedSupply() ? "Problem B: fixed supply" : "Problem A: free placement") << ")\n"
              << "status     : " << toStr(sol.status) << "\n";
    if (sol.feasible()) {
        std::cout << "makespan   : " << sol.makespan << "\n"
                  << "storage    :\n";
        for (std::size_t i = 0; i < sol.storageSizes.size(); ++i)
            std::cout << "  node " << i << "  S_i=" << sol.storageSizes[i]
                      << "  local=" << sol.localLoads[i] << "\n";
        std::cout << "transfers  :\n";
        for (std::size_t i = 0; i < sol.transferLoads.size(); ++i)
            for (std::size_t j = 0; j < sol.transferLoads[i].size(); ++j)
                if (i != j && sol.transferLoads[i][j] > 1e-9)
                    std::cout << "  " << i << " -> " << j << "  " << sol.transferLoads[i][j] << "\n";
    }
    std::cout << "time (s)   : " << timer.CalcWallTime() / 1e6 << "\n";
    return sol.feasible() ? 0 : 1;
}
#endif

// ---- Multilayer ----------------------------------------------------------
int doSolveMultilayer(const Args& a) {
    if (a.file.empty()) { std::cerr << "Error: an instance file is required.\n"; return 1; }
    std::ifstream in(a.file);
    if (!in) { std::cerr << "Error: cannot open " << a.file << "\n"; return 1; }
    MultilayerInstance inst; std::string err;
    if (!readMultilayerInstance(in, inst, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    if (a.loadOverride) inst.totalLoad = *a.loadOverride;

    Time timer; timer.SetStartTime();
    MultilayerSolution sol = MultilayerSolver().solve(inst);
    timer.SetStopTime();

    std::cout << "class      : multilayer\nsolver     : multilayer\n"
              << "status     : " << toStr(sol.status) << "\n";
    if (sol.feasible()) {
        std::cout << "makespan   : " << sol.makespan << "\nmapperTime : " << sol.mapperTime << "\n"
                  << "layers     :\n";
        for (std::size_t p = 0; p < sol.layerInput.size(); ++p)
            std::cout << "  layer " << (p + 1) << "  input=" << sol.layerInput[p]
                      << " read=" << sol.layerRead[p] << " compute=" << sol.layerCompute[p] << "\n";
    }
    std::cout << "time (s)   : " << timer.CalcWallTime() / 1e6 << "\n";
    return sol.feasible() ? 0 : 1;
}

// ---- Reducer partitioning-skew mitigation (MISTA 2013) --------------------
int doSolveSkewStatic(const Args& a) {
    if (a.file.empty()) { std::cerr << "Error: an instance file is required.\n"; return 1; }
    std::ifstream in(a.file);
    if (!in) { std::cerr << "Error: cannot open " << a.file << "\n"; return 1; }
    MapReduceSkewInstance inst; std::string err;
    if (!readMapReduceSkewInstance(in, inst, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    if (a.loadOverride) inst.totalLoad = *a.loadOverride;
    if (!inst.validate(&err)) { std::cerr << "Error: invalid instance: " << err << "\n"; return 1; }

    Time timer; timer.SetStartTime();
    StaticSkewSolution sol = StaticSkewSolver().solve(inst);
    timer.SetStopTime();

    std::cout << "class      : mapreduce-skew-static\nsolver     : fine-partitioning + LPT\n"
              << "status     : " << toStr(sol.status) << "\n";
    if (sol.feasible()) {
        std::cout << "makespan   : " << sol.makespan << "\nmasterTime : " << sol.masterTime << "\n"
                  << "reducers   :\n";
        for (std::size_t j = 0; j < sol.reducerLoads.size(); ++j)
            std::cout << "  reducer " << j << "  load=" << sol.reducerLoads[j]
                      << "  parts=" << sol.assignment[j].size() << "\n";
    }
    std::cout << "time (s)   : " << timer.CalcWallTime() / 1e6 << "\n";
    return sol.feasible() ? 0 : 1;
}

int doSolveSkewDynamic(const Args& a) {
    if (a.file.empty()) { std::cerr << "Error: an instance file is required.\n"; return 1; }
    std::ifstream in(a.file);
    if (!in) { std::cerr << "Error: cannot open " << a.file << "\n"; return 1; }
    MapReduceSkewInstance inst; std::string err;
    if (!readMapReduceSkewInstance(in, inst, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    if (a.loadOverride) inst.totalLoad = *a.loadOverride;
    if (!inst.validate(&err)) { std::cerr << "Error: invalid instance: " << err << "\n"; return 1; }

    Time timer; timer.SetStartTime();
    DynamicSkewSolution sol = DynamicSkewSolver().solve(inst);
    timer.SetStopTime();

    std::cout << "class      : mapreduce-skew-dynamic\nsolver     : single-shot rebalancing\n"
              << "status     : " << toStr(sol.status) << "\n";
    if (sol.feasible()) {
        std::cout << "makespan       : " << sol.makespan << "\n"
                  << "unmitigated    : " << sol.unmitigatedMakespan << "\n"
                  << "triggerTime    : " << sol.triggerTime << "\n"
                  << "busyAtTrigger  : " << sol.numBusyAtTrigger << "\n"
                  << "sendersRebalanced: " << sol.sendersUsed << "\n";
    }
    std::cout << "time (s)   : " << timer.CalcWallTime() / 1e6 << "\n";
    return sol.feasible() ? 0 : 1;
}

// ---- Reducer read scheduling (heterogeneous multi-channel MapReduce) ------
#ifdef DLS_WITH_HIGHS
int doSolveReducerRead(const Args& a) {
    if (a.file.empty()) { std::cerr << "Error: an instance file is required.\n"; return 1; }
    std::ifstream in(a.file);
    if (!in) { std::cerr << "Error: cannot open " << a.file << "\n"; return 1; }
    ReducerReadInstance inst; std::string err;
    if (!readReducerReadInstance(in, inst, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    if (!inst.validate(&err)) { std::cerr << "Error: invalid instance: " << err << "\n"; return 1; }

    Time timer; timer.SetStartTime();
    ReducerReadSolution sol = ReducerReadBpSolver().solve(inst);
    timer.SetStopTime();

    const bool ok = sol.status == SolveStatus::Optimal;
    std::cout << "class      : reducer-read\nsolver     : branch-and-price\n"
              << "status     : " << toStr(sol.status) << "\n";
    if (ok) {
        std::cout << "objective  : " << (inst.objective == ReducerReadObjective::Makespan
                                         ? "makespan" : "balance") << "\n"
                  << "value      : " << sol.makespan << "\n"
                  << "nodes      : " << sol.nodesExplored << "\n"
                  << "schedule   :\n";
        for (const auto& col : sol.schedule) {
            std::cout << "  reducer " << col.reducer << "  finish=" << col.finish << "  reads:";
            auto reads = col.starts;
            std::sort(reads.begin(), reads.end(),
                     [](auto& x, auto& y) { return x.second < y.second; });
            for (auto& [i, t] : reads) std::cout << " mapper" << i << "@" << t;
            std::cout << "\n";
        }
    }
    std::cout << "time (s)   : " << timer.CalcWallTime() / 1e6 << "\n";
    return ok ? 0 : 1;
}
#endif

// ---- topologies (chain / tree / general graph) ---------------------------
namespace {
template <typename ReadFn, typename Inst>
bool readTopologyFile(const std::string& file, ReadFn read, Inst& inst, std::string& err) {
    if (file.empty()) { err = "an instance file is required"; return false; }
    std::ifstream in(file);
    if (!in) { err = "cannot open " + file; return false; }
    return read(in, inst, err);
}
void printTopology(const char* cls, const char* solver, SolveStatus status,
                   bool ok, double makespan, const std::vector<double>& loads, double secs) {
    std::cout << "class      : " << cls << "\nsolver     : " << solver << "\n"
              << "status     : " << toStr(status) << "\n";
    if (ok) {
        std::cout << "makespan   : " << makespan << "\nloads      :\n";
        for (std::size_t i = 0; i < loads.size(); ++i)
            std::cout << "  node " << i << ": " << loads[i] << "\n";
    }
    std::cout << "time (s)   : " << secs << "\n";
}
}  // namespace

int doSolveChain(const Args& a) {
    LinearChainInstance inst; std::string err;
    if (!readTopologyFile(a.file, readChainInstance, inst, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    Time t; t.SetStartTime(); ChainSolution s = LinearChainSolver().solve(inst); t.SetStopTime();
    printTopology("chain", "linear-chain", s.status, s.feasible(), s.makespan, s.loads, t.CalcWallTime()/1e6);
    return s.feasible() ? 0 : 1;
}

int doSolveTree(const Args& a) {
    TreeInstance inst; std::string err;
    if (!readTopologyFile(a.file, readTreeInstance, inst, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    Time t; t.SetStartTime(); TreeSolution s = TreeSolver().solve(inst); t.SetStopTime();
    printTopology("tree", "tree", s.status, s.feasible(), s.makespan, s.loads, t.CalcWallTime()/1e6);
    return s.feasible() ? 0 : 1;
}

int doSolveGraph(const Args& a) {
    GraphInstance inst; std::string err;
    if (!readTopologyFile(a.file, readGraphInstance, inst, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    Time t; t.SetStartTime(); GraphSolution s = GraphSolver().solve(inst); t.SetStopTime();
    printTopology("graph", "graph", s.status, s.feasible(), s.makespan, s.loads, t.CalcWallTime()/1e6);
    if (s.feasible()) { std::cout << "arborescence:\n"; for (std::size_t i = 0; i < s.parent.size(); ++i) std::cout << "  node " << i << " <- parent " << s.parent[i] << "\n"; }
    return s.feasible() ? 0 : 1;
}

// Dispatch the solve subcommand to the requested problem class.
int doSolve(const Args& a) {
    if (a.problemClass == "dls")        return doSolveSingleLoad(a);
    if (a.problemClass == "mlsd")       return doSolveMlsd(a);
    if (a.problemClass == "mapreduce")  return doSolveMapReduce(a);
    if (a.problemClass == "multilayer") return doSolveMultilayer(a);
    if (a.problemClass == "chain")      return doSolveChain(a);
    if (a.problemClass == "tree")       return doSolveTree(a);
    if (a.problemClass == "graph")      return doSolveGraph(a);
    if (a.problemClass == "mapreduce-skew-static")  return doSolveSkewStatic(a);
    if (a.problemClass == "mapreduce-skew-dynamic") return doSolveSkewDynamic(a);
#ifdef DLS_WITH_HIGHS
    if (a.problemClass == "mapreduce-bwidth") return doSolveMapReduceBwidth(a);
    if (a.problemClass == "reducer-read")     return doSolveReducerRead(a);
    if (a.problemClass == "multisource")      return doSolveMultiSource(a);
#endif
    std::cerr << "Error: unknown --class '" << a.problemClass
              << "' (expected dls|mlsd|mapreduce|multilayer|chain|tree|graph"
              << "|mapreduce-skew-static|mapreduce-skew-dynamic"
#ifdef DLS_WITH_HIGHS
              << "|mapreduce-bwidth|reducer-read|multisource"
#endif
              << ")\n";
    return 1;
}

// Re-emit a parsed instance in canonical form (exercises the reader + writer).
int doShow(const Args& a) {
    if (a.file.empty()) { std::cerr << "Error: an instance file is required.\n"; return 1; }
    DLSInstance inst;
    std::string err;
    if (!readInstanceFile(a.file, inst, err)) { std::cerr << "Error: " << err << "\n"; return 1; }
    writeInstance(std::cout, inst);
    return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) { printUsage(argv[0]); return 1; }

    Args a;
    a.subcommand = argv[1];
    for (int i = 2; i < argc; ++i) {
        std::string s = argv[i];
        auto val = [&](const std::string& key) { return s.substr(key.size()); };
        if      (s.rfind("--solver=", 0) == 0)       a.solver = val("--solver=");
        else if (s.rfind("--class=", 0) == 0)        a.problemClass = val("--class=");
        else if (s.rfind("--backend=", 0) == 0)      a.opt.evaluatorBackend = val("--backend=");
        else if (s.rfind("--installments=", 0) == 0) a.opt.maxInstallments = std::atoi(val("--installments=").c_str());
        else if (s.rfind("--repeats=", 0) == 0)      a.opt.allowRepeats = std::atoi(val("--repeats=").c_str()) != 0;
        else if (s.rfind("--warm=", 0) == 0)         a.opt.warmStart = std::atoi(val("--warm=").c_str()) != 0;
        else if (s.rfind("--nodes=", 0) == 0)        a.opt.nodeBudget = std::atol(val("--nodes=").c_str());
        else if (s.rfind("--deadline=", 0) == 0)     { double d; if (parseDouble(val("--deadline="), d)) a.opt.deadline = d; }
        else if (s.rfind("--epsilon=", 0) == 0)      { double e; if (parseDouble(val("--epsilon="), e)) a.opt.epsilon = e; }
        else if (s.rfind("--cost=", 0) == 0)         { double c; if (parseDouble(val("--cost="), c)) a.opt.costLimit = c; }
        else if (s.rfind("--load=", 0) == 0)         { double v; if (parseDouble(val("--load="), v)) a.loadOverride = v; }
        else if (s.rfind("--beta=", 0) == 0)         { double b; if (parseDouble(val("--beta="), b)) a.betaOverride = b; }
        else if (s.rfind("--seed=", 0) == 0)         a.seed = static_cast<std::uint64_t>(std::strtoull(val("--seed=").c_str(), nullptr, 10));
        else if (s == "--json")                      a.json = true;
        else if (s.rfind("--chunk=", 0) == 0)        { std::string c = val("--chunk="); a.opt.onlineChunk = (c == "ssc" || c == "SSC") ? ChunkRule::SSC : ChunkRule::GSS; }
        else if (s.rfind("--psr=", 0) == 0)          { std::string r = val("--psr="); a.opt.onlinePsr =
                                                         r == "compute" ? Psr::ComputeRate : r == "comm" ? Psr::CommRate :
                                                         r == "startup" ? Psr::CommStartup : r == "memory" ? Psr::MemoryDesc :
                                                         r == "energy"  ? Psr::EnergySlope : Psr::All; }
        else if (s.rfind("--", 0) == 0)              { std::cerr << "Error: unknown option '" << s << "'\n"; return 1; }
        else                                          a.file = s;   // positional instance file
    }

    if (a.subcommand == "list")  return doList();
    if (a.subcommand == "solve") return doSolve(a);
    if (a.subcommand == "show")  return doShow(a);
    printUsage(argv[0]);
    return 1;
}
