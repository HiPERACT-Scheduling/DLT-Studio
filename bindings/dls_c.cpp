//---------------------------------------------------------------------------
// bindings/dls_c.cpp
//
// A tiny C ABI over the DLS library, so any language with a C FFI can drive the
// solver portfolio. It is the dependency-free binding layer the Python package
// (python/dls/) loads via ctypes — no pybind11, no Python headers, no install.
//
// Contract: instances go IN as the library's text format (core/instance_io.hpp);
// results come OUT as the JSON contract (core/json_io.hpp). Strings returned by
// the solve/list calls are heap-allocated and must be released with dls_free().
//
// Options are a flat "key=value;key=value" string (so no JSON parser is needed):
//   solver         already passed separately
//   maxInstallments, nodeBudget, populationSize, maxGenerations,
//   noImprovementLimit, chunkDivisor                              (integers)
//   costLimit, deadline, epsilon, makespanLimit                  (doubles)
//   seed                                                         (uint64)
//   backend  = simplex|highs        allowRepeats = 0|1
//   chunk    = ssc|gss              psr = compute|comm|startup|memory|energy|super
//   warmStart    = 0|1  (exact: seed the incumbent from heuristics; default 1)
//   minimizeCost = 0|1  (exact-milp: minimize cost/energy G s.t. Cmax <= makespanLimit,
//                        instead of the default minimize Cmax s.t. G <= costLimit)
//---------------------------------------------------------------------------

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <vector>

#include "bench/benchmark.hpp"          // computePerformanceMap, runBenchmark
#include "cli/class_io.hpp"             // chain/tree/graph readers + topology solvers + readMlsdInstance
#include "mapreduce/mapreduce_solver.hpp"  // closed-form MapReduce (multilayer solver comes via class_io.hpp)
#include "mapreduce/reducer_read_instance.hpp"
#include "mapreduce/mapreduce_skew_instance.hpp"
#include "mapreduce/skew_static_solver.hpp"
#include "mapreduce/skew_dynamic_solver.hpp"
#include "mlsd/mlsd_solver.hpp"
#include "mlsd/mlsd_ga_solver.hpp"
#include "heuristics/ml/ml_mlsd_solver.hpp"
#include "mapreduce/multisource_instance.hpp"
#ifdef DLS_WITH_HIGHS
#include "exact/milp/mlsd_milp_solver.hpp"   // exact MLSD oracle (β = 0), HiGHS build only
#include "exact/branch_and_price/reducer_read_bp_solver.hpp"  // heterogeneous multi-channel reducer read scheduling
#include "mapreduce/mapreduce_bwidth_solver.hpp"               // bisection-width-limited MapReduce ("third method")
#include "mapreduce/multisource_solver.hpp"                     // multi-source map-phase scheduling LP
#endif
#include "core/bounds.hpp"
#include "core/instance_features.hpp"
#include "core/instance_io.hpp"
#include "core/json_io.hpp"
#include "core/pareto.hpp"               // timeEnergyFront
#include "core/schedule_expand.hpp"
#include "core/simplex_schedule_evaluator.hpp"
#include "core/solver_registry.hpp"
#include "heuristics/auto/difficulty_predictor.hpp"
#include "util/time.hpp"

using namespace dls;

namespace {

// Goal:   duplicate a std::string into a heap C string (freed by dls_free).
char* dup(const std::string& s) {
    char* p = static_cast<char*>(std::malloc(s.size() + 1));
    if (p) std::memcpy(p, s.data(), s.size() + 1);
    return p;
}

// A parsed "key=value;..." options string with typed lookups.
struct Opts {
    std::vector<std::pair<std::string, std::string>> kv;
    bool has(const std::string& k) const {
        for (const auto& p : kv) if (p.first == k) return true;
        return false;
    }
    std::string str(const std::string& k, const std::string& d = "") const {
        for (const auto& p : kv) if (p.first == k) return p.second;
        return d;
    }
    double dbl(const std::string& k, double d) const {
        try { return has(k) ? std::stod(str(k)) : d; } catch (...) { return d; }
    }
    int ival(const std::string& k, int d) const {
        try { return has(k) ? std::stoi(str(k)) : d; } catch (...) { return d; }
    }
};

Opts parseOptsT(const char* opts) {
    Opts o;
    if (!opts) return o;
    std::string s(opts), item;
    std::stringstream ss(s);
    while (std::getline(ss, item, ';')) {
        const auto eq = item.find('=');
        if (eq == std::string::npos) continue;
        o.kv.emplace_back(item.substr(0, eq), item.substr(eq + 1));
    }
    return o;
}

// Back-compat thin wrapper (dls_solve still uses the pair list directly).
std::vector<std::pair<std::string, std::string>> parseOpts(const char* opts) {
    return parseOptsT(opts).kv;
}

// Goal:   map a map-axis name to MapAxis (defaults to Processors on unknown).
bench::MapAxis axis(const std::string& s) {
    using A = bench::MapAxis;
    if (s == "load" || s == "V")     return A::Load;
    if (s == "comm" || s == "C")     return A::CommRate;
    if (s == "compute" || s == "A")  return A::ComputeRate;
    if (s == "startup" || s == "S")  return A::CommStartup;
    if (s == "pnet")                 return A::PowerNetwork;
    if (s == "pidle")                return A::PowerIdle;
    if (s == "k" || s == "energy")   return A::EnergySlope;
    return A::Processors;
}

// Goal:   build a minimal JSON error object {status, error}.
std::string errorJson(const std::string& msg) {
    return std::string("{\"solution\":{\"status\":\"Failure\",\"feasible\":false,\"error\":")
           + json::str(msg) + "}}";
}

}  // namespace

extern "C" {

// Goal:   JSON array of solver names available in this build.
// Output: heap string like ["auto","ga",...]; release with dls_free.
char* dls_solvers() {
    std::string out = "[";
    const auto names = availableSolvers();
    for (std::size_t i = 0; i < names.size(); ++i) out += (i ? "," : "") + json::str(names[i]);
    out += "]";
    return dup(out);
}

// Goal:   solve an instance (text format) with a named solver.
// Input:  instanceText - the instance; solverName - registry name; optsText -
//         "key=value;..." options (may be null/empty).
// Output: heap JSON {instance, lowerBound, solution} (with full Gantt timing),
//         or {solution:{status:Failure,error:...}} on a bad instance / solver.
//         Release with dls_free.
char* dls_solve(const char* instanceText, const char* solverName, const char* optsText) {
    DLSInstance inst;
    std::string err;
    std::istringstream in(instanceText ? instanceText : "");
    if (!readInstance(in, inst, err)) return dup(errorJson("parse: " + err));

    SolverOptions opt;
    std::uint64_t seed = 0;
    bool haveSeed = false;
    for (const auto& [k, v] : parseOpts(optsText)) {
        try {
            if      (k == "maxInstallments") opt.maxInstallments = std::stoi(v);
            else if (k == "nodeBudget")      opt.nodeBudget = std::stol(v);
            else if (k == "populationSize")  opt.populationSize = std::stoi(v);
            else if (k == "maxGenerations")  opt.maxGenerations = std::stoi(v);
            else if (k == "noImprovementLimit") opt.noImprovementLimit = std::stoi(v);
            else if (k == "chunkDivisor")    opt.chunkDivisor = std::stoi(v);
            else if (k == "warmStart")       opt.warmStart = (v != "0" && v != "false");
            else if (k == "costLimit")       opt.costLimit = std::stod(v);
            else if (k == "minimizeCost")    opt.minimizeCost = (v != "0" && v != "false");
            else if (k == "makespanLimit")   opt.makespanLimit = std::stod(v);
            else if (k == "deadline")        opt.deadline = std::stod(v);
            else if (k == "epsilon")         opt.epsilon = std::stod(v);
            else if (k == "autoEpsilon")     opt.autoEpsilon = (v != "0" && v != "false");
            else if (k == "seed")          { seed = std::stoull(v); haveSeed = true; }
            else if (k == "backend")         opt.evaluatorBackend = v;
            else if (k == "allowRepeats")    opt.allowRepeats = (v != "0");
            else if (k == "chunk")           opt.onlineChunk = (v == "ssc") ? ChunkRule::SSC : ChunkRule::GSS;
            else if (k == "psr")             opt.onlinePsr =
                v == "compute" ? Psr::ComputeRate : v == "comm" ? Psr::CommRate :
                v == "startup" ? Psr::CommStartup : v == "memory" ? Psr::MemoryDesc :
                v == "energy"  ? Psr::EnergySlope : Psr::All;
        } catch (...) { return dup(errorJson("bad option value for '" + k + "'")); }
    }
    if (opt.maxInstallments < static_cast<int>(inst.numProcessors()))
        opt.maxInstallments = static_cast<int>(inst.numProcessors());   // room for the exact search

    if (!inst.validate(&err)) return dup(errorJson("invalid instance: " + err));

    std::unique_ptr<DLSSolver> solver = makeSolver(solverName ? solverName : "", opt);
    if (!solver) return dup(errorJson(std::string("unknown/unavailable solver '") +
                                      (solverName ? solverName : "") + "'"));

    SolverConfig cfg;
    if (haveSeed) cfg.seed = seed;
    Time timer; timer.SetStartTime();
    DLSSolution sol = solver->solve(inst, cfg);
    timer.SetStopTime();
    sol.wallTimeSec = timer.CalcWallTime() / 1e6;
    expandSchedule(inst, sol);

    std::ostringstream out;
    out << "{\"solver\":" << json::str(solver->name());
    // Meta-solvers dispatch to a concrete method; report which one they chose.
    if (auto* meta = dynamic_cast<AutoSolver*>(solver.get()); meta && !meta->chosenSolver().empty())
        out << ",\"chosen\":" << json::str(meta->chosenSolver());
    if (auto* meta = dynamic_cast<AutoMlSolver*>(solver.get()); meta && !meta->chosenSolver().empty())
        out << ",\"chosen\":" << json::str(meta->chosenSolver());
    // FPTAS solvers: report the epsilon actually used (auto-derived or manual).
    if (auto* s = dynamic_cast<FptasOptVSolver*>(solver.get()))
        out << ",\"epsilon\":" << json::num(s->computedEpsilon());
    if (auto* s = dynamic_cast<FptasOptTSolver*>(solver.get()))
        out << ",\"epsilon\":" << json::num(s->computedEpsilon());
    // ML solver: report the ML-predicted makespan alongside the scheduled one.
    if (auto* s = dynamic_cast<MlSolver*>(solver.get()))
        out << ",\"predictedMakespan\":" << json::num(s->predictedMakespan());
    // ML energy solver: report the ML-predicted minimum energy alongside the
    // scheduled one (see ml_energy_solver.hpp — they are not expected to match
    // exactly, since Stage 2 still ultimately minimizes makespan).
    if (auto* s = dynamic_cast<EnergyMlSolver*>(solver.get()))
        out << ",\"predictedEnergy\":" << json::num(s->predictedEnergy());
    // Instance difficulty prediction (property of the instance, solver-independent).
    out << ",\"difficulty\":" << json::str(DifficultyPredictor::predict(computeFeatures(inst)));
    out << ",\"instance\":"; writeInstanceJson(out, inst);
    out << ",\"lowerBound\":" << json::num(divisibleLoadLowerBoundTight(inst))
        << ",\"solution\":"; writeSolutionJson(out, sol);
    out << "}";
    return dup(out.str());
}

// Goal:   solve an MLSD instance (text format) with a named MLSD solver.
// Input:  instanceText - task/proc format (see cli/class_io.hpp readMlsdInstance);
//         solverName   - "mlsd-exact" | "mlsd-ga";
//         optsText     - "key=value;..." options (maxInstallments, seed).
// Output: heap JSON {solver, status, makespan, taskOrder, loads:[[..],...],
// predictedMakespan (ml-mlsd only)} or an error object. Release with dls_free.
char* dls_mlsd_solve(const char* instanceText, const char* solverName, const char* optsText) {
    MlsdInstance inst;
    std::string err;
    std::istringstream in(instanceText ? instanceText : "");
    if (!readMlsdInstance(in, inst, err)) return dup(errorJson("parse: " + err));

    Opts o = parseOptsT(optsText);
    const std::string sname = solverName ? solverName : "mlsd-exact";

    auto emit = [&](const MlsdSolution& sol, double predictedMakespan, bool havePrediction) {
        std::ostringstream out;
        out << "{\"solver\":" << json::str(sname)
            << ",\"status\":" << json::str(sol.status == SolveStatus::Optimal ? "Optimal"
                                        : sol.status == SolveStatus::Feasible ? "Feasible"
                                        : "Failure")
            << ",\"makespan\":" << json::num(sol.makespan);
        if (havePrediction) out << ",\"predictedMakespan\":" << json::num(predictedMakespan);
        out << ",\"taskOrder\":[";
        for (std::size_t i = 0; i < sol.taskOrder.size(); ++i) out << (i ? "," : "") << sol.taskOrder[i];
        out << "],\"loads\":[";
        for (std::size_t l = 0; l < sol.loads.size(); ++l) {
            out << (l ? "," : "") << "[";
            for (std::size_t k = 0; k < sol.loads[l].size(); ++k)
                out << (k ? "," : "") << json::num(sol.loads[l][k]);
            out << "]";
        }
        out << "]}";
        return out.str();
    };

    if (sname == "mlsd-exact") {
        return dup(emit(MlsdSolver().solve(inst), 0.0, false));
    } else if (sname == "mlsd-ga") {
        return dup(emit(MlsdGaSolver(MlsdGaSolver::Params{}).solve(inst), 0.0, false));
    } else if (sname == "ml-mlsd") {
        MlMlsdSolver solver;
        MlsdSolution sol = solver.solve(inst);
        return dup(emit(sol, solver.predictedMakespan(), true));
#ifdef DLS_WITH_HIGHS
    } else if (sname == "mlsd-milp") {
        return dup(emit(MlsdMilpSolver(MlsdMilpParams{}).solve(inst), 0.0, false));
#endif
    }
    return dup(errorJson(std::string("unknown MLSD solver '") + sname + "'"));
}

// Goal:   time–energy Pareto front for an instance (GUI "trade-off explorer").
// Input:  instanceText; solverName picks the activation order (default best-rate);
//         optsText: points (default 12), maxInstallments, seed.
// Output: heap JSON {sequence, points:[{makespan,energy}]} from fast/high-energy
//         to slow/low-energy, or an error object. Release with dls_free.
char* dls_pareto(const char* instanceText, const char* solverName, const char* optsText) {
    DLSInstance inst;
    std::string err;
    std::istringstream in(instanceText ? instanceText : "");
    if (!readInstance(in, inst, err)) return dup(errorJson("parse: " + err));
    if (!inst.validate(&err))         return dup(errorJson("invalid instance: " + err));

    (void)solverName;                       // sequence is the full machine set (see below)
    Opts o = parseOptsT(optsText);
    const int points = o.ival("points", 12);

    // The front explores the trade-off across ALL machines, so the activation
    // order is every processor once (single round), repeated enough times to hold
    // the load when memory is limited (Σ B_i < V). A heuristic order could drop
    // machines (e.g. best-rate concentrates on one with ample memory), which would
    // flatten the trade-off — using the full set keeps it meaningful.
    const int N = static_cast<int>(inst.numProcessors());
    double sumB = 0.0;
    for (const Processor& p : inst.processors()) sumB += (p.memoryLimit > 0.0) ? p.memoryLimit : inst.totalLoad();
    int rounds = 1;
    if (sumB > 0.0) rounds = std::max(1, static_cast<int>(std::ceil(inst.totalLoad() / sumB)));
    rounds = std::min(rounds, 50);                            // safety cap
    std::vector<int> seq;
    for (int r = 0; r < rounds; ++r)
        for (int i = 0; i < N; ++i) seq.push_back(i);

    SimplexScheduleEvaluator ev;
    std::vector<ParetoPoint> front = timeEnergyFront(inst, ev, seq, points);

    std::ostringstream out;
    out << "{\"basis\":\"all-processors\",\"rounds\":" << rounds << ",\"sequence\":[";
    for (std::size_t i = 0; i < seq.size(); ++i) out << (i ? "," : "") << seq[i];
    out << "],\"points\":[";
    for (std::size_t i = 0; i < front.size(); ++i)
        out << (i ? "," : "") << "{\"makespan\":" << json::num(front[i].makespan)
            << ",\"energy\":" << json::num(front[i].energy) << "}";
    out << "]}";
    return dup(out.str());
}

// Goal:   isoefficiency / isoenergy map (GUI heatmap view).
// Input:  optsText: x, y (axis names: procs|load|comm|compute|startup|pnet|pidle|k),
//         xmin,xmax,xsteps,ymin,ymax,ysteps, metric (makespan|energy), solver, and
//         base params procs,load,S,C,A,memory,pidle,pstartup,pnet,k,origpnet,origpidle,seed.
// Output: heap JSON {xAxis,yAxis,metric,xs,ys,grid}. Release with dls_free.
char* dls_map(const char* optsText) {
    Opts o = parseOptsT(optsText);
    bench::MapConfig c;
    c.xAxis = axis(o.str("x", "procs"));   c.yAxis = axis(o.str("y", "comm"));
    c.xMin = o.dbl("xmin", 1);  c.xMax = o.dbl("xmax", 16);  c.xSteps = o.ival("xsteps", 16);
    c.yMin = o.dbl("ymin", 0.01); c.yMax = o.dbl("ymax", 1.0); c.ySteps = o.ival("ysteps", 16);
    c.metric = (o.str("metric") == "energy") ? bench::MapMetric::Energy : bench::MapMetric::Makespan;
    c.solver = o.str("solver", "best-rate");
    c.baseProcessors  = o.ival("procs", 4);
    c.baseLoad        = o.dbl("load", 100.0);
    c.baseCommStartup = o.dbl("S", 0.0);
    c.baseCommRate    = o.dbl("C", 0.1);
    c.baseComputeRate = o.dbl("A", 0.2);
    c.baseMemoryFactor = o.dbl("memory", 4.0);
    c.basePowerIdle    = o.dbl("pidle", 0.0);
    c.basePowerStartup = o.dbl("pstartup", 0.0);
    c.basePowerNetwork = o.dbl("pnet", 0.0);
    c.baseEnergySlope  = o.dbl("k", 0.0);
    c.originatorPowerNetwork = o.dbl("origpnet", 0.0);
    c.originatorPowerIdle    = o.dbl("origpidle", 0.0);
    c.seed = static_cast<std::uint64_t>(o.ival("seed", 1));

    if (c.baseLoad        <= 0.0) return dup("{\"error\":\"load must be positive\"}");
    if (c.baseComputeRate <= 0.0) return dup("{\"error\":\"A (compute rate) must be positive\"}");
    if (c.baseCommRate    <  0.0) return dup("{\"error\":\"C (comm rate) must be >= 0\"}");
    if (c.baseCommStartup <  0.0) return dup("{\"error\":\"S (comm startup) must be >= 0\"}");
    if (c.baseProcessors  <  1)   return dup("{\"error\":\"procs must be >= 1\"}");
    if (c.xSteps          <  1)   return dup("{\"error\":\"xsteps must be >= 1\"}");
    if (c.ySteps          <  1)   return dup("{\"error\":\"ysteps must be >= 1\"}");
    if (c.xSteps > 1 && c.xMin >= c.xMax) return dup("{\"error\":\"xmin must be < xmax\"}");
    if (c.ySteps > 1 && c.yMin >= c.yMax) return dup("{\"error\":\"ymin must be < ymax\"}");

    bench::PerformanceMap m = bench::computePerformanceMap(c);
    auto arr = [](std::ostringstream& s, const std::vector<double>& v) {
        s << "[";
        for (std::size_t i = 0; i < v.size(); ++i) s << (i ? "," : "") << json::num(v[i]);
        s << "]";
    };
    std::ostringstream out;
    out << "{\"xAxis\":" << json::str(bench::toString(c.xAxis))
        << ",\"yAxis\":" << json::str(bench::toString(c.yAxis))
        << ",\"metric\":" << json::str(bench::toString(c.metric)) << ",\"xs\":";
    arr(out, m.xs); out << ",\"ys\":"; arr(out, m.ys); out << ",\"grid\":[";
    for (std::size_t yi = 0; yi < m.grid.size(); ++yi) {
        if (yi) out << ",";
        out << "[";
        for (std::size_t xi = 0; xi < m.grid[yi].size(); ++xi) {
            const double v = m.grid[yi][xi];
            out << (xi ? "," : "") << (std::isinf(v) ? "null" : json::num(v));  // infeasible -> JSON null
        }
        out << "]";
    }
    out << "]}";
    return dup(out.str());
}

// Goal:   portfolio comparison over random instances (GUI "compare" view).
// Input:  optsText: solvers (comma list), procs, load, memory, instances, seed,
//         homogeneous (0|1), installments.
// Output: heap JSON {instances,provenOptimal,solvers:[{name,feasible,total,
//         avgRelGap,maxGapToLB,avgMakespan,avgTimeSec}]}. Release with dls_free.
char* dls_bench(const char* optsText) {
    Opts o = parseOptsT(optsText);
    bench::BenchConfig cfg;
    cfg.numProcessors = o.ival("procs", 4);
    cfg.totalLoad     = o.dbl("load", 100.0);
    cfg.memoryFactor  = o.dbl("memory", 2.0);
    cfg.numInstances  = o.ival("instances", 10);
    cfg.seed          = static_cast<std::uint64_t>(o.ival("seed", 1));
    cfg.homogeneous   = (o.str("homogeneous") == "1");
    cfg.options.allowRepeats = true;
    cfg.options.maxInstallments = std::max(o.ival("installments", cfg.numProcessors), cfg.numProcessors);

    std::vector<std::string> solvers;
    { std::string s = o.str("solvers", "single-round,best-rate,ga,exact"), t;
      std::stringstream ss(s); while (std::getline(ss, t, ',')) if (!t.empty()) solvers.push_back(t); }

    bench::BenchReport rep = bench::runBenchmark(solvers, cfg);
    std::ostringstream out;
    out << "{\"instances\":" << rep.instances << ",\"provenOptimal\":" << rep.provenOptimal
        << ",\"solvers\":[";
    for (std::size_t i = 0; i < rep.solvers.size(); ++i) {
        const auto& r = rep.solvers[i];
        if (i) out << ",";
        out << "{\"name\":" << json::str(r.name) << ",\"total\":" << r.total
            << ",\"feasible\":" << r.feasible
            << ",\"avgRelGap\":" << json::num(r.avgRelGap)
            << ",\"maxGapToLB\":" << json::num(r.maxGapToLB)
            << ",\"avgMakespan\":" << json::num(r.avgMakespan)
            << ",\"avgTimeSec\":" << json::num(r.avgTimeSec) << "}";
    }
    out << "]}";
    return dup(out.str());
}

// Goal:   solve a non-star problem class (chain/tree/graph topology, or the
//         closed-form MapReduce / multilayer pipelines) divisible load.
// Input:  klass = "chain"|"tree"|"graph"|"mapreduce"|"multilayer"; text = the
//         class's line-oriented instance (see cli/class_io.hpp readers); opts unused.
// Output: heap JSON {status, feasible, makespan, loads:[...], parent:[...] (graph
//         only)} or an error object. Release with dls_free.
char* dls_topology(const char* klassC, const char* text, const char* optsText) {
    (void)optsText;
    const std::string k = klassC ? klassC : "";
    std::istringstream in(text ? text : "");
    std::string err;
    std::ostringstream out;
    auto emit = [&](SolveStatus st, bool feas, double mk,
                    const std::vector<double>& loads, const std::vector<int>* parent) {
        out << "{\"status\":" << json::str(json::statusName(st))
            << ",\"feasible\":" << (feas ? "true" : "false")
            << ",\"makespan\":" << json::num(mk) << ",\"loads\":[";
        for (std::size_t i = 0; i < loads.size(); ++i) out << (i ? "," : "") << json::num(loads[i]);
        out << "]";
        if (parent) {
            out << ",\"parent\":[";
            for (std::size_t i = 0; i < parent->size(); ++i) out << (i ? "," : "") << (*parent)[i];
            out << "]";
        }
        out << "}";
    };
    if (k == "chain") {
        LinearChainInstance inst;
        if (!readChainInstance(in, inst, err)) return dup(errorJson("parse: " + err));
        if (!inst.validate(&err)) return dup(errorJson("invalid instance: " + err));
        ChainSolution s = LinearChainSolver().solve(inst);
        emit(s.status, s.feasible(), s.makespan, s.loads, nullptr);
    } else if (k == "tree") {
        TreeInstance inst;
        if (!readTreeInstance(in, inst, err)) return dup(errorJson("parse: " + err));
        if (!inst.validate(&err)) return dup(errorJson("invalid instance: " + err));
        TreeSolution s = TreeSolver().solve(inst);
        emit(s.status, s.feasible(), s.makespan, s.loads, nullptr);
    } else if (k == "graph") {
        GraphInstance inst;
        if (!readGraphInstance(in, inst, err)) return dup(errorJson("parse: " + err));
        if (!inst.validate(&err)) return dup(errorJson("invalid instance: " + err));
        GraphSolution s = GraphSolver().solve(inst);
        emit(s.status, s.feasible(), s.makespan, s.loads, &s.parent);
    } else if (k == "mapreduce") {
        MapReduceInstance inst;
        if (!readMapReduceInstance(in, inst, err)) return dup(errorJson("parse: " + err));
        MapReduceSolution s = MapReduceSolver().solve(inst);   // validates internally
        out << "{\"status\":" << json::str(json::statusName(s.status))
            << ",\"feasible\":" << (s.feasible() ? "true" : "false")
            << ",\"makespan\":" << json::num(s.makespan)
            << ",\"reducerTime\":" << json::num(s.reducerTime)
            << ",\"loads\":[";
        for (std::size_t i = 0; i < s.mapperLoads.size(); ++i) out << (i ? "," : "") << json::num(s.mapperLoads[i]);
        out << "],\"mapperOrder\":[";
        for (std::size_t i = 0; i < s.mapperOrder.size(); ++i) out << (i ? "," : "") << s.mapperOrder[i];
        out << "]}";
    } else if (k == "multilayer") {
        MultilayerInstance inst;
        if (!readMultilayerInstance(in, inst, err)) return dup(errorJson("parse: " + err));
        MultilayerSolution s = MultilayerSolver().solve(inst); // validates internally
        out << "{\"status\":" << json::str(json::statusName(s.status))
            << ",\"feasible\":" << (s.feasible() ? "true" : "false")
            << ",\"makespan\":" << json::num(s.makespan)
            << ",\"mapperTime\":" << json::num(s.mapperTime)
            << ",\"layerInput\":[";
        for (std::size_t i = 0; i < s.layerInput.size(); ++i) out << (i ? "," : "") << json::num(s.layerInput[i]);
        out << "],\"layerRead\":[";
        for (std::size_t i = 0; i < s.layerRead.size(); ++i) out << (i ? "," : "") << json::num(s.layerRead[i]);
        out << "],\"layerCompute\":[";
        for (std::size_t i = 0; i < s.layerCompute.size(); ++i) out << (i ? "," : "") << json::num(s.layerCompute[i]);
        out << "]}";
    } else {
        return dup(errorJson("unknown non-star class '" + k + "'"));
    }
    return dup(out.str());
}

// Goal:   solve a reducer partitioning-skew instance with the static
//         (fine-partitioning + LPT bin-packing) mitigation algorithm (see
//         mapreduce/skew_static_solver.hpp). Dependency-free.
// Input:  instanceText - the mapreduce-skew class instance text (V/mappers/
//         mapper_rate/readrate/gamma0/epsilon/bisection/reducers/sort_rate/
//         reduce_rate/master_rate/k, plus k*r "partition <size>" lines);
//         solverName, optsText unused.
// Output: heap JSON {status, makespan, masterTime, reducerLoads,
//         assignment:[[partitionIdx,...],...]} or an error object.
char* dls_skew_static_solve(const char* instanceText, const char* /*solverName*/,
                           const char* /*optsText*/) {
    MapReduceSkewInstance inst;
    std::string err;
    std::istringstream in(instanceText ? instanceText : "");
    if (!readMapReduceSkewInstance(in, inst, err)) return dup(errorJson("parse: " + err));
    if (!inst.validate(&err)) return dup(errorJson("invalid instance: " + err));

    StaticSkewSolution sol = StaticSkewSolver().solve(inst);

    std::ostringstream out;
    out << "{\"status\":" << json::str(json::statusName(sol.status))
        << ",\"feasible\":" << (sol.feasible() ? "true" : "false")
        << ",\"makespan\":" << json::num(sol.makespan)
        << ",\"masterTime\":" << json::num(sol.masterTime)
        << ",\"reducerLoads\":[";
    for (std::size_t j = 0; j < sol.reducerLoads.size(); ++j) out << (j ? "," : "") << json::num(sol.reducerLoads[j]);
    out << "],\"assignment\":[";
    for (std::size_t j = 0; j < sol.assignment.size(); ++j) {
        out << (j ? "," : "") << "[";
        for (std::size_t p = 0; p < sol.assignment[j].size(); ++p)
            out << (p ? "," : "") << sol.assignment[j][p];
        out << "]";
    }
    out << "]}";
    return dup(out.str());
}

// Goal:   solve a reducer partitioning-skew instance with the dynamic
//         (single-shot post-sort rebalancing) mitigation algorithm (see
//         mapreduce/skew_dynamic_solver.hpp). Dependency-free. Requires k=1
//         (the base r-way partition — see the solver's header for why).
// Input:  instanceText - same format as dls_skew_static_solve, but with
//         exactly r "partition <size>" lines (k must be 1 or omitted).
// Output: heap JSON {status, makespan, unmitigatedMakespan, triggerTime,
//         numBusyAtTrigger, sendersUsed} or an error object.
char* dls_skew_dynamic_solve(const char* instanceText, const char* /*solverName*/,
                            const char* /*optsText*/) {
    MapReduceSkewInstance inst;
    std::string err;
    std::istringstream in(instanceText ? instanceText : "");
    if (!readMapReduceSkewInstance(in, inst, err)) return dup(errorJson("parse: " + err));
    if (!inst.validate(&err)) return dup(errorJson("invalid instance: " + err));

    DynamicSkewSolution sol = DynamicSkewSolver().solve(inst);

    std::ostringstream out;
    out << "{\"status\":" << json::str(json::statusName(sol.status))
        << ",\"feasible\":" << (sol.feasible() ? "true" : "false")
        << ",\"makespan\":" << json::num(sol.makespan)
        << ",\"unmitigatedMakespan\":" << json::num(sol.unmitigatedMakespan)
        << ",\"triggerTime\":" << json::num(sol.triggerTime)
        << ",\"numBusyAtTrigger\":" << sol.numBusyAtTrigger
        << ",\"sendersUsed\":" << sol.sendersUsed
        << "}";
    return dup(out.str());
}

#ifdef DLS_WITH_HIGHS
// Goal:   solve a MapReduce instance under a bisection-width read-channel
//         limit (see mapreduce/mapreduce_bwidth_solver.hpp, the "third
//         method" LP). HiGHS build only.
// Input:  instanceText - the standard mapreduce class instance text (V/
//         startup/readrate/gamma0/reducers/reducer_startup/reducer_rate/
//         mapper, plus the optional "bisection <l>" directive); solverName,
//         optsText unused (one algorithm, no runtime options).
// Output: heap JSON {status, makespan, reducerTime, mapperOrder, loads} or
//         an error object. Release with dls_free.
char* dls_mapreduce_bwidth_solve(const char* instanceText, const char* /*solverName*/,
                                 const char* /*optsText*/) {
    MapReduceInstance inst;
    std::string err;
    std::istringstream in(instanceText ? instanceText : "");
    if (!readMapReduceInstance(in, inst, err)) return dup(errorJson("parse: " + err));
    if (!inst.validate(&err)) return dup(errorJson("invalid instance: " + err));

    MapReduceBwidthSolution sol = MapReduceBwidthSolver().solve(inst);

    std::ostringstream out;
    out << "{\"status\":" << json::str(json::statusName(sol.status))
        << ",\"feasible\":" << (sol.feasible() ? "true" : "false")
        << ",\"makespan\":" << json::num(sol.makespan)
        << ",\"reducerTime\":" << json::num(sol.reducerTime)
        << ",\"loads\":[";
    for (std::size_t i = 0; i < sol.mapperLoads.size(); ++i) out << (i ? "," : "") << json::num(sol.mapperLoads[i]);
    out << "],\"mapperOrder\":[";
    for (std::size_t i = 0; i < sol.mapperOrder.size(); ++i) out << (i ? "," : "") << sol.mapperOrder[i];
    out << "]}";
    return dup(out.str());
}
#endif

#ifdef DLS_WITH_HIGHS
// Goal:   solve a multi-source map-phase scheduling instance (see
//         mapreduce/multisource_solver.hpp). HiGHS build only.
// Input:  instanceText - the multisource class instance text (S/m/n/rate/
//         transfer, plus optional "storage <S_i>" lines selecting the
//         fixed-supply variant); solverName, optsText unused.
// Output: heap JSON {status, makespan, storageSizes, localLoads,
//         transfers:[{from,to,size},...], fixedSupply} or an error object.
char* dls_multisource_solve(const char* instanceText, const char* /*solverName*/,
                           const char* /*optsText*/) {
    MultiSourceInstance inst;
    std::string err;
    std::istringstream in(instanceText ? instanceText : "");
    if (!readMultiSourceInstance(in, inst, err)) return dup(errorJson("parse: " + err));
    if (!inst.validate(&err)) return dup(errorJson("invalid instance: " + err));

    MultiSourceSolution sol = MultiSourceSolver().solve(inst);

    std::ostringstream out;
    out << "{\"status\":" << json::str(json::statusName(sol.status))
        << ",\"feasible\":" << (sol.feasible() ? "true" : "false")
        << ",\"fixedSupply\":" << (inst.isFixedSupply() ? "true" : "false")
        << ",\"makespan\":" << json::num(sol.makespan)
        << ",\"storageSizes\":[";
    for (std::size_t i = 0; i < sol.storageSizes.size(); ++i) out << (i ? "," : "") << json::num(sol.storageSizes[i]);
    out << "],\"localLoads\":[";
    for (std::size_t i = 0; i < sol.localLoads.size(); ++i) out << (i ? "," : "") << json::num(sol.localLoads[i]);
    out << "],\"transfers\":[";
    bool first = true;
    for (std::size_t i = 0; i < sol.transferLoads.size(); ++i)
        for (std::size_t j = 0; j < sol.transferLoads[i].size(); ++j)
            if (i != j && sol.transferLoads[i][j] > 1e-9) {
                out << (first ? "" : ",") << "{\"from\":" << i << ",\"to\":" << j
                    << ",\"size\":" << json::num(sol.transferLoads[i][j]) << "}";
                first = false;
            }
    out << "]}";
    return dup(out.str());
}
#endif

#ifdef DLS_WITH_HIGHS
// Goal:   solve a heterogeneous multi-channel reducer read scheduling
//         instance (see mapreduce/reducer_read_instance.hpp) via branch-and-
//         price. HiGHS build only (the master LP needs HiGHS's dual values).
// Input:  instanceText - the class's line-oriented instance (capacity/
//         objective/mapper/reducer directives); solverName, optsText unused
//         (there is only one algorithm for this class, and no runtime
//         options — objective and affinity live in the instance itself).
// Output: heap JSON {status, objective, value, schedule:[{reducer, finish,
//         reads:[{mapper, start}]}]} or an error object. Release with
//         dls_free.
char* dls_reducer_read_solve(const char* instanceText, const char* /*solverName*/,
                             const char* /*optsText*/) {
    ReducerReadInstance inst;
    std::string err;
    std::istringstream in(instanceText ? instanceText : "");
    if (!readReducerReadInstance(in, inst, err)) return dup(errorJson("parse: " + err));
    if (!inst.validate(&err)) return dup(errorJson("invalid instance: " + err));

    ReducerReadSolution sol = ReducerReadBpSolver().solve(inst);

    std::ostringstream out;
    out << "{\"status\":" << json::str(json::statusName(sol.status))
        << ",\"objective\":" << json::str(inst.objective == ReducerReadObjective::Makespan
                                          ? "makespan" : "balance")
        << ",\"value\":" << json::num(sol.makespan)
        << ",\"nodesExplored\":" << sol.nodesExplored
        << ",\"schedule\":[";
    for (std::size_t j = 0; j < sol.schedule.size(); ++j) {
        const auto& col = sol.schedule[j];
        out << (j ? "," : "") << "{\"reducer\":" << col.reducer
            << ",\"finish\":" << json::num(col.finish) << ",\"reads\":[";
        auto reads = col.starts;
        std::sort(reads.begin(), reads.end(),
                 [](const auto& a, const auto& b) { return a.second < b.second; });
        for (std::size_t k = 0; k < reads.size(); ++k)
            out << (k ? "," : "") << "{\"mapper\":" << reads[k].first
                << ",\"start\":" << reads[k].second << "}";
        out << "]}";
    }
    out << "]}";
    return dup(out.str());
}
#endif

// Goal:   release a string returned by any dls_* call.
void dls_free(char* p) { std::free(p); }

}  // extern "C"
