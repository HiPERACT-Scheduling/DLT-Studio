//---------------------------------------------------------------------------
// bench/benchmark.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// A small benchmarking harness for the single-load DLS solver portfolio. It
// generates random instances in the style of the divisible-load literature
// (Berlińska thesis §3.6: A, C, S ~ Uniform[0,1]), runs each selected solver on
// the *same* instances through the registry, and aggregates two quantities the
// thesis uses to compare algorithms:
//
//   - solution quality: the average relative distance from the lower bound,
//     (Cmax − LB) / LB, where LB is divisibleLoadLowerBound (eq. 3.23);
//   - cost: the average wall-clock solve time.
//
// Together these reproduce the quality/time Pareto picture (Fig. 3.15): a good
// solver sits in the lower-left corner (small gap, small time). The harness is
// header-only (std::chrono timing) so it is exercised directly by the tests.
//---------------------------------------------------------------------------

#ifndef DLS_BENCH_BENCHMARK_HPP
#define DLS_BENCH_BENCHMARK_HPP

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <istream>
#include <limits>
#include <memory>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include <cmath>

#include "core/bounds.hpp"
#include "core/dls_instance.hpp"
#include "core/energy_model.hpp"   // scheduleEnergy for isoenergy maps
#include "core/solver_registry.hpp"
#include "heuristics/single_round/single_round_solver.hpp"   // closed-form regime-1 reference
#include "util/random.hpp"

namespace dls::bench {

// What to generate and how hard to search.
struct BenchConfig {
    int           numProcessors = 4;       // N
    double        totalLoad     = 100.0;   // V
    double        memoryFactor  = 2.0;     // Bᵢ = memoryFactor·V (>=1 ⇒ effectively unbounded)
    int           numInstances  = 10;      // instances averaged per solver
    std::uint64_t seed          = 1;       // reproducible generation + solving
    bool          homogeneous   = false;   // true ⇒ identical processors (canonical, regime-1
                                           // closed-form-tractable: the closed-form solver is optimal)
    SolverOptions options;                 // search depth / params handed to each solver
};

// Aggregated outcome for one solver over the instance set.
struct SolverBenchResult {
    std::string name;
    int    total        = 0;    // instances attempted
    int    feasible     = 0;    // instances solved feasibly
    double avgGapToBest = 0.0;  // mean (Cmax − best)/best vs the per-instance virtual best
    double avgGapToOpt  = 0.0;  // mean (Cmax − M*)/M* over instances with a PROVEN optimum (rigorous)
    int    provenScored = 0;    // # instances counted in avgGapToOpt (feasible here & proven)
    double avgRelGap    = 0.0;  // mean (Cmax − LB)/LB over feasible instances (LB = tight bound)
    double maxGapToLB   = 0.0;  // worst (Cmax − LB)/LB: a CERTIFIED bound — every solution is
                                // <= this far above the optimum, valid even where the optimum
                                // is unknown (since LB <= optimum ⇒ gap→LB >= true gap)
    double avgMakespan  = 0.0;  // mean Cmax over feasible instances
    double avgTimeSec   = 0.0;  // mean wall-clock solve time (all attempts)
};

// Whole-run summary: how trustworthy the reference (virtual best) is.
struct BenchReport {
    std::vector<SolverBenchResult> solvers;
    int instances     = 0;      // total instances
    int provenOptimal = 0;      // # instances whose reference is a PROVEN optimum
    int regime1       = 0;      // # closed-form-tractable instances (closed form IS the optimum)
    int regime2       = 0;      // # instances needing a search (proven only via an exact solver)
};

// Goal:   generate one random instance (A, C, S ~ Uniform[0,1]; Bᵢ = factor·V).
//         When cfg.homogeneous, all processors share one drawn (S, C, A) — a
//         canonical, closed-form-tractable instance (regime 1).
// Input:  rng - the generator (advanced in place); cfg - sizes/parameters.
// Output: a valid DLSInstance.
inline DLSInstance generateInstance(util::Rng& rng, const BenchConfig& cfg) {
    std::vector<Processor> ps;
    ps.reserve(cfg.numProcessors);
    // For the homogeneous case, draw the shared parameters once up front.
    const double hS = rng.uniform01();
    const double hC = rng.uniform01();
    const double hA = std::max(1e-3, rng.uniform01());
    for (int i = 0; i < cfg.numProcessors; ++i) {
        Processor p;
        if (cfg.homogeneous) {
            p.commStartup = hS; p.commRate = hC; p.computeRate = hA;
        } else {
            p.commStartup = rng.uniform01();
            p.commRate    = rng.uniform01();
            p.computeRate = std::max(1e-3, rng.uniform01());   // keep Aᵢ strictly positive
        }
        p.memoryLimit = cfg.memoryFactor * cfg.totalLoad;
        ps.push_back(p);
    }
    return DLSInstance(std::move(ps), cfg.totalLoad);
}

// Goal:   run each named solver over a fixed random instance set and aggregate,
//         tracking whether each instance's reference is a PROVEN optimum.
// Input:  solverNames - registry names to compare; cfg - the benchmark setup.
// Output: a BenchReport (per-solver results + how many references are proven).
inline BenchReport runBenchmark(const std::vector<std::string>& solverNames,
                                const BenchConfig& cfg) {
    const int S = static_cast<int>(solverNames.size());
    const int K = cfg.numInstances;
    const double kInf = std::numeric_limits<double>::infinity();

    // Pre-generate the instances (and their tight lower bounds) so every solver
    // is compared on the identical set.
    util::Rng rng(cfg.seed);
    std::vector<DLSInstance> insts;
    std::vector<double>      lbs;
    insts.reserve(K);
    for (int k = 0; k < K; ++k) {
        DLSInstance inst = generateInstance(rng, cfg);
        lbs.push_back(divisibleLoadLowerBoundTight(inst));
        insts.push_back(std::move(inst));
    }

    // Solve everything first; record makespan AND whether the solver PROVED
    // optimality on each instance (status == Optimal).
    std::vector<std::vector<double>> mk(S, std::vector<double>(K, kInf));   // makespan or +inf
    std::vector<std::vector<char>>   opt(S, std::vector<char>(K, 0));        // proved optimal?
    std::vector<double> totalTime(S, 0.0);
    std::vector<bool>   available(S, false);
    for (int s = 0; s < S; ++s) {
        if (!makeSolver(solverNames[s], cfg.options)) continue;   // unavailable in this build
        available[s] = true;
        for (int k = 0; k < K; ++k) {
            std::unique_ptr<DLSSolver> solver = makeSolver(solverNames[s], cfg.options);
            SolverConfig sc;
            sc.seed = cfg.seed + static_cast<std::uint64_t>(k);   // deterministic per instance
            const auto t0 = std::chrono::steady_clock::now();
            DLSSolution sol = solver->solve(insts[k], sc);
            const auto t1 = std::chrono::steady_clock::now();
            totalTime[s] += std::chrono::duration<double>(t1 - t0).count();
            if (sol.feasible()) mk[s][k] = sol.makespan;
            if (sol.status == SolveStatus::Optimal) opt[s][k] = 1;
        }
    }

    // Regime 1 = closed-form-tractable: a homogeneous, ample-memory instance,
    // where the single-round closed form IS the proven optimum (no search needed).
    // Everything else is Regime 2 (proven only via an exact search solver).
    const bool regime1 = cfg.homogeneous && cfg.memoryFactor >= 1.0;
    std::vector<double> cfRef(K, kInf);          // closed-form optimum on regime-1 instances
    if (regime1)
        for (int k = 0; k < K; ++k) {
            DLSSolution s = SingleRoundSolver().solve(insts[k], SolverConfig{});
            if (s.feasible()) cfRef[k] = s.makespan;
        }

    // Per-instance reference (= the proven optimum where available). On Regime 1
    // it is the cheap closed form; on Regime 2 it requires some solver to have
    // returned Optimal achieving the virtual best (its proof — only for its own
    // search space — then dominates all others here).
    std::vector<double> best(K, kInf);
    std::vector<char>   proven(K, 0);
    for (int k = 0; k < K; ++k) {
        for (int s = 0; s < S; ++s) best[k] = std::min(best[k], mk[s][k]);
        if (regime1 && cfRef[k] != kInf) {                 // closed form is the optimum
            best[k] = std::min(best[k], cfRef[k]);
            proven[k] = 1;
        } else if (best[k] != kInf) {
            for (int s = 0; s < S; ++s)
                if (opt[s][k] && mk[s][k] <= best[k] * (1.0 + 1e-9) + 1e-12) { proven[k] = 1; break; }
        }
    }

    BenchReport report;
    report.instances = K;
    for (int k = 0; k < K; ++k) report.provenOptimal += proven[k];
    report.regime1 = (regime1 ? K : 0);
    report.regime2 = K - report.regime1;

    for (int s = 0; s < S; ++s) {
        SolverBenchResult r;
        r.name = solverNames[s];
        if (!available[s]) { report.solvers.push_back(r); continue; }
        r.total      = K;
        r.avgTimeSec = totalTime[s] / K;

        double sumGapLB = 0.0, sumGapBest = 0.0, sumGapOpt = 0.0, sumMk = 0.0, maxLB = 0.0;
        for (int k = 0; k < K; ++k) {
            if (mk[s][k] == kInf) continue;                  // infeasible on this instance
            ++r.feasible;
            sumMk += mk[s][k];
            if (lbs[k]  > 0.0)   { double g = (mk[s][k] - lbs[k]) / lbs[k]; sumGapLB += g; maxLB = std::max(maxLB, g); }
            if (best[k] > 0.0 && best[k] != kInf) sumGapBest += (mk[s][k] - best[k]) / best[k];
            if (proven[k] && best[k] > 0.0) { sumGapOpt += (mk[s][k] - best[k]) / best[k]; ++r.provenScored; }
        }
        r.maxGapToLB = maxLB;
        if (r.feasible)     { r.avgMakespan = sumMk / r.feasible; r.avgRelGap = sumGapLB / r.feasible; r.avgGapToBest = sumGapBest / r.feasible; }
        if (r.provenScored)   r.avgGapToOpt = sumGapOpt / r.provenScored;
        report.solvers.push_back(r);
    }
    return report;
}

//===========================================================================
// Ground-truth labels + external-policy scoring (the "certify against the
// proven optimum, fall back to the lower-bound certificate" workflow).
//===========================================================================

// One labeled instance: the exact optimum (if proved) and the lower bound.
struct InstanceLabel {
    DLSInstance instance;
    double      optMakespan = 0.0;    // M* — the exact solver's makespan (best known if not proved)
    bool        proven      = false;  // true if the exact solver returned Optimal
    double      lowerBound  = 0.0;    // tight LB (a valid floor even when not proved)
};

// Goal:   generate instances and label each with an exact solve, for use as
//         ground truth when certifying an external (e.g. learned) partitioner.
// Input:  cfg - generation + search settings; exactSolver - registry name of the
//         exact solver to label with (default "exact").
// Output: one InstanceLabel per instance (same instances as runBenchmark(cfg)).
inline std::vector<InstanceLabel> generateLabeledInstances(const BenchConfig& cfg,
                                                           const std::string& exactSolver = "exact") {
    std::vector<InstanceLabel> labels;
    util::Rng rng(cfg.seed);
    for (int k = 0; k < cfg.numInstances; ++k) {
        InstanceLabel lab;
        lab.instance   = generateInstance(rng, cfg);
        lab.lowerBound = divisibleLoadLowerBoundTight(lab.instance);
        std::unique_ptr<DLSSolver> solver = makeSolver(exactSolver, cfg.options);
        if (solver) {
            SolverConfig sc; sc.seed = cfg.seed + static_cast<std::uint64_t>(k);
            DLSSolution s = solver->solve(lab.instance, sc);
            if (s.feasible()) lab.optMakespan = s.makespan;
            lab.proven = (s.status == SolveStatus::Optimal);
        }
        labels.push_back(std::move(lab));
    }
    return labels;
}

// ---- label CSV I/O (the on-disk ground-truth format) ----------------------

// Goal:   write labels as CSV (instance, opt_makespan, proven, lower_bound).
inline void writeLabels(std::ostream& out, const std::vector<InstanceLabel>& labels) {
    const auto oldPrec = out.precision(17);
    out << "instance,opt_makespan,proven,lower_bound\n";
    for (std::size_t k = 0; k < labels.size(); ++k)
        out << k << "," << labels[k].optMakespan << ","
            << (labels[k].proven ? 1 : 0) << "," << labels[k].lowerBound << "\n";
    out.precision(oldPrec);
}

// Goal:   read labels back from the CSV written by writeLabels. Lines whose
//         opt_makespan field is non-numeric (the header / comments) are skipped.
inline std::vector<InstanceLabel> readLabels(std::istream& in) {
    std::vector<InstanceLabel> labels;
    std::string line;
    while (std::getline(in, line)) {
        if (auto h = line.find('#'); h != std::string::npos) line.erase(h);
        std::vector<std::string> f;
        std::stringstream ss(line); std::string tok;
        while (std::getline(ss, tok, ',')) f.push_back(tok);
        if (f.size() < 4) continue;
        try {
            InstanceLabel lab;
            lab.optMakespan = std::stod(f[1]);
            lab.proven      = (std::stoi(f[2]) != 0);
            lab.lowerBound  = std::stod(f[3]);
            labels.push_back(lab);
        } catch (...) { /* header / malformed line: skip */ }
    }
    return labels;
}

// Goal:   read a policy's per-instance makespans (one per line; the last comma-
//         separated field is taken, so "5.3" and "0,5.3" both work). Non-numeric
//         lines (headers) are skipped.
inline std::vector<double> readPolicyMakespans(std::istream& in) {
    std::vector<double> v;
    std::string line;
    while (std::getline(in, line)) {
        if (auto h = line.find('#'); h != std::string::npos) line.erase(h);
        std::string last, tok;
        std::stringstream ss(line);
        while (std::getline(ss, tok, ',')) if (!tok.empty()) last = tok;
        if (last.empty()) continue;
        try { v.push_back(std::stod(last)); } catch (...) { /* header: skip */ }
    }
    return v;
}

// Aggregate quality of an external policy against the labels.
struct PolicyScore {
    int    scoredAgainstOpt = 0;     // # proven-optimum instances scored
    double avgGapToOpt      = 0.0;   // mean (M_policy − M*)/M* over proven instances (rigorous)
    double maxGapToOpt      = 0.0;   // worst such gap
    int    scoredAgainstLB  = 0;     // # instances scored against the lower bound
    double avgGapToLB       = 0.0;   // mean (M_policy − LB)/LB (a conservative certificate, valid
                                     // even where exact solving is intractable)
    double maxGapToLB       = 0.0;   // worst (M_policy − LB)/LB: the policy is CERTIFIED to be
                                     // <= this far above optimal on every instance (true gap <= this)
};

// Goal:   score a policy's makespans against the ground-truth labels.
// Input:  labels - from generateLabeledInstances; policyMakespans - the policy's
//         makespan per instance, in the same order (<= 0 marks "no solution").
// Output: a PolicyScore (optimality gap on proven instances, LB-gap everywhere).
inline PolicyScore scorePolicy(const std::vector<InstanceLabel>& labels,
                               const std::vector<double>& policyMakespans) {
    PolicyScore sc;
    const std::size_t n = std::min(labels.size(), policyMakespans.size());
    double sumOpt = 0.0, sumLB = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        const double m = policyMakespans[k];
        if (m <= 0.0) continue;                              // policy produced no schedule
        if (labels[k].lowerBound > 0.0) {
            const double g = (m - labels[k].lowerBound) / labels[k].lowerBound;
            sumLB += g; sc.maxGapToLB = std::max(sc.maxGapToLB, g); ++sc.scoredAgainstLB;
        }
        if (labels[k].proven && labels[k].optMakespan > 0.0) {
            const double g = (m - labels[k].optMakespan) / labels[k].optMakespan;
            sumOpt += g; sc.maxGapToOpt = std::max(sc.maxGapToOpt, g); ++sc.scoredAgainstOpt;
        }
    }
    if (sc.scoredAgainstOpt) sc.avgGapToOpt = sumOpt / sc.scoredAgainstOpt;
    if (sc.scoredAgainstLB)  sc.avgGapToLB  = sumLB  / sc.scoredAgainstLB;
    return sc;
}

//===========================================================================
// Isoefficiency / isoenergy maps (Marszałkowski 2020, §4 & §6.3).
//
// A 2-D sweep of two system parameters over a grid, recording a performance
// metric (makespan = time-performance, or total energy) at each point. Plotting
// contours of the resulting grid yields the thesis's isoefficiency (constant
// time) and isoenergy (constant energy) maps — visual aids for understanding how
// pairs of parameters trade off. Each grid point is a homogeneous instance built
// from a fixed base with the two swept axes overridden, solved by a chosen
// portfolio solver; the grid is emitted as CSV for an external contour plotter.
//===========================================================================

// A system parameter that can be placed on a map axis.
enum class MapAxis {
    Processors,    // N  : number of (identical) workers
    Load,          // V  : total divisible load
    CommRate,      // C  : communication time per unit load
    ComputeRate,   // A  : computation time per unit load (in-core)
    CommStartup,   // S  : per-transfer startup latency
    PowerNetwork,  // P^N: networking power (energy maps)
    PowerIdle,     // P^I: idle power (energy maps)
    EnergySlope    // k  : running energy per unit load (energy maps)
};

// What each grid cell measures.
enum class MapMetric { Makespan, Energy };

inline const char* toString(MapAxis a) {
    switch (a) {
        case MapAxis::Processors:   return "processors";
        case MapAxis::Load:         return "load";
        case MapAxis::CommRate:     return "comm-rate";
        case MapAxis::ComputeRate:  return "compute-rate";
        case MapAxis::CommStartup:  return "startup";
        case MapAxis::PowerNetwork: return "power-network";
        case MapAxis::PowerIdle:    return "power-idle";
        case MapAxis::EnergySlope:  return "energy-slope";
    }
    return "?";
}
inline const char* toString(MapMetric m) {
    return m == MapMetric::Energy ? "energy" : "makespan";
}

// Configuration of a performance-map sweep.
struct MapConfig {
    MapAxis   xAxis  = MapAxis::Processors;     // swept on the columns
    MapAxis   yAxis  = MapAxis::CommRate;       // swept on the rows
    double    xMin = 1.0,  xMax = 16.0;  int xSteps = 16;
    double    yMin = 0.01, yMax = 1.0;   int ySteps = 16;
    MapMetric metric = MapMetric::Makespan;
    std::string solver = "best-rate";           // registry solver scoring each point

    // Base (homogeneous) instance; the two axes override their parameters.
    int    baseProcessors  = 4;
    double baseLoad        = 100.0;
    double baseCommStartup = 0.0;
    double baseCommRate    = 0.1;
    double baseComputeRate = 0.2;
    double baseMemoryFactor = 4.0;              // Bᵢ = factor·V
    // Energy base (used by Energy maps / when an energy axis is swept).
    double basePowerIdle    = 0.0;
    double basePowerStartup = 0.0;
    double basePowerNetwork = 0.0;
    double baseEnergySlope  = 0.0;              // running energy k (J per unit load)
    double originatorPowerNetwork = 0.0;
    double originatorPowerIdle    = 0.0;

    std::uint64_t seed = 1;                     // forwarded to the solver (determinism)
    SolverOptions options;                      // search depth handed to the solver
};

// A computed 2-D performance map.
struct PerformanceMap {
    MapConfig config;
    std::vector<double> xs;                     // column tick values (xAxis)
    std::vector<double> ys;                     // row tick values (yAxis)
    std::vector<std::vector<double>> grid;      // grid[yi][xi] = metric (+inf if infeasible)
};

// Goal:   evenly spaced ticks in [mn, mx]; a single tick (=mn) when steps <= 1.
inline std::vector<double> axisTicks(double mn, double mx, int steps) {
    std::vector<double> v;
    if (steps <= 1) { v.push_back(mn); return v; }
    for (int i = 0; i < steps; ++i) v.push_back(mn + (mx - mn) * i / (steps - 1));
    return v;
}

// Goal:   build the homogeneous instance at one grid point (axes applied to base).
// Input:  c - the map config; xv,yv - the x/y axis values at this point.
// Output: a DLSInstance ready to solve.
inline DLSInstance buildMapInstance(const MapConfig& c, double xv, double yv) {
    double S = c.baseCommStartup, C = c.baseCommRate, A = c.baseComputeRate, V = c.baseLoad;
    double Pi = c.basePowerIdle, Ps = c.basePowerStartup, Pn = c.basePowerNetwork, K = c.baseEnergySlope;
    double Nf = static_cast<double>(c.baseProcessors);
    auto apply = [&](MapAxis ax, double v) {
        switch (ax) {
            case MapAxis::Processors:   Nf = v; break;
            case MapAxis::Load:         V  = v; break;
            case MapAxis::CommRate:     C  = v; break;
            case MapAxis::ComputeRate:  A  = v; break;
            case MapAxis::CommStartup:  S  = v; break;
            case MapAxis::PowerNetwork: Pn = v; break;
            case MapAxis::PowerIdle:    Pi = v; break;
            case MapAxis::EnergySlope:  K  = v; break;
        }
    };
    apply(c.xAxis, xv);
    apply(c.yAxis, yv);

    const int N = std::max(1, static_cast<int>(std::lround(Nf)));
    const bool energy = (K > 0.0 || Pi > 0.0 || Ps > 0.0 || Pn > 0.0);
    std::vector<Processor> ps;
    ps.reserve(N);
    for (int i = 0; i < N; ++i) {
        Processor p;
        p.commStartup = S; p.commRate = C; p.computeRate = std::max(1e-9, A);
        p.memoryLimit = c.baseMemoryFactor * V;
        p.powerIdle = Pi; p.powerStartup = Ps; p.powerNetwork = Pn;
        if (energy) p.energyPieces = {{0.0, K}};
        ps.push_back(p);
    }
    DLSInstance inst(std::move(ps), V);
    inst.setOriginatorPowerNetwork(c.originatorPowerNetwork);
    inst.setOriginatorPowerIdle(c.originatorPowerIdle);
    return inst;
}

// Goal:   compute the performance map by solving every grid point.
// Input:  c - the map configuration.
// Output: a PerformanceMap; infeasible / unavailable-solver cells are +inf.
inline PerformanceMap computePerformanceMap(const MapConfig& c) {
    PerformanceMap m;
    m.config = c;
    m.xs = axisTicks(c.xMin, c.xMax, c.xSteps);
    m.ys = axisTicks(c.yMin, c.yMax, c.ySteps);
    const double kInf = std::numeric_limits<double>::infinity();
    m.grid.assign(m.ys.size(), std::vector<double>(m.xs.size(), kInf));

    for (std::size_t yi = 0; yi < m.ys.size(); ++yi)
        for (std::size_t xi = 0; xi < m.xs.size(); ++xi) {
            DLSInstance inst = buildMapInstance(c, m.xs[xi], m.ys[yi]);
            std::unique_ptr<DLSSolver> solver = makeSolver(c.solver, c.options);
            if (!solver) continue;                       // solver unavailable in this build
            SolverConfig sc; sc.seed = c.seed;
            DLSSolution s = solver->solve(inst, sc);
            if (!s.feasible()) continue;
            m.grid[yi][xi] = (c.metric == MapMetric::Energy)
                                 ? scheduleEnergy(inst, s.fragments, s.makespan)
                                 : s.makespan;
        }
    return m;
}

// Goal:   write a performance map as CSV (rows = y axis, columns = x axis).
// Input:  out - the stream; m - the computed map. Infeasible cells print "inf".
inline void writePerformanceMap(std::ostream& out, const PerformanceMap& m) {
    const auto oldPrec = out.precision(10);
    out << "# isoline map: " << toString(m.config.metric) << " over "
        << toString(m.config.xAxis) << " (x) x " << toString(m.config.yAxis) << " (y)"
        << "  solver=" << m.config.solver << "\n";
    out << "# rows = y (" << toString(m.config.yAxis) << "), cols = x ("
        << toString(m.config.xAxis) << "); cell = " << toString(m.config.metric)
        << " (inf = infeasible)\n";
    // Header row: a corner label then the x tick values.
    out << "y\\x";
    for (double x : m.xs) out << "," << x;
    out << "\n";
    for (std::size_t yi = 0; yi < m.ys.size(); ++yi) {
        out << m.ys[yi];
        for (std::size_t xi = 0; xi < m.xs.size(); ++xi) {
            const double v = m.grid[yi][xi];
            if (v == std::numeric_limits<double>::infinity()) out << ",inf";
            else out << "," << v;
        }
        out << "\n";
    }
    out.precision(oldPrec);
}

}  // namespace dls::bench

#endif  // DLS_BENCH_BENCHMARK_HPP
