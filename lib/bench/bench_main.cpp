//---------------------------------------------------------------------------
// bench/bench_main.cpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Command-line front-end for the benchmark harness (bench/benchmark.hpp). It
// generates random DLS instances and compares the makespan-minimizing solvers
// on solution quality (gap from the lower bound) and run time.
//
// Usage:
//   dls-bench [--procs=N] [--load=V] [--memory=F] [--instances=K] [--seed=S]
//             [--installments=L] [--homogeneous] [--solvers=a,b,c]
//   dls-bench labels [--exact=NAME] [generation options]   (emit ground-truth CSV)
//   dls-bench score  --labels FILE --policy FILE           (certify a policy's makespans)
//   dls-bench map --x=AXIS --y=AXIS --xrange=a:b:n --yrange=a:b:n [--metric=makespan|energy]
//             [--solver=NAME] [--C= --A= --S= --k= --pidle= --pstartup= --pnet= ...]
//             (emit an isoefficiency/isoenergy grid as CSV; axes: procs|load|comm|
//              compute|startup|pnet|pidle|k)
//
// Defaults compare single-round, best-rate, ga, and exact (plus exact-milp in
// the HiGHS build). Output is one row per solver, sorted by average gap.
//---------------------------------------------------------------------------

#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bench/benchmark.hpp"

using namespace dls;
using namespace dls::bench;

namespace {

std::vector<std::string> split(const std::string& s, char sep) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, sep)) if (!tok.empty()) out.push_back(tok);
    return out;
}

// Goal:   parse a map-axis name into a MapAxis.
// Input:  s - the axis name; ok - set false on an unknown name.
// Output: the MapAxis (Processors on error).
MapAxis parseAxis(const std::string& s, bool& ok) {
    ok = true;
    if (s == "procs" || s == "processors") return MapAxis::Processors;
    if (s == "load"  || s == "V")          return MapAxis::Load;
    if (s == "comm"  || s == "C")          return MapAxis::CommRate;
    if (s == "compute" || s == "A")        return MapAxis::ComputeRate;
    if (s == "startup" || s == "S")        return MapAxis::CommStartup;
    if (s == "pnet")                       return MapAxis::PowerNetwork;
    if (s == "pidle")                      return MapAxis::PowerIdle;
    if (s == "k" || s == "energy")         return MapAxis::EnergySlope;
    ok = false;
    return MapAxis::Processors;
}

// Goal:   parse a "min:max:steps" range; returns false on a malformed spec.
bool parseRange(const std::string& s, double& mn, double& mx, int& steps) {
    std::vector<std::string> f = split(s, ':');
    if (f.size() != 3) return false;
    try { mn = std::stod(f[0]); mx = std::stod(f[1]); steps = std::stoi(f[2]); }
    catch (...) { return false; }
    return steps >= 1;
}

}  // namespace

int main(int argc, char* argv[]) {
    BenchConfig cfg;
    cfg.options.allowRepeats = true;
    std::vector<std::string> solvers = {"single-round", "best-rate", "ga", "exact"};
#ifdef DLS_WITH_HIGHS
    solvers.push_back("exact-milp");
#endif
    // Optional leading subcommand: "labels" exports ground truth; "score"
    // certifies an external policy against a labels file.
    std::string mode = "bench";   // bench | labels | score | map
    int firstArg = 1;
    if (argc >= 2 && (std::string(argv[1]) == "labels" || std::string(argv[1]) == "score"
                      || std::string(argv[1]) == "map")) {
        mode = argv[1]; firstArg = 2;
    }
    MapConfig mapCfg;             // populated only in map mode
    std::string exactSolver = "exact";   // solver used to label the optimum
    std::string labelsFile, policyFile;  // for score mode

    for (int i = firstArg; i < argc; ++i) {
        std::string s = argv[i];
        auto val = [&](const std::string& key) { return s.substr(key.size()); };
        // --labels / --policy accept either "=FILE" or a following positional arg.
        auto pathArg = [&](const std::string& key, std::string& dst) -> bool {
            if (s == key) { if (i + 1 < argc) dst = argv[++i]; return true; }
            if (s.rfind(key + "=", 0) == 0) { dst = s.substr(key.size() + 1); return true; }
            return false;
        };
        if      (s.rfind("--exact=", 0) == 0)        exactSolver = val("--exact=");
        else if (pathArg("--labels", labelsFile))    {}
        else if (pathArg("--policy", policyFile))    {}
        else if (s.rfind("--procs=", 0) == 0)        cfg.numProcessors = std::atoi(val("--procs=").c_str());
        else if (s.rfind("--load=", 0) == 0)         cfg.totalLoad = std::atof(val("--load=").c_str());
        else if (s.rfind("--memory=", 0) == 0)       cfg.memoryFactor = std::atof(val("--memory=").c_str());
        else if (s.rfind("--instances=", 0) == 0)    cfg.numInstances = std::atoi(val("--instances=").c_str());
        else if (s.rfind("--seed=", 0) == 0)         cfg.seed = std::strtoull(val("--seed=").c_str(), nullptr, 10);
        else if (s.rfind("--installments=", 0) == 0) cfg.options.maxInstallments = std::atoi(val("--installments=").c_str());
        else if (s == "--homogeneous")               cfg.homogeneous = true;
        else if (s.rfind("--solvers=", 0) == 0)      solvers = split(val("--solvers="), ',');
        // ---- map-mode options ----
        else if (s.rfind("--x=", 0) == 0)            { bool ok; mapCfg.xAxis = parseAxis(val("--x="), ok); if (!ok) { std::cerr << "Unknown axis: " << val("--x=") << "\n"; return 1; } }
        else if (s.rfind("--y=", 0) == 0)            { bool ok; mapCfg.yAxis = parseAxis(val("--y="), ok); if (!ok) { std::cerr << "Unknown axis: " << val("--y=") << "\n"; return 1; } }
        else if (s.rfind("--xrange=", 0) == 0)       { if (!parseRange(val("--xrange="), mapCfg.xMin, mapCfg.xMax, mapCfg.xSteps)) { std::cerr << "Bad --xrange (want min:max:steps)\n"; return 1; } }
        else if (s.rfind("--yrange=", 0) == 0)       { if (!parseRange(val("--yrange="), mapCfg.yMin, mapCfg.yMax, mapCfg.ySteps)) { std::cerr << "Bad --yrange (want min:max:steps)\n"; return 1; } }
        else if (s.rfind("--metric=", 0) == 0)       mapCfg.metric = (val("--metric=") == "energy") ? MapMetric::Energy : MapMetric::Makespan;
        else if (s.rfind("--solver=", 0) == 0)       mapCfg.solver = val("--solver=");
        else if (s.rfind("--S=", 0) == 0)            mapCfg.baseCommStartup = std::atof(val("--S=").c_str());
        else if (s.rfind("--C=", 0) == 0)            mapCfg.baseCommRate    = std::atof(val("--C=").c_str());
        else if (s.rfind("--A=", 0) == 0)            mapCfg.baseComputeRate = std::atof(val("--A=").c_str());
        else if (s.rfind("--pidle=", 0) == 0)        mapCfg.basePowerIdle    = std::atof(val("--pidle=").c_str());
        else if (s.rfind("--pstartup=", 0) == 0)     mapCfg.basePowerStartup = std::atof(val("--pstartup=").c_str());
        else if (s.rfind("--pnet=", 0) == 0)         mapCfg.basePowerNetwork = std::atof(val("--pnet=").c_str());
        else if (s.rfind("--k=", 0) == 0)            mapCfg.baseEnergySlope  = std::atof(val("--k=").c_str());
        else if (s.rfind("--orig-pnet=", 0) == 0)    mapCfg.originatorPowerNetwork = std::atof(val("--orig-pnet=").c_str());
        else if (s.rfind("--orig-pidle=", 0) == 0)   mapCfg.originatorPowerIdle    = std::atof(val("--orig-pidle=").c_str());
        else { std::cerr << "Unknown option: " << s << "\n"; return 1; }
    }
    if (cfg.options.maxInstallments < cfg.numProcessors)
        cfg.options.maxInstallments = cfg.numProcessors;   // give the exact search room

    // ---- score mode: certify a policy's makespans against a labels file -----
    if (mode == "score") {
        if (labelsFile.empty() || policyFile.empty()) {
            std::cerr << "Usage: " << argv[0] << " score --labels FILE --policy FILE\n"; return 1;
        }
        std::ifstream lin(labelsFile), pin(policyFile);
        if (!lin) { std::cerr << "Error: cannot open " << labelsFile << "\n"; return 1; }
        if (!pin) { std::cerr << "Error: cannot open " << policyFile << "\n"; return 1; }
        std::vector<InstanceLabel> labels = readLabels(lin);
        std::vector<double>        policy = readPolicyMakespans(pin);
        if (labels.empty()) { std::cerr << "Error: no labels read from " << labelsFile << "\n"; return 1; }
        if (policy.size() < labels.size())
            std::cerr << "Warning: " << policy.size() << " policy values for " << labels.size()
                      << " labels; scoring the first " << policy.size() << ".\n";
        PolicyScore sc = scorePolicy(labels, policy);
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "Policy certification (" << policyFile << " vs " << labelsFile << "):\n";
        std::cout << "  proven-optimum instances : " << sc.scoredAgainstOpt << "\n";
        std::cout << "    avg gap to optimum     : " << (sc.avgGapToOpt * 100.0) << "%\n";
        std::cout << "    max gap to optimum     : " << (sc.maxGapToOpt * 100.0) << "%\n";
        std::cout << "  certified (vs lower bound, all " << sc.scoredAgainstLB << " instances):\n";
        std::cout << "    avg   certified <=     : " << (sc.avgGapToLB * 100.0) << "% above optimal\n";
        std::cout << "    worst certified <=     : " << (sc.maxGapToLB * 100.0) << "% above optimal\n";
        std::cout << "  (the certificate is a valid upper bound on the true optimality gap —\n"
                  << "   Cmax <= (1 + gap)·optimum — so it holds even where exact solving is intractable.)\n";
        return 0;
    }

    // ---- map mode: emit an isoefficiency / isoenergy grid as CSV ------------
    if (mode == "map") {
        // Inherit shared generation params (procs/load/memory/seed/installments).
        mapCfg.baseProcessors = cfg.numProcessors;
        mapCfg.baseLoad       = cfg.totalLoad;
        mapCfg.baseMemoryFactor = cfg.memoryFactor;
        mapCfg.seed           = cfg.seed;
        mapCfg.options        = cfg.options;
        PerformanceMap m = computePerformanceMap(mapCfg);
        writePerformanceMap(std::cout, m);
        return 0;
    }

    // ---- labels mode: emit ground-truth (instance -> M*, proven, LB) --------
    if (mode == "labels") {
        std::vector<InstanceLabel> labels = generateLabeledInstances(cfg, exactSolver);
        std::cout << "# labels via '" << exactSolver << "'  (N=" << cfg.numProcessors
                  << " V=" << cfg.totalLoad << " memory=" << cfg.memoryFactor
                  << "·V instances=" << cfg.numInstances << " seed=" << cfg.seed << ")\n";
        writeLabels(std::cout, labels);
        return 0;
    }

    std::cout << "Benchmark: N=" << cfg.numProcessors << " V=" << cfg.totalLoad
              << " memory=" << cfg.memoryFactor << "·V  instances=" << cfg.numInstances
              << " seed=" << cfg.seed
              << (cfg.homogeneous ? "  (homogeneous / closed-form-tractable)" : "  (A,C,S ~ U[0,1])")
              << "\n\n";

    BenchReport report = runBenchmark(solvers, cfg);
    std::sort(report.solvers.begin(), report.solvers.end(),
              [](const SolverBenchResult& a, const SolverBenchResult& b) {
        if (a.total == 0) return false;        // unavailable solvers sink to the bottom
        if (b.total == 0) return true;
        return a.avgGapToBest < b.avgGapToBest;   // rank by distance from the best found
    });

    // "gap→opt" is the rigorous distance from the PROVEN optimum (only over the
    // proven instances); "gap→best" is vs the per-instance virtual best (= opt
    // when proven); "gap→LB" is the certificate floor vs the tight lower bound.
    std::cout << "regime 1 (closed-form optimum): " << report.regime1
              << "   regime 2 (search): " << report.regime2 << "\n";
    std::cout << report.provenOptimal << "/" << report.instances
              << " instances have a proven-optimal reference"
              << (report.regime1 ? " (via the closed form on regime 1)" : "") << ".\n\n";
    std::cout << std::left << std::setw(15) << "solver"
              << std::right << std::setw(10) << "feasible"
              << std::setw(11) << "gap->opt%"
              << std::setw(11) << "gap->best%"
              << std::setw(10) << "gap->LB%"
              << std::setw(13) << "avg time ms"
              << std::setw(14) << "avg makespan" << "\n";
    std::cout << std::string(84, '-') << "\n";
    std::cout << std::fixed;
    for (const SolverBenchResult& r : report.solvers) {
        std::cout << std::left << std::setw(15) << r.name << std::right;
        if (r.total == 0) { std::cout << std::setw(10) << "n/a" << "\n"; continue; }
        std::ostringstream feas; feas << r.feasible << "/" << r.total;
        std::ostringstream gOpt;
        if (r.provenScored) gOpt << std::fixed << std::setprecision(2) << (r.avgGapToOpt * 100.0) << "%";
        else                gOpt << "-";
        std::cout << std::setw(10) << feas.str()
                  << std::setw(11) << gOpt.str()
                  << std::setw(10) << std::setprecision(2) << (r.avgGapToBest * 100.0) << "%"
                  << std::setw(9)  << std::setprecision(1) << (r.avgRelGap * 100.0) << "%"
                  << std::setw(13) << std::setprecision(3) << (r.avgTimeSec * 1000.0)
                  << std::setw(14) << std::setprecision(2) << r.avgMakespan << "\n";
    }
    std::cout << "\ngap->opt = distance from the PROVEN optimum (where available).\n"
              << "gap->LB  = a CERTIFIED upper bound on the optimality gap: each solution is\n"
              << "           <= gap->LB above optimal (Cmax <= (1+gap->LB)·optimum), valid even\n"
              << "           where the optimum is unknown. Worst-case certified bound per solver:\n";
    for (const SolverBenchResult& r : report.solvers)
        if (r.total) std::cout << "  " << std::left << std::setw(13) << r.name << std::right
                               << "<= " << std::setprecision(1) << (r.maxGapToLB * 100.0) << "% above optimal\n";
    return 0;
}
