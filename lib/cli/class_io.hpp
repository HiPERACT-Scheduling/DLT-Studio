//---------------------------------------------------------------------------
// cli/class_io.hpp
//
// Text readers for the three non-single-load problem classes, so the `dls` CLI
// can drive all four classes with `--class=`. Each format is line-oriented with
// '#' comments, mirroring core/instance_io.hpp. Structural errors only; semantic
// checks are left to each instance's validate().
//
// MLSD (--class=mlsd):
//   task <size> [beta]              one divisible task (β = result fraction)
//   proc <S> <C> <A> <B>            one processor
//
// MapReduce (--class=mapreduce):
//   V <load>                        input load
//   startup <S> / readrate <C> / gamma0 <γ0> / reducers <r>
//   reducer_startup <s_red> / reducer_rate <a_red>
//   mapper <A>                      one mapper (macroscopic rate); repeatable
//
// Multilayer (--class=multilayer):
//   V <load> / mappers <m> / mapper_rate <A> / startup <S> / readrate <C>
//   gamma0 <γ0> / bisection <l>
//   layer <count> <s_red> <a_red> <gamma>   one reducer layer; repeatable
//---------------------------------------------------------------------------

#ifndef DLS_CLI_CLASS_IO_HPP
#define DLS_CLI_CLASS_IO_HPP

#include <istream>
#include <sstream>
#include <string>
#include <vector>

#include "mlsd/mlsd_instance.hpp"
#include "mapreduce/mapreduce_instance.hpp"
#include "mapreduce/multilayer_solver.hpp"
#include "topology/linear_chain.hpp"
#include "topology/tree.hpp"
#include "topology/graph.hpp"

namespace dls {

// Goal:   read an MLSD instance (tasks + processors).
inline bool readMlsdInstance(std::istream& in, MlsdInstance& inst, std::string& error) {
    std::vector<MlsdTask> tasks;
    std::vector<Processor> procs;
    std::string line;
    int n = 0;
    auto fail = [&](const std::string& m) { error = "line " + std::to_string(n) + ": " + m; return false; };
    while (std::getline(in, line)) {
        ++n;
        if (auto h = line.find('#'); h != std::string::npos) line.erase(h);
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;
        if (key == "task") {
            MlsdTask t;
            if (!(ls >> t.size)) return fail("task needs a size");
            ls >> t.resultFraction;                       // optional β
            tasks.push_back(t);
        } else if (key == "proc") {
            Processor p;
            if (!(ls >> p.commStartup >> p.commRate >> p.computeRate >> p.memoryLimit))
                return fail("proc needs 'S C A B'");
            procs.push_back(p);
        } else return fail("unknown directive '" + key + "' (expected task/proc)");
    }
    if (tasks.empty()) { error = "no tasks"; return false; }
    if (procs.empty()) { error = "no processors"; return false; }
    inst = MlsdInstance(std::move(tasks), std::move(procs));
    error.clear();
    return true;
}

// Goal:   read a MapReduce instance.
inline bool readMapReduceInstance(std::istream& in, MapReduceInstance& inst, std::string& error) {
    std::vector<double> rates;
    double V = 0.0;
    inst = MapReduceInstance();
    std::string line;
    int n = 0;
    auto fail = [&](const std::string& m) { error = "line " + std::to_string(n) + ": " + m; return false; };
    auto num  = [&](std::istringstream& ls, double& out, const char* k) { if (ls >> out) return true; return fail(std::string("expected a value after '") + k + "'"); };
    while (std::getline(in, line)) {
        ++n;
        if (auto h = line.find('#'); h != std::string::npos) line.erase(h);
        std::istringstream ls(line);
        std::string key; double x;
        if (!(ls >> key)) continue;
        if      (key == "V")               { if (!num(ls, V, "V")) return false; }
        else if (key == "startup")         { if (!num(ls, x, "startup")) return false;         inst.setStartup(x); }
        else if (key == "readrate")        { if (!num(ls, x, "readrate")) return false;        inst.setReadRate(x); }
        else if (key == "gamma0")          { if (!num(ls, x, "gamma0")) return false;          inst.setResultFraction(x); }
        else if (key == "reducers")        { if (!num(ls, x, "reducers")) return false;        inst.setNumReducers(static_cast<int>(x)); }
        else if (key == "reducer_startup") { if (!num(ls, x, "reducer_startup")) return false; inst.setReducerStartup(x); }
        else if (key == "reducer_rate")    { if (!num(ls, x, "reducer_rate")) return false;    inst.setReducerRate(x); }
        else if (key == "mapper")          { if (!num(ls, x, "mapper")) return false;          rates.push_back(x); }
        else return fail("unknown directive '" + key + "'");
    }
    if (rates.empty()) { error = "no mappers"; return false; }
    inst.mapperRates() = std::move(rates);
    inst.setTotalLoad(V);
    error.clear();
    return true;
}

// Goal:   read a Multilayer instance.
inline bool readMultilayerInstance(std::istream& in, MultilayerInstance& inst, std::string& error) {
    inst = MultilayerInstance();
    std::string line;
    int n = 0;
    auto fail = [&](const std::string& m) { error = "line " + std::to_string(n) + ": " + m; return false; };
    auto num  = [&](std::istringstream& ls, double& out, const char* k) { if (ls >> out) return true; return fail(std::string("expected a value after '") + k + "'"); };
    while (std::getline(in, line)) {
        ++n;
        if (auto h = line.find('#'); h != std::string::npos) line.erase(h);
        std::istringstream ls(line);
        std::string key; double x;
        if (!(ls >> key)) continue;
        if      (key == "V")           { if (!num(ls, x, "V")) return false;           inst.totalLoad = x; }
        else if (key == "mappers")     { if (!num(ls, x, "mappers")) return false;     inst.numMappers = static_cast<int>(x); }
        else if (key == "mapper_rate") { if (!num(ls, x, "mapper_rate")) return false; inst.mapperRate = x; }
        else if (key == "startup")     { if (!num(ls, x, "startup")) return false;     inst.startup = x; }
        else if (key == "readrate")    { if (!num(ls, x, "readrate")) return false;    inst.readRate = x; }
        else if (key == "gamma0")      { if (!num(ls, x, "gamma0")) return false;      inst.mapperFraction = x; }
        else if (key == "bisection")   { if (!num(ls, x, "bisection")) return false;   inst.bisectionWidth = static_cast<int>(x); }
        else if (key == "layer") {
            ReducerLayer L;
            double cnt;
            if (!(ls >> cnt >> L.startup >> L.rate >> L.resultFraction))
                return fail("layer needs 'count s_red a_red gamma'");
            L.count = static_cast<int>(cnt);
            inst.layers.push_back(L);
        } else return fail("unknown directive '" + key + "'");
    }
    if (inst.layers.empty()) { error = "no reducer layers"; return false; }
    error.clear();
    return true;
}

// Goal:   read a linear daisy chain. Lines: "V <load>" and
//         "node <A> <C>" per processor in chain order (P0's C is unused).
inline bool readChainInstance(std::istream& in, LinearChainInstance& inst, std::string& error) {
    std::vector<ChainProcessor> ps;
    double V = 0.0;
    std::string line; int n = 0;
    auto fail = [&](const std::string& m) { error = "line " + std::to_string(n) + ": " + m; return false; };
    while (std::getline(in, line)) {
        ++n;
        if (auto h = line.find('#'); h != std::string::npos) line.erase(h);
        std::istringstream ls(line); std::string key;
        if (!(ls >> key)) continue;
        if (key == "V") { if (!(ls >> V)) return fail("expected a value after 'V'"); }
        else if (key == "node") { ChainProcessor p; if (!(ls >> p.computeRate >> p.linkRate)) return fail("node needs 'A C'"); ps.push_back(p); }
        else return fail("unknown directive '" + key + "' (expected V/node)");
    }
    if (ps.empty()) { error = "no nodes"; return false; }
    inst = LinearChainInstance(std::move(ps), V);
    error.clear();
    return true;
}

// Goal:   read a multi-level tree. Lines: "V <load>" and
//         "node <A> <C> <parent>" in node-index order (node 0 = root, parent -1).
inline bool readTreeInstance(std::istream& in, TreeInstance& inst, std::string& error) {
    std::vector<TreeNode> ns;
    double V = 0.0;
    std::string line; int n = 0;
    auto fail = [&](const std::string& m) { error = "line " + std::to_string(n) + ": " + m; return false; };
    while (std::getline(in, line)) {
        ++n;
        if (auto h = line.find('#'); h != std::string::npos) line.erase(h);
        std::istringstream ls(line); std::string key;
        if (!(ls >> key)) continue;
        if (key == "V") { if (!(ls >> V)) return fail("expected a value after 'V'"); }
        else if (key == "node") { TreeNode t; if (!(ls >> t.computeRate >> t.linkRate >> t.parent)) return fail("node needs 'A C parent'"); ns.push_back(t); }
        else return fail("unknown directive '" + key + "' (expected V/node)");
    }
    if (ns.empty()) { error = "no nodes"; return false; }
    inst = TreeInstance(std::move(ns), V);
    error.clear();
    return true;
}

// Goal:   read a general graph. Lines: "V <load>", "node <A>" (index order,
//         node 0 = source), and "edge <u> <v> <rate>".
inline bool readGraphInstance(std::istream& in, GraphInstance& inst, std::string& error) {
    std::vector<double> A;
    std::vector<GraphEdge> edges;
    double V = 0.0;
    std::string line; int n = 0;
    auto fail = [&](const std::string& m) { error = "line " + std::to_string(n) + ": " + m; return false; };
    while (std::getline(in, line)) {
        ++n;
        if (auto h = line.find('#'); h != std::string::npos) line.erase(h);
        std::istringstream ls(line); std::string key;
        if (!(ls >> key)) continue;
        if (key == "V") { if (!(ls >> V)) return fail("expected a value after 'V'"); }
        else if (key == "node") { double a; if (!(ls >> a)) return fail("node needs 'A'"); A.push_back(a); }
        else if (key == "edge") { GraphEdge e; if (!(ls >> e.u >> e.v >> e.rate)) return fail("edge needs 'u v rate'"); edges.push_back(e); }
        else return fail("unknown directive '" + key + "' (expected V/node/edge)");
    }
    if (A.empty()) { error = "no nodes"; return false; }
    inst = GraphInstance(std::move(A), std::move(edges), V);
    error.clear();
    return true;
}

}  // namespace dls

#endif  // DLS_CLI_CLASS_IO_HPP
