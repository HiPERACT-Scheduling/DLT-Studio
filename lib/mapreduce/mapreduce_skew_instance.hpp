//---------------------------------------------------------------------------
// mapreduce/mapreduce_skew_instance.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Reducer-side partitioning-skew model (Berlińska & Drozdowski, MISTA 2013,
// "Mitigating Partitioning Skew in MapReduce Computations"). A homogeneous
// MapReduce system (all mappers rate A, all reducers rate a^sort/a^red) where
// the r-way key partition is NOT balanced: some reducers receive more
// intermediate data than others. Neither mapreduce_solver.hpp (equal γ₀V/r
// per reducer, by construction) nor reducer_read_bp_solver.hpp (a different
// problem — heterogeneous per-reducer READ scheduling with full-mapper-output
// reads, not key-partition skew) models this.
//
// The instance carries the raw partition sizes at k*r granularity (k=1 is
// the plain r-way partition every reducer gets exactly one of; k>1 is the
// finer partitioning the static mitigation algorithm (skew_static_solver.hpp)
// needs to have something to rebalance). skew_dynamic_solver.hpp instead
// requires k=1 (it works by moving load AFTER reading, not by repartitioning
// before it — see that file's header for why).
//
// One-port model throughout (Section 2 of the paper): a processor opens at
// most one communication channel at a time; the network-wide bisection width
// `l` caps concurrent channels before bandwidth is shared. `epsilon` is the
// size of one "location" control message (mapper/reducer/partition metadata),
// distinct from the bulk data rate `C`.
//---------------------------------------------------------------------------

#ifndef DLS_MAPREDUCE_SKEW_INSTANCE_HPP
#define DLS_MAPREDUCE_SKEW_INSTANCE_HPP

#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace dls {

struct MapReduceSkewInstance {
    double totalLoad     = 0.0;   // V
    int    numMappers    = 1;     // m
    double mapperRate     = 0.0;  // A (homogeneous)
    double readRate       = 0.0;  // C (time/byte, bulk data)
    double resultFraction = 1.0;  // γ: mapper output/input ratio
    double epsilon         = 0.0; // size of one control-message piece
    int    bisectionWidth  = 1;   // l: concurrent channels
    int    numReducers     = 1;   // r
    double sortRate  = 0.0;       // a^sort (τ(x) = a^sort · x · log2 x)
    double reduceRate = 0.0;      // a^red (linear: reduce(x) = a^red · x)
    double masterRate = 0.0;      // a^master (static method's assignment cost)
    int    k = 1;                 // over-partition factor; partitionSizes.size() == k*r
    std::vector<double> partitionSizes;   // raw sizes, length k*r, summing to γV

    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (totalLoad <= 0.0)   return fail("totalLoad (V) must be positive");
        if (numMappers < 1)     return fail("numMappers (m) must be >= 1");
        if (mapperRate < 0.0 || readRate < 0.0 || resultFraction < 0.0 || epsilon < 0.0)
            return fail("a mapper/communication parameter is negative");
        if (bisectionWidth < 1) return fail("bisectionWidth (l) must be >= 1");
        if (numReducers < 1)    return fail("numReducers (r) must be >= 1");
        if (sortRate < 0.0 || reduceRate < 0.0 || masterRate < 0.0)
            return fail("a reducer/master rate is negative");
        if (k < 1) return fail("k must be >= 1");
        if (static_cast<int>(partitionSizes.size()) != k * numReducers)
            return fail("partitionSizes must have exactly k*r entries");
        for (double p : partitionSizes) if (p < 0.0) return fail("a partition size is negative");
        if (error) error->clear();
        return true;
    }

    // Goal:   the total intermediate data volume (sum of all k*r partitions).
    double totalIntermediate() const {
        return std::accumulate(partitionSizes.begin(), partitionSizes.end(), 0.0);
    }
};

// Goal:   parse the line-oriented text format:
//   "V <load>" "mappers <m>" "mapper_rate <A>" "readrate <C>" "gamma0 <γ>"
//   "epsilon <ε>" "bisection <l>" "reducers <r>" "sort_rate <a_sort>"
//   "reduce_rate <a_red>" "master_rate <a_master>" "k <k>"
//   "partition <size>"   (repeated exactly k*r times, in order)
// Input:  in - the text stream; inst - filled on success; error - filled on
//         failure.
// Output: true on success.
inline bool readMapReduceSkewInstance(std::istream& in, MapReduceSkewInstance& inst,
                                      std::string& error) {
    inst = MapReduceSkewInstance();
    std::vector<double> sizes;
    std::string line;
    int n = 0;
    auto fail = [&](const std::string& msg) { error = "line " + std::to_string(n) + ": " + msg; return false; };
    auto num = [&](std::istringstream& ls, double& out, const char* k_) {
        if (ls >> out) return true; return fail(std::string("expected a value after '") + k_ + "'");
    };
    while (std::getline(in, line)) {
        ++n;
        if (auto h = line.find('#'); h != std::string::npos) line.erase(h);
        std::istringstream ls(line);
        std::string key; double x;
        if (!(ls >> key)) continue;
        if      (key == "V")           { if (!num(ls, x, "V")) return false;           inst.totalLoad = x; }
        else if (key == "mappers")     { if (!num(ls, x, "mappers")) return false;     inst.numMappers = static_cast<int>(x); }
        else if (key == "mapper_rate") { if (!num(ls, x, "mapper_rate")) return false; inst.mapperRate = x; }
        else if (key == "readrate")    { if (!num(ls, x, "readrate")) return false;    inst.readRate = x; }
        else if (key == "gamma0")      { if (!num(ls, x, "gamma0")) return false;      inst.resultFraction = x; }
        else if (key == "epsilon")     { if (!num(ls, x, "epsilon")) return false;     inst.epsilon = x; }
        else if (key == "bisection")   { if (!num(ls, x, "bisection")) return false;   inst.bisectionWidth = static_cast<int>(x); }
        else if (key == "reducers")    { if (!num(ls, x, "reducers")) return false;    inst.numReducers = static_cast<int>(x); }
        else if (key == "sort_rate")   { if (!num(ls, x, "sort_rate")) return false;   inst.sortRate = x; }
        else if (key == "reduce_rate") { if (!num(ls, x, "reduce_rate")) return false; inst.reduceRate = x; }
        else if (key == "master_rate") { if (!num(ls, x, "master_rate")) return false; inst.masterRate = x; }
        else if (key == "k")           { if (!num(ls, x, "k")) return false;           inst.k = static_cast<int>(x); }
        else if (key == "partition")   { if (!num(ls, x, "partition")) return false;   sizes.push_back(x); }
        else return fail("unknown directive '" + key + "'");
    }
    if (sizes.empty()) { error = "no partitions"; return false; }
    inst.partitionSizes = std::move(sizes);
    error.clear();
    return true;
}

}  // namespace dls

#endif  // DLS_MAPREDUCE_SKEW_INSTANCE_HPP
