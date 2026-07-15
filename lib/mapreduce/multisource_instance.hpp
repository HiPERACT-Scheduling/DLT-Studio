//---------------------------------------------------------------------------
// mapreduce/multisource_instance.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Multi-source map-phase scheduling (Gu, Liao, Yang & Li, "Scheduling
// Divisible Loads from Multiple Input Sources in MapReduce", IEEE CSE 2013).
// A genuinely different topology from the rest of the library: instead of
// one root distributing to m workers (the star, and every chain/tree/graph
// relay built on it), here m STORAGE nodes hold the input data and n >= m
// MAPPER nodes (the first m of which are the same physical nodes as the
// storage nodes — a mapper can process its own node's data for free) pull
// from potentially SEVERAL storage nodes concurrently. Bipartite many-to-
// many, not a tree.
//
// Only the map phase is modeled (the paper explicitly scopes reduce-phase
// optimization out as future work: "the performance of reduce phase largely
// depends on map phase"). Node j's map-phase finish time, given its local
// share L_j (only defined for j < m) and the shares R_ij pulled from each
// storage node i != j, is
//     T_j = (L_j + Σ_i R_ij) · t_j + Σ_i (R_ij · w_ij)
// — a mapper node's own compute rate t_j prices the whole volume it
// processes, while each individual transfer additionally costs R_ij·w_ij.
// The optimum has every node finish simultaneously (the standard DLT
// argument: an idle node could always take on more load to shorten the
// schedule), giving n equal-finish equations solved jointly with a
// conservation constraint by multisource_solver.hpp's LP.
//
// Two variants share this same instance, distinguished only by whether
// storageSizes is populated:
//   - "Problem A" (storageSizes empty): the LP is free to place the data
//     among storage nodes however is optimal; S_i is a DERIVED output.
//   - "Problem B" (storageSizes has exactly m entries): S_i is fixed input;
//     the LP only decides the read schedule (L_i, R_ij) honoring each
//     node's given supply.
//---------------------------------------------------------------------------

#ifndef DLS_MAPREDUCE_MULTISOURCE_INSTANCE_HPP
#define DLS_MAPREDUCE_MULTISOURCE_INSTANCE_HPP

#include <sstream>
#include <string>
#include <vector>

namespace dls {

struct MultiSourceInstance {
    double totalLoad      = 0.0;   // S
    int    numStorageNodes = 0;    // m
    int    numMappers      = 0;    // n (m <= n; nodes 0..m-1 are storage+mapper, m..n-1 are mapper-only)
    std::vector<double> computeRate;               // t_j, length n
    std::vector<std::vector<double>> transferRate;  // w_ij, size m x n (only i != j entries used)
    std::vector<double> storageSizes;               // S_i, length m if fixed-supply ("Problem B"); empty => free ("Problem A")

    bool isFixedSupply() const { return !storageSizes.empty(); }

    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (totalLoad <= 0.0) return fail("totalLoad (S) must be positive");
        if (numStorageNodes < 1) return fail("numStorageNodes (m) must be >= 1");
        if (numMappers < numStorageNodes) return fail("numMappers (n) must be >= numStorageNodes (m)");
        if (static_cast<int>(computeRate.size()) != numMappers)
            return fail("computeRate must have exactly n entries");
        for (double t : computeRate) if (t <= 0.0) return fail("a compute rate is non-positive");
        if (static_cast<int>(transferRate.size()) != numStorageNodes)
            return fail("transferRate must have exactly m rows");
        for (const auto& row : transferRate) {
            if (static_cast<int>(row.size()) != numMappers)
                return fail("each transferRate row must have exactly n entries");
            for (double w : row) if (w < 0.0) return fail("a transfer rate is negative");
        }
        if (!storageSizes.empty()) {
            if (static_cast<int>(storageSizes.size()) != numStorageNodes)
                return fail("storageSizes, if given, must have exactly m entries");
            for (double s : storageSizes) if (s < 0.0) return fail("a storage size is negative");
        }
        if (error) error->clear();
        return true;
    }
};

// Goal:   parse the line-oriented text format:
//   "S <load>" "m <numStorageNodes>" "n <numMappers>"
//   "rate <t>"              (repeated n times, node order 0..n-1)
//   "transfer <i> <j> <w>"  (repeated; i in [0,m), j in [0,n), j != i)
//   "storage <S_i>"         (repeated m times; if present at all, must be
//                            exactly m lines -- selects "Problem B" fixed
//                            supply; omitted entirely -- "Problem A" free)
// Input:  in - the text stream; inst - filled on success; error - filled on
//         failure.
// Output: true on success.
inline bool readMultiSourceInstance(std::istream& in, MultiSourceInstance& inst,
                                    std::string& error) {
    inst = MultiSourceInstance();
    std::vector<double> rates, storage;
    std::string line;
    int n = 0;
    auto fail = [&](const std::string& msg) { error = "line " + std::to_string(n) + ": " + msg; return false; };
    while (std::getline(in, line)) {
        ++n;
        if (auto h = line.find('#'); h != std::string::npos) line.erase(h);
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;
        if (key == "S") {
            if (!(ls >> inst.totalLoad)) return fail("expected a value after 'S'");
        } else if (key == "m") {
            if (!(ls >> inst.numStorageNodes)) return fail("expected a value after 'm'");
        } else if (key == "n") {
            if (!(ls >> inst.numMappers)) return fail("expected a value after 'n'");
        } else if (key == "rate") {
            double t; if (!(ls >> t)) return fail("expected a value after 'rate'"); rates.push_back(t);
        } else if (key == "transfer") {
            int i, j; double w;
            if (!(ls >> i >> j >> w)) return fail("transfer needs 'i j rate'");
            if (static_cast<int>(inst.transferRate.size()) != inst.numStorageNodes ||
                (inst.numMappers > 0 && (inst.transferRate.empty() ||
                 static_cast<int>(inst.transferRate[0].size()) != inst.numMappers))) {
                if (inst.numStorageNodes <= 0 || inst.numMappers <= 0)
                    return fail("'m' and 'n' must be declared before 'transfer' lines");
                inst.transferRate.assign(inst.numStorageNodes, std::vector<double>(inst.numMappers, 0.0));
            }
            if (i < 0 || i >= inst.numStorageNodes || j < 0 || j >= inst.numMappers)
                return fail("transfer indices out of range");
            inst.transferRate[i][j] = w;
        } else if (key == "storage") {
            double s; if (!(ls >> s)) return fail("expected a value after 'storage'"); storage.push_back(s);
        } else {
            return fail("unknown directive '" + key + "'");
        }
    }
    if (rates.empty()) { error = "no compute rates"; return false; }
    inst.computeRate = std::move(rates);
    if (inst.transferRate.empty() && inst.numStorageNodes > 0 && inst.numMappers > 0)
        inst.transferRate.assign(inst.numStorageNodes, std::vector<double>(inst.numMappers, 0.0));
    inst.storageSizes = std::move(storage);
    error.clear();
    return true;
}

}  // namespace dls

#endif  // DLS_MAPREDUCE_MULTISOURCE_INSTANCE_HPP
