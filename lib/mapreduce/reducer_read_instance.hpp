//---------------------------------------------------------------------------
// mapreduce/reducer_read_instance.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Types for the heterogeneous multi-channel reducer read scheduling problem:
// m mapper outputs, each ready at an integer tick, must each be read by every
// reducer that needs it (affinity, default: every reducer needs every
// mapper); reducers have heterogeneous integer read rates; at most `capacity`
// reads may be in progress system-wide at any tick (the shared channel /
// bisection-width limit). Decide, for every reducer, the order and exact
// start tick of its reads, minimizing the chosen objective.
//
// This fills a gap the existing MapReduce model (mapreduce_solver.hpp,
// Berlińska thesis §4.4) doesn't solve: its "many reducers" treatment either
// aggregates reducers into one r-times-faster resource (no real per-reducer
// decision), or — with a channel-capacity limit — fixes the read order by a
// closed-form round-robin formula rather than optimizing it. See
// exact/branch_and_price/reducer_read_bp_solver.hpp for the solver
// (heterogeneous-multi-channel-mapreduce.py's C++ port; that Python
// prototype is validated against exhaustive brute force — see
// tools/bp_reducer_scheduling.py).
//
// Discrete integer ticks throughout (exact arithmetic — this models the
// scheduling DECISION, not continuous timing; ready ticks and rates are
// taken as already-quantized inputs, e.g. from the existing closed-form
// mapper-side split upstream).
//---------------------------------------------------------------------------

#ifndef DLS_MAPREDUCE_REDUCER_READ_INSTANCE_HPP
#define DLS_MAPREDUCE_REDUCER_READ_INSTANCE_HPP

#include <algorithm>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

namespace dls {

enum class ReducerReadObjective { Makespan, Balance };
// Makespan: minimize the latest reducer finish tick (classic Cmax).
// Balance:  minimize the SUM of reducer finish ticks (classic total
//           completion time ΣCj) — a reducer left waiting while others hog
//           the channel counts against this objective even when Cmax is
//           unchanged. Not "total busy time": under fixed affinity that's a
//           constant, not something scheduling can influence.

struct ReducerReadInstance {
    std::vector<int> ready;                 // ready[i]: tick mapper i's output becomes readable
    std::vector<int> size;                  // size[i]: mapper i's output size (work units)
    std::vector<int> rate;                  // rate[j]: reducer j's ticks-per-unit read rate
    int capacity = 1;                       // l: max reads in progress system-wide, any tick
    std::vector<std::vector<int>> affinity; // affinity[j]: sorted mapper indices reducer j must read
    ReducerReadObjective objective = ReducerReadObjective::Makespan;

    int m() const { return static_cast<int>(ready.size()); }
    int r() const { return static_cast<int>(rate.size()); }
    int duration(int i, int j) const { return size[i] * rate[j]; }

    // Goal:   a safe upper bound on any reducer's finish tick, accounting for
    //         worst-case channel contention (every read across every reducer
    //         could end up fully serialized with capacity as low as 1), not
    //         just a reducer's own reads.
    // Output: max ready tick + sum of every read duration across every
    //         reducer's affinity set + 1.
    int horizon() const {
        int total = 0;
        for (int j = 0; j < r(); ++j)
            for (int i : affinity[j]) total += duration(i, j);
        int maxReady = ready.empty() ? 0 : *std::max_element(ready.begin(), ready.end());
        return maxReady + total + 1;
    }

    // Goal:   validate structural consistency (sizes, ranges, non-negativity).
    // Output: true if valid; else false with `error` set.
    bool validate(std::string* error) const {
        auto fail = [&](const std::string& msg) { if (error) *error = msg; return false; };
        if (ready.size() != size.size()) return fail("ready/size length mismatch");
        if (rate.empty()) return fail("no reducers");
        if (static_cast<int>(affinity.size()) != r()) return fail("affinity/reducer count mismatch");
        if (capacity < 1) return fail("capacity must be >= 1");
        for (int v : ready) if (v < 0) return fail("mapper ready tick must be >= 0");
        for (int v : size) if (v < 1) return fail("mapper size must be >= 1");
        for (int v : rate) if (v < 1) return fail("reducer rate must be >= 1");
        for (const auto& aff : affinity)
            for (int i : aff) if (i < 0 || i >= m()) return fail("affinity references an out-of-range mapper");
        return true;
    }
};

// One reducer's complete, feasible read plan (a branch-and-price "column").
struct ReducerReadColumn {
    int reducer = 0;
    std::vector<std::pair<int, int>> starts;  // (mapper index, start tick), one per mapper read
    int finish = 0;                            // tick the last read of this plan completes
    std::vector<int> occupied;                 // sorted ticks during which some read is in progress

    // Goal: a canonical, comparable identity for pool deduplication.
    // Output: a sorted copy of `starts` (independent of insertion order).
    std::vector<std::pair<int, int>> key() const {
        std::vector<std::pair<int, int>> k = starts;
        std::sort(k.begin(), k.end());
        return k;
    }
};

// Goal:   parse the line-oriented text format:
//   "capacity <l>"
//   "objective makespan|balance"          (optional, default makespan)
//   "mapper <ready> <size>"                (repeated, index = order of appearance)
//   "reducer <rate> [<m0> <m1> ...]"       (repeated; explicit mapper-index
//                                           affinity list, or omitted = every
//                                           mapper read by this reducer)
// Input:  in - the text stream; inst - filled on success; error - filled on
//         failure.
// Output: true on success.
inline bool readReducerReadInstance(std::istream& in, ReducerReadInstance& inst,
                                    std::string& error) {
    std::vector<int> ready, size, rate;
    std::vector<std::vector<int>> affinity;
    int capacity = 1;
    ReducerReadObjective objective = ReducerReadObjective::Makespan;
    std::string line;
    int n = 0;
    auto fail = [&](const std::string& msg) { error = "line " + std::to_string(n) + ": " + msg; return false; };
    while (std::getline(in, line)) {
        ++n;
        if (auto h = line.find('#'); h != std::string::npos) line.erase(h);
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;
        if (key == "capacity") {
            if (!(ls >> capacity)) return fail("capacity needs an integer");
        } else if (key == "objective") {
            std::string mode;
            if (!(ls >> mode)) return fail("objective needs makespan|balance");
            if (mode == "makespan") objective = ReducerReadObjective::Makespan;
            else if (mode == "balance") objective = ReducerReadObjective::Balance;
            else return fail("objective must be 'makespan' or 'balance'");
        } else if (key == "mapper") {
            int rd, sz;
            if (!(ls >> rd >> sz)) return fail("mapper needs 'ready size'");
            ready.push_back(rd); size.push_back(sz);
        } else if (key == "reducer") {
            int rt;
            if (!(ls >> rt)) return fail("reducer needs a rate");
            rate.push_back(rt);
            std::vector<int> aff;
            int idx;
            while (ls >> idx) aff.push_back(idx);
            affinity.push_back(std::move(aff));
        } else {
            return fail("unknown directive '" + key + "' (expected capacity/objective/mapper/reducer)");
        }
    }
    if (ready.empty()) { error = "no mappers"; return false; }
    if (rate.empty()) { error = "no reducers"; return false; }
    // affinity omitted (empty list after "reducer <rate>") -> every mapper
    std::vector<int> allMappers(ready.size());
    std::iota(allMappers.begin(), allMappers.end(), 0);
    for (auto& aff : affinity) {
        if (aff.empty()) aff = allMappers;
        else std::sort(aff.begin(), aff.end());
    }
    inst.ready = std::move(ready);
    inst.size = std::move(size);
    inst.rate = std::move(rate);
    inst.capacity = capacity;
    inst.affinity = std::move(affinity);
    inst.objective = objective;
    error.clear();
    return true;
}

}  // namespace dls

#endif  // DLS_MAPREDUCE_REDUCER_READ_INSTANCE_HPP
