//---------------------------------------------------------------------------
// mapreduce/mapreduce_bwidth_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Bisection-width-limited MapReduce scheduler ("the third method", Berlińska
// & Drozdowski, technical report RA-09/09 "Dominance Properties for Divisible
// MapReduce Computations", section 5.2, Eqs. 59-64). The plain closed-form
// MapReduceSolver (mapreduce_solver.hpp) assumes the network can serve every
// reducer's read at full rate simultaneously; this solver instead caps the
// number of concurrent mapper-to-reducer read channels at `l` (the instance's
// bisectionWidth), which is the genuine constraint on a real cluster network.
//
// Model: mappers activate in order of non-decreasing rate Aᵢ (Lemma 1 of the
// same report — proven optimal for a single reducer, reused here as the
// report itself does for the many-reducer methods). All r reducers then read
// mapper outputs in that SAME fixed order (a permutation-flowshop-style
// simplification: "reducer read operations are ordered as in the permutation
// flowshop"), starting as soon as both the source mapper has finished AND a
// read channel (of the l available) is free. Function itv(i,j) (Eq. 59)
// gives the index of the time interval in which reducer j reads mapper i's
// output; vti(i) (Eq. 60) is the set of mappers read during interval i. The
// LP (Eqs. 61-64) below is the *direct* (non-periodicity-reduced) form: it
// has O(mr) rows instead of the report's further-reduced O(ml), which is a
// deliberate trade — this is simpler to get right and validate, and mr stays
// small for realistic cluster sizes; nothing here requires the periodicity
// argument the report uses to shrink the LP for very large r/l ratios.
//
// A mapper assigned zero load still occupies its position's iS startup slot
// (Eq. 68 ties t_i to position i regardless of αᵢ) — this is a property of
// the paper's own formulation (it assumes exactly m mappers, fixed order,
// no "drop the slowest mapper" step like the plain closed form has), not a
// bug: supplying more mappers than useful can only be discovered by the
// caller comparing objective values across different m, which this solver
// does not do on its own.
//
// Requires HiGHS; guarded by DLS_WITH_HIGHS like the other exact/ solvers.
//---------------------------------------------------------------------------

#ifndef DLS_MAPREDUCE_BWIDTH_SOLVER_HPP
#define DLS_MAPREDUCE_BWIDTH_SOLVER_HPP

#ifdef DLS_WITH_HIGHS

#include <algorithm>
#include <cmath>
#include <numeric>
#include <string>
#include <vector>

#include "Highs.h"
#include "core/dls_solution.hpp"
#include "mapreduce/mapreduce_instance.hpp"

namespace dls {

struct MapReduceBwidthSolution {
    SolveStatus         status   = SolveStatus::NotSolved;
    double              makespan = 0.0;   // mapper+transfer finish + reducerTime
    std::vector<double> mapperLoads;      // αᵢ for the mappers, in activation order
    std::vector<int>    mapperOrder;      // original indices, in activation (increasing Aᵢ) order
    double              reducerTime = 0.0;// t_red, the (equal) reducer execution time

    bool feasible() const {
        return status == SolveStatus::Optimal || status == SolveStatus::Feasible;
    }
};

class MapReduceBwidthSolver {
public:
    std::string name() const { return "mapreduce-bwidth"; }

    // Goal:   compute the optimal "third method" schedule under a bisection-
    //         width read-channel limit.
    // Input:  instance - the problem (mapper rates, V, r reducers, l channels).
    // Output: a MapReduceBwidthSolution; Optimal with loads/order/makespan, or
    //         Infeasible when the instance or the LP is invalid.
    MapReduceBwidthSolution solve(const MapReduceInstance& instance) const {
        MapReduceBwidthSolution sol;
        std::string error;
        if (!instance.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }

        const int m = static_cast<int>(instance.numMappers());
        const int r = instance.numReducers();
        const int l = instance.bisectionWidth();
        const double V = instance.totalLoad();
        const double S = instance.startup();
        const double gOverR = instance.resultFraction() * instance.readRate()
                             / static_cast<double>(r);   // γC/r, the per-read rate factor (Eq. 63)

        // Mapper indices sorted by non-decreasing rate Aᵢ (Lemma 1).
        std::vector<int> order(m);
        std::iota(order.begin(), order.end(), 0);
        const auto& A = instance.mapperRates();
        std::stable_sort(order.begin(), order.end(),
                         [&](int a, int b) { return A[a] < A[b]; });

        // itv(i,j): the 1-indexed interval in which reducer j (1..r) reads
        // mapper i's (1..m, sorted position) output (Eq. 59).
        auto itv = [&](int i, int j) {
            const int ceilJoverL = (j + l - 1) / l;
            return (ceilJoverL - 1) * m + i + (j - 1) % l;
        };
        const int N = itv(m, r);   // total number of read intervals

        // vti(i): distinct (sorted-position) mapper indices read during
        // interval i, deduplicated per mapper (Eq. 60).
        std::vector<std::vector<int>> vti(N + 1);   // 1-indexed; vti[0] unused
        for (int i = 1; i <= m; ++i) {
            std::vector<bool> seenInterval(N + 1, false);
            for (int j = 1; j <= r; ++j) {
                const int iv = itv(i, j);
                if (!seenInterval[iv]) { seenInterval[iv] = true; vti[iv].push_back(i); }
            }
        }

        // Variables: αᵢ (0-indexed 0..m-1, sorted position), then tᵢ
        // (1-indexed i=1..N+1, mapped to columns m..m+N).
        const int nvars = m + (N + 1);
        auto tCol = [&](int i) { return m + (i - 1); };   // t_i (1-indexed) -> column

        HighsModel model;
        HighsLp& lp = model.lp_;
        lp.num_col_ = nvars;
        lp.sense_ = ObjSense::kMinimize;
        lp.offset_ = 0.0;
        lp.col_cost_.assign(nvars, 0.0);
        lp.col_cost_[tCol(N + 1)] = 1.0;         // minimize t_{N+1} (Eq. 61)
        lp.col_lower_.assign(nvars, 0.0);
        lp.col_upper_.assign(nvars, kHighsInf);

        int nIneq = 0;
        for (int i = 1; i <= N; ++i) nIneq += static_cast<int>(vti[i].size());
        const int nRows = m + nIneq + 1;   // Eq.62 rows, Eq.63 rows, Eq.64 row
        lp.num_row_ = nRows;
        lp.row_lower_.assign(nRows, 0.0);
        lp.row_upper_.assign(nRows, 0.0);

        std::vector<std::vector<std::pair<int, double>>> rowEntries(nRows);

        // Eq. 62: iS + Aᵢαᵢ = tᵢ  =>  Aᵢαᵢ - tᵢ = -iS   for i = 1..m.
        for (int i = 1; i <= m; ++i) {
            const int row = i - 1;
            const int mapperIdx = order[i - 1];
            rowEntries[row].emplace_back(i - 1, A[mapperIdx]);
            rowEntries[row].emplace_back(tCol(i), -1.0);
            lp.row_lower_[row] = -static_cast<double>(i) * S;
            lp.row_upper_[row] = -static_cast<double>(i) * S;
        }

        // Eq. 63: (γC/r)·αₖ ≤ t_{i+1} - tᵢ  for i = 1..N, k ∈ vti(i).
        int rowCursor = m;
        for (int i = 1; i <= N; ++i) {
            for (int k : vti[i]) {
                const int row = rowCursor++;
                rowEntries[row].emplace_back(k - 1, gOverR);
                rowEntries[row].emplace_back(tCol(i + 1), -1.0);
                rowEntries[row].emplace_back(tCol(i), 1.0);
                lp.row_lower_[row] = -kHighsInf;
                lp.row_upper_[row] = 0.0;
            }
        }

        // Eq. 64: Σ αᵢ = V.
        {
            const int row = rowCursor;
            for (int i = 0; i < m; ++i) rowEntries[row].emplace_back(i, 1.0);
            lp.row_lower_[row] = V;
            lp.row_upper_[row] = V;
        }

        HighsSparseMatrix& Amat = lp.a_matrix_;
        Amat.format_ = MatrixFormat::kRowwise;
        Amat.num_col_ = nvars;
        Amat.num_row_ = nRows;
        // Amat.start_ already holds a leading {0} from HighsSparseMatrix's own
        // constructor — do not push another one here (see the reducer-read
        // solver's note on this exact gotcha).
        for (auto& row : rowEntries) {
            std::sort(row.begin(), row.end());
            for (auto& [c, v] : row) { Amat.index_.push_back(c); Amat.value_.push_back(v); }
            Amat.start_.push_back(static_cast<HighsInt>(Amat.index_.size()));
        }

        Highs highs;
        highs.setOptionValue("output_flag", false);
        if (highs.passModel(model) != HighsStatus::kOk) { sol.status = SolveStatus::Failure; return sol; }
        highs.run();
        if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
            sol.status = SolveStatus::Infeasible; return sol;
        }

        const HighsSolution& hsol = highs.getSolution();
        sol.mapperLoads.assign(m, 0.0);
        for (int i = 0; i < m; ++i) sol.mapperLoads[i] = hsol.col_value[i];
        sol.mapperOrder = order;

        const double z = instance.resultFraction() * V / static_cast<double>(r);
        sol.reducerTime = instance.reducerStartup()
                         + instance.reducerRate() * std::max(0.0, z * std::log2(std::max(z, 1.0)));

        sol.makespan = hsol.col_value[tCol(N + 1)] + sol.reducerTime;
        sol.status   = SolveStatus::Optimal;
        return sol;
    }
};

}  // namespace dls

#endif  // DLS_WITH_HIGHS

#endif  // DLS_MAPREDUCE_BWIDTH_SOLVER_HPP
