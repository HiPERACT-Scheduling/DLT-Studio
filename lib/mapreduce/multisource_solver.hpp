//---------------------------------------------------------------------------
// mapreduce/multisource_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// LP solver for multi-source map-phase scheduling (see
// multisource_instance.hpp for the model). Node j's finish time is
//     T_j = (L_j + Σ_i R_ij)·t_j + Σ_i (R_ij·w_ij) = L_j·t_j + Σ_i R_ij·(t_j + w_ij)
// (L_j only exists for j < m). The optimum has every one of the n nodes
// finish simultaneously (the classic DLT argument — an idle node could
// always be given more load to shorten the schedule), so a shared variable
// T is introduced and each node contributes one equality row "T_j - T = 0"
// with per-column coefficients read directly off the formula above — a
// cleaner equivalent of the paper's own pairwise chained equations (T_1=T_2,
// T_2=T_3, ...), not a re-derivation of different mathematics.
//
// "Problem A" (instance.storageSizes empty): one extra row, total volume
// conservation ΣL + ΣR = S — the LP is free to place data among storage
// nodes however minimizes T; S_i is DERIVED from the solution afterward.
// "Problem B" (storageSizes has m entries): m extra rows, one per storage
// node's fixed supply L_i + Σ_j R_ij = S_i.
//
// This is a genuine LP, not a closed form: L,R have no upper bound and the
// equality-constraint count (n + 1, or n + m) is generally far fewer than
// the m·(n-1)+m variables, so non-negativity can bind (some R_ij would want
// to go negative in an unconstrained solve) — the same reason MapReduce's
// bisection-width solver (mapreduce_bwidth_solver.hpp) needs HiGHS rather
// than a closed-form recurrence. Requires HiGHS; guarded by DLS_WITH_HIGHS.
//---------------------------------------------------------------------------

#ifndef DLS_MAPREDUCE_MULTISOURCE_SOLVER_HPP
#define DLS_MAPREDUCE_MULTISOURCE_SOLVER_HPP

#ifdef DLS_WITH_HIGHS

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "Highs.h"
#include "core/dls_solution.hpp"
#include "mapreduce/multisource_instance.hpp"

namespace dls {

struct MultiSourceSolution {
    SolveStatus status   = SolveStatus::NotSolved;
    double      makespan = 0.0;   // T, the common map-phase finish time
    std::vector<double> localLoads;               // L_i, length m
    std::vector<std::vector<double>> transferLoads; // R_ij, size m x n (0 where i == j or unused)
    std::vector<double> storageSizes;             // S_i: derived (Problem A) or echoed (Problem B)

    bool feasible() const { return status == SolveStatus::Optimal || status == SolveStatus::Feasible; }
};

class MultiSourceSolver {
public:
    std::string name() const { return "multisource"; }

    // Goal:   solve for the optimal map-phase schedule (and, for "Problem
    //         A", the optimal data placement too).
    // Input:  instance - m storage nodes, n mappers, per-mapper compute
    //         rates, m x n transfer rates, and optionally fixed supplies.
    // Output: a MultiSourceSolution; Optimal, or Infeasible if the instance
    //         or the LP is invalid.
    MultiSourceSolution solve(const MultiSourceInstance& instance) const {
        MultiSourceSolution sol;
        std::string error;
        if (!instance.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }

        const int m = instance.numStorageNodes, n = instance.numMappers;
        const bool fixedSupply = instance.isFixedSupply();

        // Columns: L_0..L_{m-1}, then R_ij for i in [0,m), j in [0,n), j!=i, then T.
        std::vector<std::vector<int>> rCol(m, std::vector<int>(n, -1));
        int col = m;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (j != i) rCol[i][j] = col++;
        const int tCol = col;
        const int nvars = tCol + 1;

        HighsModel model;
        HighsLp& lp = model.lp_;
        lp.num_col_ = nvars;
        lp.sense_ = ObjSense::kMinimize;
        lp.offset_ = 0.0;
        lp.col_cost_.assign(nvars, 0.0);
        lp.col_cost_[tCol] = 1.0;   // minimize T
        lp.col_lower_.assign(nvars, 0.0);
        lp.col_upper_.assign(nvars, kHighsInf);

        const int nRows = n + (fixedSupply ? m : 1);
        lp.num_row_ = nRows;
        lp.row_lower_.assign(nRows, 0.0);
        lp.row_upper_.assign(nRows, 0.0);   // all rows are equalities

        std::vector<std::vector<std::pair<int, double>>> rowEntries(nRows);

        // n rows: T_j - T = 0, i.e. Ljt_j + sum_i Rij(t_j+w_ij) - T = 0.
        for (int j = 0; j < n; ++j) {
            auto& row = rowEntries[j];
            const double tj = instance.computeRate[j];
            if (j < m) row.emplace_back(j, tj);   // L_j
            for (int i = 0; i < m; ++i) {
                if (i == j) continue;
                row.emplace_back(rCol[i][j], tj + instance.transferRate[i][j]);
            }
            row.emplace_back(tCol, -1.0);
        }

        if (fixedSupply) {
            // m rows: L_i + sum_j R_ij = S_i.
            for (int i = 0; i < m; ++i) {
                const int row = n + i;
                rowEntries[row].emplace_back(i, 1.0);
                for (int j = 0; j < n; ++j) if (j != i) rowEntries[row].emplace_back(rCol[i][j], 1.0);
                lp.row_lower_[row] = instance.storageSizes[i];
                lp.row_upper_[row] = instance.storageSizes[i];
            }
        } else {
            // 1 row: total volume conservation, sum L + sum R = S.
            const int row = n;
            for (int i = 0; i < m; ++i) {
                rowEntries[row].emplace_back(i, 1.0);
                for (int j = 0; j < n; ++j) if (j != i) rowEntries[row].emplace_back(rCol[i][j], 1.0);
            }
            lp.row_lower_[row] = instance.totalLoad;
            lp.row_upper_[row] = instance.totalLoad;
        }

        HighsSparseMatrix& A = lp.a_matrix_;
        A.format_ = MatrixFormat::kRowwise;
        A.num_col_ = nvars;
        A.num_row_ = nRows;
        // A.start_ already holds a leading {0} from HighsSparseMatrix's own
        // constructor -- do not push another one here (see the reducer-read
        // and bisection-width solvers' notes on this exact gotcha).
        for (auto& row : rowEntries) {
            std::sort(row.begin(), row.end());
            for (auto& [c, v] : row) { A.index_.push_back(c); A.value_.push_back(v); }
            A.start_.push_back(static_cast<HighsInt>(A.index_.size()));
        }

        Highs highs;
        highs.setOptionValue("output_flag", false);
        if (highs.passModel(model) != HighsStatus::kOk) { sol.status = SolveStatus::Failure; return sol; }
        highs.run();
        if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
            sol.status = SolveStatus::Infeasible; return sol;
        }

        const HighsSolution& hsol = highs.getSolution();
        sol.localLoads.assign(m, 0.0);
        for (int i = 0; i < m; ++i) sol.localLoads[i] = hsol.col_value[i];
        sol.transferLoads.assign(m, std::vector<double>(n, 0.0));
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                if (j != i) sol.transferLoads[i][j] = hsol.col_value[rCol[i][j]];

        sol.storageSizes.assign(m, 0.0);
        for (int i = 0; i < m; ++i) {
            double s = sol.localLoads[i];
            for (int j = 0; j < n; ++j) if (j != i) s += sol.transferLoads[i][j];
            sol.storageSizes[i] = fixedSupply ? instance.storageSizes[i] : s;
        }

        sol.makespan = hsol.col_value[tCol];
        sol.status   = SolveStatus::Optimal;
        return sol;
    }
};

}  // namespace dls

#endif  // DLS_WITH_HIGHS

#endif  // DLS_MAPREDUCE_MULTISOURCE_SOLVER_HPP
