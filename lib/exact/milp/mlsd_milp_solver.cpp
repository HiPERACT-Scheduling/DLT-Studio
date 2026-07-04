//---------------------------------------------------------------------------
// exact/milp/mlsd_milp_solver.cpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// HiGHS MILP for MLSD (see the header). One MILP per task order decides the
// per-task processor slot assignment and the load fractions; the n! orders are
// enumerated outside. The model mirrors mlsd_evaluator's LP exactly (β = 0).
//
// ---- columns (n tasks in the fixed order, m processors, m slots/task) -----
//   y[l][s][p] in {0,1}   processor p occupies slot s of task l
//   alpha[l][s] >= 0       load in slot s of task l
//   w[l][s][p] >= 0        McCormick of y[l][s][p]*alpha[l][s] (U = max_l V_l)
//   Cmax       >= 0        makespan (objective)
//---------------------------------------------------------------------------

#include "exact/milp/mlsd_milp_solver.hpp"

#include <algorithm>
#include <numeric>
#include <vector>

#include "Highs.h"

namespace dls {

namespace {

// Solve one fixed-task-order MILP. `order[l]` is the task index in position l.
// Returns the makespan (and fills `loads`) on success; sets ok=false otherwise.
double solveFixedOrder(const MlsdInstance& inst, const std::vector<int>& order,
                       const MlsdMilpParams& params,
                       std::vector<std::vector<double>>& loads, bool& ok) {
    ok = false;
    const int n = static_cast<int>(order.size());
    const int m = static_cast<int>(inst.numProcessors());
    const auto& P = inst.processors();

    std::vector<double> Vl(n);
    for (int l = 0; l < n; ++l) Vl[l] = inst.tasks()[order[l]].size;

    double maxV = 0.0, maxS = 0.0, maxC = 0.0, maxA = 0.0, sumV = 0.0;
    for (int l = 0; l < n; ++l) { maxV = std::max(maxV, Vl[l]); sumV += Vl[l]; }
    for (const Processor& p : P) { maxS = std::max(maxS, p.commStartup); maxC = std::max(maxC, p.commRate); maxA = std::max(maxA, p.computeRate); }
    const double U    = maxV;                                  // alpha[l][s] <= V_l <= maxV
    const double bigM = (maxC + maxA) * sumV + n * m * maxS + 1.0;   // Cmax upper bound

    // Column layout.
    const int OFF_Y = 0;
    const int OFF_A = n * m * m;
    const int OFF_W = n * m * m + n * m;
    const int CMAX  = 2 * n * m * m + n * m;
    const int numCols = CMAX + 1;
    auto yCol = [&](int l, int s, int p) { return OFF_Y + (l * m + s) * m + p; };
    auto aCol = [&](int l, int s)        { return OFF_A + l * m + s; };
    auto wCol = [&](int l, int s, int p) { return OFF_W + (l * m + s) * m + p; };

    HighsModel model;
    HighsLp& lp = model.lp_;
    lp.num_col_ = numCols;
    lp.sense_   = ObjSense::kMinimize;
    lp.offset_  = 0.0;
    lp.col_cost_.assign(numCols, 0.0);
    lp.col_cost_[CMAX] = 1.0;
    lp.col_lower_.assign(numCols, 0.0);
    lp.col_upper_.assign(numCols, kHighsInf);
    lp.integrality_.assign(numCols, HighsVarType::kContinuous);
    for (int l = 0; l < n; ++l)
        for (int s = 0; s < m; ++s) {
            lp.col_upper_[aCol(l, s)] = U;
            for (int p = 0; p < m; ++p) {
                lp.col_upper_[yCol(l, s, p)] = 1.0;
                lp.integrality_[yCol(l, s, p)] = HighsVarType::kInteger;
                lp.col_upper_[wCol(l, s, p)] = U;
            }
        }

    std::vector<double> rlo, rhi;
    std::vector<HighsInt> astart{0}, aindex;
    std::vector<double> avalue;
    auto addRow = [&](const std::vector<std::pair<int,double>>& terms, double lo, double hi) {
        for (const auto& t : terms) { aindex.push_back(t.first); avalue.push_back(t.second); }
        astart.push_back(static_cast<HighsInt>(aindex.size()));
        rlo.push_back(lo); rhi.push_back(hi);
    };
    // A dense-row helper for the big completion constraint (avoids duplicate cols).
    std::vector<double> dense(numCols, 0.0);
    auto addDense = [&](double lo, double hi) {
        std::vector<std::pair<int,double>> terms;
        for (int j = 0; j < numCols; ++j) if (dense[j] != 0.0) { terms.push_back({j, dense[j]}); dense[j] = 0.0; }
        addRow(terms, lo, hi);
    };

    for (int l = 0; l < n; ++l) {
        // Slot holds at most one processor; processor used at most once per task.
        for (int s = 0; s < m; ++s) {
            std::vector<std::pair<int,double>> r;
            for (int p = 0; p < m; ++p) r.push_back({yCol(l, s, p), 1.0});
            addRow(r, -kHighsInf, 1.0);
        }
        for (int p = 0; p < m; ++p) {
            std::vector<std::pair<int,double>> r;
            for (int s = 0; s < m; ++s) r.push_back({yCol(l, s, p), 1.0});
            addRow(r, -kHighsInf, 1.0);
        }
        // Symmetry: pack used slots toward the front.
        for (int s = 0; s + 1 < m; ++s) {
            std::vector<std::pair<int,double>> r;
            for (int p = 0; p < m; ++p) { r.push_back({yCol(l, s, p), 1.0}); r.push_back({yCol(l, s + 1, p), -1.0}); }
            addRow(r, 0.0, kHighsInf);
        }
        // alpha = sum_p w, McCormick, and per-task conservation.
        for (int s = 0; s < m; ++s) {
            std::vector<std::pair<int,double>> r{{aCol(l, s), 1.0}};
            for (int p = 0; p < m; ++p) r.push_back({wCol(l, s, p), -1.0});
            addRow(r, 0.0, 0.0);
            for (int p = 0; p < m; ++p) {
                addRow({{wCol(l,s,p), 1.0}, {yCol(l,s,p), -U}}, -kHighsInf, 0.0);                 // w <= U y
                addRow({{wCol(l,s,p), 1.0}, {aCol(l,s), -1.0}}, -kHighsInf, 0.0);                 // w <= alpha
                addRow({{wCol(l,s,p), 1.0}, {aCol(l,s), -1.0}, {yCol(l,s,p), -U}}, -U, kHighsInf);// w >= alpha-U(1-y)
            }
        }
        std::vector<std::pair<int,double>> conv;
        for (int s = 0; s < m; ++s) conv.push_back({aCol(l, s), 1.0});
        addRow(conv, Vl[l], Vl[l]);
    }

    // Completion per (task l, slot s, processor p): when y[l][s][p] = 1,
    //   Cmax >= comm(l,s) + A_p * sum_{ll>=l} (p's load in task ll).
    for (int l = 0; l < n; ++l)
        for (int s = 0; s < m; ++s)
            for (int p = 0; p < m; ++p) {
                // comm: earlier tasks fully sent
                for (int ll = 0; ll < l; ++ll)
                    for (int sp = 0; sp < m; ++sp)
                        for (int q = 0; q < m; ++q) { dense[yCol(ll,sp,q)] += -P[q].commStartup; dense[wCol(ll,sp,q)] += -P[q].commRate; }
                // comm: task l slots 0..s
                for (int sp = 0; sp <= s; ++sp)
                    for (int q = 0; q < m; ++q) { dense[yCol(l,sp,q)] += -P[q].commStartup; dense[wCol(l,sp,q)] += -P[q].commRate; }
                // compute: tasks ll>=l, processor p
                for (int ll = l; ll < n; ++ll)
                    for (int sp = 0; sp < m; ++sp) dense[wCol(ll,sp,p)] += -P[p].computeRate;
                dense[CMAX]          += 1.0;
                dense[yCol(l,s,p)]   += -bigM;            // big-M: only binds when y=1
                addDense(-bigM, kHighsInf);               // Cmax - comm - compute - M*y >= -M
            }

    lp.num_row_   = static_cast<HighsInt>(rlo.size());
    lp.row_lower_ = rlo;
    lp.row_upper_ = rhi;
    HighsSparseMatrix& A = lp.a_matrix_;
    A.format_  = MatrixFormat::kRowwise;
    A.num_col_ = numCols;
    A.num_row_ = lp.num_row_;
    A.start_   = astart;
    A.index_   = aindex;
    A.value_   = avalue;

    Highs highs;
    highs.setOptionValue("output_flag", false);
    if (params.timeLimitSec > 0.0) highs.setOptionValue("time_limit", params.timeLimitSec);
    if (params.mipGap       > 0.0) highs.setOptionValue("mip_rel_gap", params.mipGap);
    if (highs.passModel(model) != HighsStatus::kOk) return 0.0;
    highs.run();
    if (highs.getModelStatus() != HighsModelStatus::kOptimal) return 0.0;

    const HighsSolution& sv = highs.getSolution();
    if (!sv.value_valid) return 0.0;

    // Extract per-task loads in slot order (skip empty slots).
    loads.assign(n, {});
    for (int l = 0; l < n; ++l)
        for (int s = 0; s < m; ++s) {
            int chosen = -1;
            for (int p = 0; p < m; ++p) if (sv.col_value[yCol(l, s, p)] > 0.5) { chosen = p; break; }
            if (chosen < 0) continue;
            loads[l].push_back(sv.col_value[aCol(l, s)]);
        }
    ok = true;
    return sv.col_value[CMAX];
}

}  // namespace

MlsdSolution MlsdMilpSolver::solve(const MlsdInstance& inst) const {
    MlsdSolution best;
    std::string error;
    if (!inst.validate(&error)) { best.status = SolveStatus::Infeasible; return best; }

    const int n = static_cast<int>(inst.numTasks());
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);

    bool any = false;
    double bestMk = 0.0;
    do {
        std::vector<std::vector<double>> loads;
        bool ok = false;
        const double mk = solveFixedOrder(inst, order, params_, loads, ok);
        if (ok && (!any || mk < bestMk - 1e-9)) {
            any = true; bestMk = mk;
            best.makespan  = mk;
            best.loads     = loads;
            best.taskOrder = order;
        }
    } while (std::next_permutation(order.begin(), order.end()));

    best.status = any ? SolveStatus::Optimal : SolveStatus::Failure;
    return best;
}

}  // namespace dls
