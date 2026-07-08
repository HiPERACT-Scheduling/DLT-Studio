//---------------------------------------------------------------------------
// exact/milp/multi_milp_solver.cpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// HiGHS MILP for the multi-installment DLS makespan problem. See the header for
// scope. The schedule is K port slots; a processor may occupy several. Per-slot
// finish times f_k with a big-M same-processor FIFO coupling reproduce the
// carried-load (overlapped pipeline) makespan of the shared LP.
//
// ---- columns (N processors, K slots) -------------------------------------
//   x[i][k] in {0,1}   processor i in slot k
//   alpha_k >= 0        load in slot k
//   w[i][k] >= 0        McCormick of x[i][k]*alpha_k (U = V)
//   t_k     >= 0        communication start of slot k (t_0 = 0)
//   f_k     >= 0        computation finish of slot k
//   Cmax    >= 0        makespan (objective)
//---------------------------------------------------------------------------

#include "exact/milp/multi_milp_solver.hpp"

#include <algorithm>
#include <vector>

#include "core/dls_lp_model.hpp"   // scheduleCost
#include "Highs.h"

namespace dls {

DLSSolution MultiMilpSolver::solve(const DLSInstance& instance, const SolverConfig& config) {
    (void)config;                                   // MILP optimum is RNG-independent
    DLSSolution sol;

    std::string error;
    if (!instance.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }
    if (!params_.validate(&error))  { sol.status = SolveStatus::Failure;    return sol; }

    const int N = static_cast<int>(instance.numProcessors());   // processors
    const int K = params_.maxInstallments;                      // port slots
    const double V = instance.totalLoad();

    // Big-M (a valid upper bound on any finish time) and McCormick bound U.
    double maxStartup = 0.0, maxRate = 0.0, maxCompStart = 0.0;
    for (const Processor& p : instance.processors()) {
        maxStartup   = std::max(maxStartup,   p.commStartup);
        maxRate      = std::max(maxRate,      p.commRate + p.computeRate);
        maxCompStart = std::max(maxCompStart, p.computeStartup);
    }
    const double bigM = K * (maxStartup + maxCompStart) + maxRate * V + 1.0;
    const double U    = V;                           // alpha_k <= V by conservation

    // ---- column layout ---------------------------------------------------
    const int OFF_X     = 0;
    const int OFF_ALPHA = N * K;
    const int OFF_W     = N * K + K;
    const int OFF_T     = 2 * N * K + K;
    const int OFF_F     = 2 * N * K + 2 * K;
    const int OFF_CMAX  = 2 * N * K + 3 * K;
    const int numCols   = OFF_CMAX + 1;
    auto xCol     = [&](int i, int k) { return OFF_X + i * K + k; };
    auto alphaCol = [&](int k)        { return OFF_ALPHA + k; };
    auto wCol     = [&](int i, int k) { return OFF_W + i * K + k; };
    auto tCol     = [&](int k)        { return OFF_T + k; };
    auto fCol     = [&](int k)        { return OFF_F + k; };
    const int cmaxCol = OFF_CMAX;

    HighsModel model;
    HighsLp& lp = model.lp_;
    lp.num_col_ = numCols;
    lp.sense_   = ObjSense::kMinimize;
    lp.offset_  = 0.0;
    lp.col_cost_.assign(numCols, 0.0);
    lp.col_cost_[cmaxCol] = 1.0;                     // minimize Cmax
    lp.col_lower_.assign(numCols, 0.0);
    lp.col_upper_.assign(numCols, kHighsInf);
    lp.integrality_.assign(numCols, HighsVarType::kContinuous);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < K; ++k) { lp.col_upper_[xCol(i, k)] = 1.0; lp.integrality_[xCol(i, k)] = HighsVarType::kInteger; }
    for (int k = 0; k < K; ++k) {
        lp.col_upper_[alphaCol(k)] = U;
        for (int i = 0; i < N; ++i) lp.col_upper_[wCol(i, k)] = U;
    }
    lp.col_upper_[tCol(0)] = 0.0;                    // t_0 = 0

    // ---- rows ------------------------------------------------------------
    std::vector<double> rlo, rhi;
    std::vector<HighsInt> astart{0}, aindex;
    std::vector<double> avalue;
    auto addRow = [&](const std::vector<std::pair<int,double>>& terms, double lo, double hi) {
        for (const auto& t : terms) { aindex.push_back(t.first); avalue.push_back(t.second); }
        astart.push_back(static_cast<HighsInt>(aindex.size()));
        rlo.push_back(lo); rhi.push_back(hi);
    };
    const auto& P = instance.processors();

    // Each slot holds at most one processor.
    for (int k = 0; k < K; ++k) {
        std::vector<std::pair<int,double>> r;
        for (int i = 0; i < N; ++i) r.push_back({xCol(i, k), 1.0});
        addRow(r, -kHighsInf, 1.0);
    }
    // Per-chunk buffer (and empty-slot forcing): alpha_k <= sum_i B_i x[i][k].
    // Clamp the coefficient to min(B_i, V): since alpha_k <= U = V always holds,
    // any memory >= V is non-binding, and an unbounded-memory sentinel (e.g. 1e18)
    // as a raw coefficient wrecks HiGHS conditioning (returns Failure). min(B_i,V)
    // preserves the feasible region exactly while keeping coefficients O(V).
    for (int k = 0; k < K; ++k) {
        std::vector<std::pair<int,double>> r{{alphaCol(k), 1.0}};
        for (int i = 0; i < N; ++i) r.push_back({xCol(i, k), -std::min(P[i].memoryLimit, V)});
        addRow(r, -kHighsInf, 0.0);
    }
    // alpha_k = sum_i w[i][k].
    for (int k = 0; k < K; ++k) {
        std::vector<std::pair<int,double>> r{{alphaCol(k), 1.0}};
        for (int i = 0; i < N; ++i) r.push_back({wCol(i, k), -1.0});
        addRow(r, 0.0, 0.0);
    }
    // McCormick of w = x*alpha.
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < K; ++k) {
            addRow({{wCol(i,k), 1.0}, {xCol(i,k), -U}}, -kHighsInf, 0.0);                   // w <= U x
            addRow({{wCol(i,k), 1.0}, {alphaCol(k), -1.0}}, -kHighsInf, 0.0);               // w <= alpha
            addRow({{wCol(i,k), 1.0}, {alphaCol(k), -1.0}, {xCol(i,k), -U}}, -U, kHighsInf);// w >= alpha-U(1-x)
        }
    // Port serialization: t_k >= t_{k-1} + sum_i(S_i x + C_i w)[k-1].
    for (int k = 1; k < K; ++k) {
        std::vector<std::pair<int,double>> r{{tCol(k), 1.0}, {tCol(k-1), -1.0}};
        for (int i = 0; i < N; ++i) { r.push_back({xCol(i,k-1), -P[i].commStartup}); r.push_back({wCol(i,k-1), -P[i].commRate}); }
        addRow(r, 0.0, kHighsInf);
    }
    // Finish >= receive + compute:  f_k >= t_k + sum_i((S+p)x + (C+A)w)[k].
    for (int k = 0; k < K; ++k) {
        std::vector<std::pair<int,double>> r{{fCol(k), 1.0}, {tCol(k), -1.0}};
        for (int i = 0; i < N; ++i) {
            r.push_back({xCol(i,k), -(P[i].commStartup + P[i].computeStartup)});
            r.push_back({wCol(i,k), -(P[i].commRate + P[i].computeRate)});
        }
        addRow(r, 0.0, kHighsInf);
    }
    // Same-processor FIFO (no-overlap carried-load model of the shared LP):
    // a later chunk on the same processor adds its FULL comm+compute after the
    // previous completion, so for processor i and slots k' < k,
    //   f_k >= f_{k'} + cost_k - M(2 - x[i][k] - x[i][k']),
    //   cost_k = sum_j((S_j+p_j) x[j][k] + (C_j+A_j) w[j][k]).
    for (int i = 0; i < N; ++i)
        for (int k = 1; k < K; ++k)
            for (int kp = 0; kp < k; ++kp) {
                std::vector<std::pair<int,double>> r{{fCol(k), 1.0}, {fCol(kp), -1.0}, {xCol(i,kp), -bigM}};
                for (int j = 0; j < N; ++j) {
                    const double xc = -(P[j].commStartup + P[j].computeStartup) - (j == i ? bigM : 0.0);
                    r.push_back({xCol(j,k), xc});
                    r.push_back({wCol(j,k), -(P[j].commRate + P[j].computeRate)});
                }
                addRow(r, -2.0 * bigM, kHighsInf);
            }
    // Makespan.
    for (int k = 0; k < K; ++k) addRow({{cmaxCol, 1.0}, {fCol(k), -1.0}}, 0.0, kHighsInf);
    // Load conservation.
    {
        std::vector<std::pair<int,double>> r;
        for (int k = 0; k < K; ++k) r.push_back({alphaCol(k), 1.0});
        addRow(r, V, V);
    }
    // Symmetry break: pack used slots toward the front.
    for (int k = 0; k < K - 1; ++k) {
        std::vector<std::pair<int,double>> r;
        for (int i = 0; i < N; ++i) { r.push_back({xCol(i,k), 1.0}); r.push_back({xCol(i,k+1), -1.0}); }
        addRow(r, 0.0, kHighsInf);
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

    // ---- solve -----------------------------------------------------------
    Highs highs;
    highs.setOptionValue("output_flag", false);
    if (params_.timeLimitSec > 0.0) highs.setOptionValue("time_limit", params_.timeLimitSec);
    if (params_.mipGap       > 0.0) highs.setOptionValue("mip_rel_gap", params_.mipGap);
    if (highs.passModel(model) != HighsStatus::kOk) { sol.status = SolveStatus::Failure; return sol; }
    highs.run();

    const HighsModelStatus ms = highs.getModelStatus();
    const HighsSolution& s = highs.getSolution();
    if (ms == HighsModelStatus::kOptimal && s.value_valid)      sol.status = SolveStatus::Optimal;
    else if (s.value_valid && (ms == HighsModelStatus::kTimeLimit || ms == HighsModelStatus::kIterationLimit ||
                               ms == HighsModelStatus::kSolutionLimit)) sol.status = SolveStatus::Feasible;
    else if (ms == HighsModelStatus::kInfeasible) { sol.status = SolveStatus::Infeasible; return sol; }
    else if (ms == HighsModelStatus::kUnbounded || ms == HighsModelStatus::kUnboundedOrInfeasible) { sol.status = SolveStatus::Unbounded; return sol; }
    else { sol.status = SolveStatus::Failure; return sol; }

    sol.makespan = s.col_value[cmaxCol];
    for (int k = 0; k < K; ++k) {
        int chosen = -1;
        for (int i = 0; i < N; ++i) if (s.col_value[xCol(i, k)] > 0.5) { chosen = i; break; }
        if (chosen < 0) continue;
        LoadFragment f;
        f.processorId = chosen;
        f.loadSize    = s.col_value[alphaCol(k)];
        f.commStart   = s.col_value[tCol(k)];
        sol.sequence.push_back(chosen);
        sol.fragments.push_back(f);
    }
    sol.cost = scheduleCost(instance, sol.fragments);
    return sol;
}

}  // namespace dls
