//---------------------------------------------------------------------------
// mlsd/mlsd_evaluator.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// For a FIXED MLSD structure (task order + per-task processor sequences), the
// optimal load fractions alpha_{i,j} are a linear program — the analogue of the
// single-load schedule evaluator. This builds that LP (thesis eqs lp-eq1/lp-eq2)
// and solves it with CSimplex.
//
// ---- variables -----------------------------------------------------------
//   alpha[l][k] : load of the l-th task sent to the k-th processor in its
//                 activation sequence (flat index offset[l] + k),
//   Cmax        : makespan (objective).
//---------------------------------------------------------------------------

#ifndef DLS_MLSD_EVALUATOR_HPP
#define DLS_MLSD_EVALUATOR_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "core/simplex.h"        // CSimplex + result codes
#include "mlsd/mlsd_instance.hpp"
#include "engines/highs/highs_lp_solver.hpp"   // decl-only; HiGHS path under DLS_WITH_HIGHS

namespace dls {

class MlsdScheduleEvaluator {
public:
    // backend: "simplex" (default, dependency-free) or "highs" (DLS_WITH_HIGHS).
    explicit MlsdScheduleEvaluator(std::string backend = "simplex")
        : backend_(std::move(backend)) {}

    // Goal:   solve the load distribution for a fixed MLSD structure.
    // Input:  inst - the MLSD problem; s - task order + per-task processor
    //         sequences; seed - LP anti-cycling seed; maxIter - LP iteration cap.
    // Output: an MlsdSolution (status, makespan, per-(task,proc) loads).
    MlsdSolution evaluate(const MlsdInstance& inst, const MlsdStructure& s,
                          std::uint64_t seed = 0, int maxIter = 1000) const {
        MlsdSolution sol;
        const int n = static_cast<int>(s.taskOrder.size());
        if (n == 0) { sol.status = SolveStatus::Infeasible; return sol; }

        // Flat variable layout: alpha[l][k] at offset[l] + k.
        std::vector<int> offset(n);
        int v = 0;
        for (int l = 0; l < n; ++l) { offset[l] = v; v += static_cast<int>(s.procSeq[l].size()); }
        const int numLoad = v;
        auto var = [&](int l, int k) { return offset[l] + k; };

        // Result return (model A): when any task has β > 0, add a Tcomp variable
        // (computation makespan) and Cmax = Tcomp + total collection time. With
        // no return (default) the layout is just [alpha..., Cmax].
        bool hasReturn = false;
        for (const MlsdTask& t : inst.tasks()) if (t.resultFraction > 0.0) hasReturn = true;
        const int tcompVar      = hasReturn ? numLoad : -1;
        const int cmaxVar       = hasReturn ? numLoad + 1 : numLoad;
        const int nVars         = cmaxVar + 1;
        const int completionVar = hasReturn ? tcompVar : cmaxVar;  // lp-eq1 bounds this

        // Position of processor p in the l-th task's sequence, or -1.
        auto posOf = [&](int l, int p) -> int {
            const std::vector<int>& seq = s.procSeq[l];
            for (int k = 0; k < static_cast<int>(seq.size()); ++k)
                if (seq[k] == p) return k;
            return -1;
        };
        const std::vector<Processor>& P = inst.processors();

        std::vector<std::vector<double>> rows;   // each row: coeffs (size nVars)
        std::vector<double> rhs;                  // coeffs . x <= rhs
        std::vector<double> c(nVars, 0.0);
        auto zero  = [&] { std::fill(c.begin(), c.end(), 0.0); };
        auto addLE = [&](double r) { rows.push_back(c); rhs.push_back(r); };

        // (lp-eq2) Per-task load conservation: sum_k alpha[l][k] == V_{task l}.
        for (int l = 0; l < n; ++l) {
            const double Vl = inst.tasks()[s.taskOrder[l]].size;
            zero();
            for (int k = 0; k < static_cast<int>(s.procSeq[l].size()); ++k) c[var(l, k)] = 1.0;
            addLE(Vl);
            zero();
            for (int k = 0; k < static_cast<int>(s.procSeq[l].size()); ++k) c[var(l, k)] = -1.0;
            addLE(-Vl);
        }

        // (lp-eq1) Completion of the k-th processor of task l before Cmax:
        //   [send all tasks < l] + [send to procs 0..k of task l]
        //   + [compute tasks l..n-1 on that processor]  <= Cmax.
        for (int l = 0; l < n; ++l) {
            for (int k = 0; k < static_cast<int>(s.procSeq[l].size()); ++k) {
                const int p = s.procSeq[l][k];
                zero();
                double sConst = 0.0;                 // accumulated startup constants
                for (int lp = 0; lp < l; ++lp)        // send all earlier tasks fully
                    for (int kp = 0; kp < static_cast<int>(s.procSeq[lp].size()); ++kp) {
                        int q = s.procSeq[lp][kp];
                        c[var(lp, kp)] += P[q].commRate;
                        sConst += P[q].commStartup;
                    }
                for (int kp = 0; kp <= k; ++kp) {     // send to procs 0..k of task l
                    int q = s.procSeq[l][kp];
                    c[var(l, kp)] += P[q].commRate;
                    sConst += P[q].commStartup;
                }
                for (int ll = l; ll < n; ++ll) {      // compute tasks l..n-1 on p
                    int kk = posOf(ll, p);
                    if (kk >= 0) c[var(ll, kk)] += P[p].computeRate;
                }
                c[completionVar] = -1.0;
                addLE(-sConst);
            }
        }

        // Result return (model A): Cmax >= Tcomp + sum_{l,k} (S_p + C_p β_l α[l][k]).
        // All computation finishes by Tcomp, then results are collected over the
        // single port (cruder than per-installment FIFO, but well-defined).
        if (hasReturn) {
            zero();
            c[cmaxVar]  = -1.0;
            c[tcompVar] = 1.0;
            double sSum = 0.0;
            for (int l = 0; l < n; ++l) {
                const double betaL = inst.tasks()[s.taskOrder[l]].resultFraction;
                for (int k = 0; k < static_cast<int>(s.procSeq[l].size()); ++k) {
                    const int p = s.procSeq[l][k];
                    c[var(l, k)] += P[p].commRate * betaL;
                    sSum += P[p].commStartup;
                }
            }
            addLE(-sSum);
        }

        std::vector<double> objective(nVars, 0.0);
        objective[cmaxVar] = 1.0;                     // minimize Cmax

#ifdef DLS_WITH_HIGHS
        if (backend_ == "highs") {
            HighsLpSolution r = solveLpViaHighs(nVars, objective, rows, rhs);
            sol.status = r.status;
            if (r.status == SolveStatus::Optimal) {
                sol.makespan = r.values[cmaxVar];
                sol.loads.resize(n);
                for (int l = 0; l < n; ++l) {
                    sol.loads[l].resize(s.procSeq[l].size());
                    for (int k = 0; k < static_cast<int>(s.procSeq[l].size()); ++k)
                        sol.loads[l][k] = r.values[var(l, k)];
                }
            }
            return sol;
        }
#endif

        // ---- default: solve with CSimplex --------------------------------
        CSimplex lp(nVars, static_cast<int>(rows.size()), seed);
        lp.SetMaxNumberOfIter(maxIter);
        lp.SetObjectiveFunction(objective.data());
        for (std::size_t r = 0; r < rows.size(); ++r) lp.AddConstraintLE(rows[r].data(), rhs[r]);

        int res = lp.Solve();
        if (res == OPTIMAL) {
            sol.status = SolveStatus::Optimal;
            sol.makespan = lp.GetSolution(cmaxVar + 1);
            sol.loads.resize(n);
            for (int l = 0; l < n; ++l) {
                sol.loads[l].resize(s.procSeq[l].size());
                for (int k = 0; k < static_cast<int>(s.procSeq[l].size()); ++k)
                    sol.loads[l][k] = lp.GetSolution(var(l, k) + 1);
            }
        } else if (res == INFEASIBLE) {
            sol.status = SolveStatus::Infeasible;
        } else if (res == UNBOUNDED) {
            sol.status = SolveStatus::Unbounded;
        } else {
            sol.status = SolveStatus::Failure;
        }
        lp.Clear();
        return sol;
    }

private:
    std::string backend_;   // "simplex" (default) or "highs"
};

}  // namespace dls

#endif  // DLS_MLSD_EVALUATOR_HPP
