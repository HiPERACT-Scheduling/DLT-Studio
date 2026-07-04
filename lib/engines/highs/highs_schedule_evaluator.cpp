//---------------------------------------------------------------------------
// engines/highs/highs_schedule_evaluator.cpp
//
// HiGHS-backed implementation of the DLS per-sequence LP. Builds the shared
// model (core/dls_lp_model.hpp) as a HiGHS LP — all variables non-negative,
// every row an inequality coeffs . x <= rhs — solves it, and maps the result
// back into a DLSSolution.
//---------------------------------------------------------------------------

#include "engines/highs/highs_schedule_evaluator.hpp"

#include "core/dls_lp_model.hpp"
#include "core/energy_model.hpp"
#include "Highs.h"

namespace dls {

// Goal:   solve the DLS per-sequence LP with HiGHS.
// Input:  instance - the problem; sequence - fixed activation order;
//         config   - runtime knobs (epsilon used as a feasibility tolerance).
// Output: DLSSolution with status Optimal/Infeasible/Unbounded/Failure and,
//         on Optimal, the makespan and per-installment load + timing.
DLSSolution HighsScheduleEvaluator::evaluate(const DLSInstance&      instance,
                                             const std::vector<int>& sequence,
                                             const EvaluatorConfig&  config) const {
    DLSSolution sol;
    sol.sequence = sequence;

    // MaxLoad (OptV) maximizes Σαᵢ within a deadline; needs a finite makespan
    // limit and drops the load conservation (Σαᵢ = V) so the load is free.
    const bool maxLoad = config.objective == EvalObjective::MaxLoad;
    if (maxLoad && !(config.makespanLimit < kHighsInf)) {
        sol.status = SolveStatus::Failure; return sol;       // OptV without a deadline
    }

    DlsLpModel m = buildDlsLp(instance, sequence, config.costLimit, config.makespanLimit, maxLoad);
    if (m.status == LpBuildStatus::EmptySequence)   { sol.status = SolveStatus::Infeasible; return sol; }
    if (m.status == LpBuildStatus::IndexOutOfRange) { sol.status = SolveStatus::Failure;    return sol; }

    const int L = static_cast<int>(sequence.size());

    // ---- translate the shared model into a HiGHS LP ----------------------
    HighsModel model;
    HighsLp& lp = model.lp_;
    lp.num_col_ = m.numVars;
    lp.num_row_ = static_cast<HighsInt>(m.rows.size());
    lp.sense_   = ObjSense::kMinimize;
    lp.offset_  = 0.0;

    lp.col_cost_.assign(m.numVars, 0.0);
    if (config.objective == EvalObjective::MinCost) {
        if (m.hasEnergy) {                               // minimize total four-state energy
            for (int v = 0; v < m.numVars; ++v) lp.col_cost_[v] = m.energyObjCoeffs[v];
        } else {                                         // minimize sum_i l_{d_i} alpha_i
            for (int i = 0; i < static_cast<int>(sequence.size()); ++i)
                lp.col_cost_[m.alphaVars[i]] = instance.processors()[sequence[i]].linearCost;
        }
    } else if (maxLoad) {                                 // maximize Σαᵢ == minimize -Σαᵢ
        for (int i = 0; i < L; ++i) lp.col_cost_[m.alphaVars[i]] = -1.0;
    } else {
        lp.col_cost_[m.cmaxVar] = 1.0;                   // minimize Cmax
    }
    lp.col_lower_.assign(m.numVars, 0.0);                // all variables >= 0
    lp.col_upper_.assign(m.numVars, kHighsInf);
    lp.row_lower_.assign(m.rows.size(), -kHighsInf);     // -inf <= row
    lp.row_upper_ = m.rhs;                                // row <= rhs

    // Row-wise sparse matrix built from the dense model rows.
    HighsSparseMatrix& A = lp.a_matrix_;
    A.format_  = MatrixFormat::kRowwise;
    A.num_col_ = m.numVars;
    A.num_row_ = static_cast<HighsInt>(m.rows.size());
    A.start_.clear(); A.index_.clear(); A.value_.clear();
    A.start_.push_back(0);
    for (const std::vector<double>& row : m.rows) {
        for (int j = 0; j < m.numVars; ++j)
            if (row[j] != 0.0) { A.index_.push_back(j); A.value_.push_back(row[j]); }
        A.start_.push_back(static_cast<HighsInt>(A.index_.size()));
    }

    // ---- solve -----------------------------------------------------------
    Highs highs;
    highs.setOptionValue("output_flag", false);                       // quiet
    highs.setOptionValue("primal_feasibility_tolerance", config.epsilon);
    highs.setOptionValue("dual_feasibility_tolerance", config.epsilon);
    if (highs.passModel(model) != HighsStatus::kOk) { sol.status = SolveStatus::Failure; return sol; }
    highs.run();

    switch (highs.getModelStatus()) {
        case HighsModelStatus::kOptimal: {
            const HighsSolution& s = highs.getSolution();
            sol.status   = SolveStatus::Optimal;
            sol.makespan = s.col_value[m.cmaxVar];
            sol.fragments.resize(L);
            for (int i = 0; i < L; ++i) {
                sol.fragments[i].processorId = sequence[i];
                sol.fragments[i].loadSize    = s.col_value[m.alphaVars[i]];
                sol.fragments[i].commStart   = s.col_value[m.timeVars[i]];
                sol.fragments[i].carriedLoad = s.col_value[m.carryVars[i]];
            }
            sol.cost   = scheduleCost(instance, sol.fragments);
            sol.energy = scheduleEnergy(instance, sol.fragments, sol.makespan);
            break;
        }
        case HighsModelStatus::kInfeasible:
            sol.status = SolveStatus::Infeasible; break;
        case HighsModelStatus::kUnbounded:
        case HighsModelStatus::kUnboundedOrInfeasible:
            sol.status = SolveStatus::Unbounded; break;
        default:
            sol.status = SolveStatus::Failure; break;
    }
    return sol;
}

}  // namespace dls
