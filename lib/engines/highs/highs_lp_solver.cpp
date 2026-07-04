//---------------------------------------------------------------------------
// engines/highs/highs_lp_solver.cpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// HiGHS implementation of the generic LP solve (see highs_lp_solver.hpp).
//---------------------------------------------------------------------------

#include "engines/highs/highs_lp_solver.hpp"

#include "Highs.h"

namespace dls {

HighsLpSolution solveLpViaHighs(int numVars,
                                const std::vector<double>& objective,
                                const std::vector<std::vector<double>>& rows,
                                const std::vector<double>& rhs,
                                double epsilon) {
    HighsLpSolution out;

    HighsModel model;
    HighsLp& lp = model.lp_;
    lp.num_col_ = numVars;
    lp.num_row_ = static_cast<HighsInt>(rows.size());
    lp.sense_   = ObjSense::kMinimize;
    lp.offset_  = 0.0;
    lp.col_cost_  = objective;
    lp.col_lower_.assign(numVars, 0.0);            // x >= 0
    lp.col_upper_.assign(numVars, kHighsInf);
    lp.row_lower_.assign(rows.size(), -kHighsInf); // -inf <= row <= rhs
    lp.row_upper_ = rhs;

    HighsSparseMatrix& A = lp.a_matrix_;
    A.format_  = MatrixFormat::kRowwise;
    A.num_col_ = numVars;
    A.num_row_ = static_cast<HighsInt>(rows.size());
    A.start_.clear(); A.index_.clear(); A.value_.clear();
    A.start_.push_back(0);
    for (const std::vector<double>& row : rows) {
        for (int j = 0; j < numVars; ++j)
            if (row[j] != 0.0) { A.index_.push_back(j); A.value_.push_back(row[j]); }
        A.start_.push_back(static_cast<HighsInt>(A.index_.size()));
    }

    Highs highs;
    highs.setOptionValue("output_flag", false);
    highs.setOptionValue("primal_feasibility_tolerance", epsilon);
    highs.setOptionValue("dual_feasibility_tolerance", epsilon);
    if (highs.passModel(model) != HighsStatus::kOk) { out.status = SolveStatus::Failure; return out; }
    highs.run();

    switch (highs.getModelStatus()) {
        case HighsModelStatus::kOptimal:
            out.status = SolveStatus::Optimal;
            out.values = highs.getSolution().col_value;
            break;
        case HighsModelStatus::kInfeasible:
            out.status = SolveStatus::Infeasible; break;
        case HighsModelStatus::kUnbounded:
        case HighsModelStatus::kUnboundedOrInfeasible:
            out.status = SolveStatus::Unbounded; break;
        default:
            out.status = SolveStatus::Failure; break;
    }
    return out;
}

}  // namespace dls
