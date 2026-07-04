//---------------------------------------------------------------------------
// engines/highs/highs_schedule_evaluator.hpp
//
// Optional ScheduleEvaluator backend that solves the DLS per-sequence LP with
// the external HiGHS solver. It solves the SAME model as the CSimplex backend
// (both consume core/dls_lp_model.hpp), so the two are interchangeable and
// mutually validating.
//
// Only the declaration lives here (no HiGHS headers), so the evaluator factory
// can include it cheaply when DLS_WITH_HIGHS is set; the implementation (which
// includes Highs.h) is compiled separately in highs_schedule_evaluator.cpp and
// linked only in builds configured with -DDLS_WITH_HIGHS=ON.
//---------------------------------------------------------------------------

#ifndef DLS_BACKENDS_HIGHS_SCHEDULE_EVALUATOR_HPP
#define DLS_BACKENDS_HIGHS_SCHEDULE_EVALUATOR_HPP

#include <string>
#include <vector>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/schedule_evaluator.hpp"

namespace dls {

class HighsScheduleEvaluator : public ScheduleEvaluator {
public:
    // Goal:   identify this backend.
    // Output: "highs".
    std::string name() const override { return "highs"; }

    // Goal:   solve the DLS per-sequence LP with HiGHS (same model as CSimplex).
    // Input:  instance, sequence, config (see ScheduleEvaluator::evaluate).
    // Output: a DLSSolution (status + makespan + per-installment fragments).
    DLSSolution evaluate(const DLSInstance&      instance,
                         const std::vector<int>& sequence,
                         const EvaluatorConfig&  config) const override;
};

}  // namespace dls

#endif  // DLS_BACKENDS_HIGHS_SCHEDULE_EVALUATOR_HPP
