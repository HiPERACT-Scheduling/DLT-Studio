//---------------------------------------------------------------------------
// core/evaluator_factory.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Runtime selection of a ScheduleEvaluator backend by name. This is the seam
// that lets solvers switch between the built-in CSimplex backend and optional
// external backends (e.g. HiGHS) without any solver code knowing the details.
//
// Backend availability is two-tiered:
//   - compile-time: a backend is only registered here if it was compiled in
//     (external backends are guarded by their feature macro);
//   - runtime: makeScheduleEvaluator(name) returns the matching backend, or
//     nullptr for an unknown / not-compiled-in name.
//
// "simplex" (the dependency-free CSimplex backend) is always available.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_EVALUATOR_FACTORY_HPP
#define DLS_CORE_EVALUATOR_FACTORY_HPP

#include <memory>
#include <string>

#include "schedule_evaluator.hpp"
#include "simplex_schedule_evaluator.hpp"
#ifdef DLS_WITH_HIGHS
#include "engines/highs/highs_schedule_evaluator.hpp"   // registered in Stage 3
#endif

namespace dls {

// Goal:   create a schedule-evaluator backend by name.
// Input:  name - "simplex" (always available); "highs" only when built with
//         DLS_WITH_HIGHS.
// Output: an owning pointer to the backend, or nullptr if the name is unknown
//         or that backend was not compiled in.
inline std::unique_ptr<ScheduleEvaluator> makeScheduleEvaluator(const std::string& name) {
    if (name == "simplex") return std::make_unique<SimplexScheduleEvaluator>();
#ifdef DLS_WITH_HIGHS
    if (name == "highs")   return std::make_unique<HighsScheduleEvaluator>();
#endif
    return nullptr;
}

}  // namespace dls

#endif  // DLS_CORE_EVALUATOR_FACTORY_HPP
