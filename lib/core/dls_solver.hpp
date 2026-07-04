//---------------------------------------------------------------------------
// core/dls_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// The top-level contract every algorithm implements, exact or heuristic.
// This is the uniform interface: a caller (CLI driver, benchmark harness,
// test) holds a DLSSolver*, hands it a DLSInstance plus shared run config,
// and gets back a DLSSolution -- regardless of which algorithm is behind it.
//
// Parameter split enforced by this design:
//   - PROBLEM params  -> DLSInstance (passed to solve()).
//   - RUNTIME params  -> SolverConfig (passed to solve()): seed, time limit,
//                        verbosity. Shared by all algorithms.
//   - ALGORITHM params -> constructor of each concrete solver (e.g. a
//                        GAParams struct for the GA, branching rules for an
//                        exact method). They never leak into this interface,
//                        which is what keeps it uniform.
//
// Solvers that need to score fixed sequences (the GA, branch-and-bound)
// receive a ScheduleEvaluator via their own constructor (dependency
// injection); the base class does not mandate one, so a monolithic MILP
// solver can implement DLSSolver without any evaluator.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_DLS_SOLVER_HPP
#define DLS_CORE_DLS_SOLVER_HPP

#include <cstdint>
#include <optional>
#include <string>

#include "dls_instance.hpp"
#include "dls_solution.hpp"

namespace dls {

// Top-level categorization requested for the library.
enum class SolverCategory {
    Exact,      // returns a provably optimal schedule
    Heuristic   // returns a good schedule without an optimality guarantee
};

inline const char* toString(SolverCategory c) {
    return c == SolverCategory::Exact ? "exact" : "heuristic";
}

// Shared runtime configuration, common to every algorithm.
struct SolverConfig {
    // Seed for any randomness. std::nullopt => the solver picks/derives one
    // (e.g. from a clock). Set it explicitly for reproducible experiments.
    std::optional<std::uint64_t> seed;

    // Wall-clock budget in seconds; 0 means unlimited.
    double timeLimitSeconds = 0.0;

    // 0 = silent, higher = more diagnostic output.
    int verbosity = 0;

    // Config forwarded to the schedule evaluator, for solvers that use one.
    // (Defined in schedule_evaluator.hpp; kept here as a value would create a
    // dependency, so solvers that need it accept EvaluatorConfig separately.)
};

class DLSSolver {
public:
    virtual ~DLSSolver() = default;

    // Solve `instance` under shared `config`. Implementations should call
    // instance.validate() first and return a DLSSolution whose status
    // reflects the outcome (Optimal for exact, Feasible for heuristics that
    // found a schedule, Infeasible/Failure otherwise).
    virtual DLSSolution solve(const DLSInstance&  instance,
                              const SolverConfig& config) = 0;

    // Stable identifier for registry / CLI selection, e.g. "ga", "milp".
    virtual std::string name() const = 0;

    // Exact vs heuristic.
    virtual SolverCategory category() const = 0;
};

}  // namespace dls

#endif  // DLS_CORE_DLS_SOLVER_HPP
