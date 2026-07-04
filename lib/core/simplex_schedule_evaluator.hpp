//---------------------------------------------------------------------------
// core/simplex_schedule_evaluator.hpp
//
// Concrete ScheduleEvaluator backed by the legacy CSimplex LP solver
// (core/simplex.h). Given a DLS instance and a FIXED activation sequence,
// it builds the same linear program that the GA's EvaluatePopulation()
// constructed inline and returns the optimal makespan + load fragmentation.
//
// This is the bridge that lets the GA (and any future solver) call into
// core/ instead of carrying its own LP code. The LP itself is reproduced
// faithfully from heuristics/ga/ga.hpp:993-1440; only the data sources
// change (DLSInstance + sequence instead of TabProcParam + chromosome list).
//
// ---- LP variable layout (L = sequence length) ----------------------------
//   alpha_i  = load assigned to installment i        var index  i        (0..L-1)
//   t_i      = communication timing of installment i  var index  i + L
//   y_i      = "carried" load term for installment i  var index  i + 2L
//   Cmax     = makespan (objective)                   var index  3L
// Total variables: 3L + 1. (CSimplex variables are 1-based; GetSolution(k+1)
// returns the value of the variable at 0-based layout index k; GetSolution(0)
// returns the objective value.)
//---------------------------------------------------------------------------

#ifndef DLS_CORE_SIMPLEX_SCHEDULE_EVALUATOR_HPP
#define DLS_CORE_SIMPLEX_SCHEDULE_EVALUATOR_HPP

#include <vector>

#include "dls_instance.hpp"
#include "dls_lp_model.hpp"   // shared LP builder (single source of truth)
#include "dls_solution.hpp"
#include "energy_model.hpp"   // scheduleEnergy: four-state total energy
#include "schedule_evaluator.hpp"
#include "simplex.h"   // global CSimplex + OPTIMAL/UNBOUNDED/INFEASIBLE/FAILURE

namespace dls {

class SimplexScheduleEvaluator : public ScheduleEvaluator {
public:
    // Goal:   identify this evaluator backend for logging/reporting.
    // Output: the backend name, "simplex".
    std::string name() const override { return "simplex"; }

    // Goal:   for a FIXED activation sequence, compute the optimal load split
    //         and resulting makespan by building and solving the DLS LP.
    // Input:  instance - the DLS problem (processor costs, total load V);
    //         sequence - ordered 0-based processor indices, one per installment;
    //         config   - LP runtime knobs (max iterations, tolerance).
    // Output: a DLSSolution carrying the same sequence, a status (Optimal /
    //         Infeasible / Unbounded / Failure), and on success the makespan
    //         plus a per-installment fragment list (processor, loadSize, timing).
    DLSSolution evaluate(const DLSInstance&      instance,
                         const std::vector<int>& sequence,
                         const EvaluatorConfig&  config) const override {
        DLSSolution sol;
        sol.sequence = sequence;

        // MaxLoad (OptV) maximizes Σαᵢ within a deadline; it needs a finite
        // makespan limit (else the load — and the LP — is unbounded) and drops
        // the Σαᵢ = V conservation so the load is free to grow.
        const bool maxLoad = config.objective == EvalObjective::MaxLoad;
        if (maxLoad && !(config.makespanLimit < std::numeric_limits<double>::infinity())) {
            sol.status = SolveStatus::Failure; return sol;       // OptV without a deadline
        }

        // Build the shared LP, then hand its rows to CSimplex.
        DlsLpModel m = buildDlsLp(instance, sequence, config.costLimit, config.makespanLimit, maxLoad);
        if (m.status == LpBuildStatus::EmptySequence)   { sol.status = SolveStatus::Infeasible; return sol; }
        if (m.status == LpBuildStatus::IndexOutOfRange) { sol.status = SolveStatus::Failure;    return sol; }

        const int L = static_cast<int>(sequence.size());        // installments
        const int nCons = static_cast<int>(m.rows.size());      // constraint count

        CSimplex lp(m.numVars, nCons, config.seed);   // seeded anti-cycling RNG (no global rand)
        lp.SetMaxNumberOfIter(config.maxIterations);

        std::vector<double> gf(m.numVars, 0.0);  // objective coefficient vector
        if (config.objective == EvalObjective::MinCost) {
            if (m.hasEnergy) {
                // minimize total four-state energy (linear form for this sequence)
                gf = m.energyObjCoeffs;
            } else {
                // minimize the linear part of the cost: sum_i l_{d_i} alpha_i
                for (int i = 0; i < L; ++i)
                    gf[m.alphaVars[i]] = instance.processors()[sequence[i]].linearCost;
            }
        } else if (maxLoad) {
            // maximize total load Σαᵢ == minimize -Σαᵢ
            for (int i = 0; i < L; ++i) gf[m.alphaVars[i]] = -1.0;
        } else {
            gf[m.cmaxVar] = 1.0;                   // minimize Cmax
        }
        lp.SetObjectiveFunction(gf.data());

        for (int r = 0; r < nCons; ++r)
            lp.AddConstraintLE(m.rows[r].data(), m.rhs[r]);

        // ---- solve and translate the result ------------------------------
        switch (lp.Solve()) {
            case OPTIMAL: {
                sol.status   = SolveStatus::Optimal;
                sol.makespan = lp.GetSolution(m.cmaxVar + 1);  // the Cmax variable (mode-independent)
                sol.fragments.resize(L);
                for (int i = 0; i < L; ++i) {
                    LoadFragment& f = sol.fragments[i];
                    f.processorId = sequence[i];
                    // CSimplex GetSolution(k+1) returns the 0-based variable k.
                    f.loadSize    = lp.GetSolution(m.alphaVars[i] + 1);
                    f.commStart   = lp.GetSolution(m.timeVars[i] + 1);
                    f.carriedLoad = lp.GetSolution(m.carryVars[i] + 1);
                }
                sol.cost   = scheduleCost(instance, sol.fragments);
                sol.energy = scheduleEnergy(instance, sol.fragments, sol.makespan);
                break;
            }
            case UNBOUNDED:  sol.status = SolveStatus::Unbounded;  break;
            case INFEASIBLE: sol.status = SolveStatus::Infeasible; break;
            case FAILURE:    sol.status = SolveStatus::Failure;    break;
            default:         sol.status = SolveStatus::Failure;    break;
        }

        lp.Clear();
        return sol;
    }
};

}  // namespace dls

#endif  // DLS_CORE_SIMPLEX_SCHEDULE_EVALUATOR_HPP
