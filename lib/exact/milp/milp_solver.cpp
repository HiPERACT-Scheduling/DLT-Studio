//---------------------------------------------------------------------------
// exact/milp/milp_solver.cpp
//
// HiGHS MILP implementation of the single-installment DLS problem. Builds one
// MILP over processor-to-slot assignment binaries, loads, and timing, and
// solves it. See milp_solver.hpp for scope/semantics.
//
// ---- model (N processors, K = min(maxInstallments, N) ordered slots) ------
//   x[i][k] in {0,1}   processor i assigned to slot k
//   alpha_k >= 0        load in slot k
//   w[i][k] >= 0        linearizes x[i][k]*alpha_k (McCormick, U = max_i B_i)
//   t_k     >= 0        communication start time of slot k (t_0 = 0)
//   Cmax    >= 0        makespan (objective)
//---------------------------------------------------------------------------

#include "exact/milp/milp_solver.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include "core/dls_lp_model.hpp"   // scheduleCost
#include "core/energy_model.hpp"   // scheduleEnergy (four-state total energy)
#include "Highs.h"

namespace dls {

DLSSolution MilpSolver::solve(const DLSInstance& instance, const SolverConfig& config) {
    (void)config;                                  // MILP optimum is RNG-independent
    DLSSolution sol;

    std::string error;
    if (!instance.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }
    if (!params_.validate(&error))  { sol.status = SolveStatus::Failure;    return sol; }

    const int N = static_cast<int>(instance.numProcessors());          // processors
    const int K = std::min(params_.maxInstallments, N);                // slots
    const double V = instance.totalLoad();

    // Upper bound on any single load (McCormick big-M and alpha/w bounds). A
    // single installment never carries more than the whole load V, so cap there:
    // this keeps the big-M finite even when memoryLimit is effectively unbounded
    // (e.g. 1e18), which HiGHS rejects as a matrix coefficient > 1e15.
    double U = 0.0;
    for (const Processor& p : instance.processors())
        U = std::max(U, (p.memoryLimit > 0.0) ? p.memoryLimit : V);
    U = std::min(U, V);

    // Piecewise-linear convex processing time: active when some processor has a
    // multi-piece processing function. Default (single implicit piece {p, A})
    // keeps the classic inline term, so existing MILP models are unchanged.
    bool hasPieces = false;
    for (const Processor& p : instance.processors())
        if (p.effectivePieces().size() > 1) { hasPieces = true; break; }

    // Energy criterion (Marszałkowski 2020). When the instance carries an energy
    // model, "cost" means total four-state energy E (consistent with the per-
    // sequence evaluator): minimizeCost => minimize E, costLimit => E-limit.
    // The energy block is materialized only when E enters the objective or a
    // limit; otherwise E is still reported post-hoc via scheduleEnergy().
    const bool energyMode = instance.usesEnergyModel();
    const bool addEnergy  = energyMode &&
        (params_.minimizeCost || params_.costLimit < std::numeric_limits<double>::infinity());
    // The energy idle term subtracts run time, so run time must be PINNED, not
    // free. That holds for affine (single-piece) processing; multi-piece convex
    // time would let the solver inflate run time to fake-reduce idle energy, so
    // optimizing energy with piecewise TIME is not supported (piecewise ENERGY
    // is — it does not set hasPieces).
    if (addEnergy && hasPieces) { sol.status = SolveStatus::Failure; return sol; }

    // Availability (p, r, d): bigM = a valid upper bound on any completion time.
    bool anyRelease = false, anyDeadline = false;
    double bigM = 0.0, maxR = 0.0;
    for (const Processor& p : instance.processors()) {
        if (p.releaseTime > 0.0) anyRelease = true;
        if (p.deadline    > 0.0) anyDeadline = true;
        // Processing upper bound = max piece evaluated at the full load V
        // (load_k <= V, slopes >= 0). For the single piece this is p + A*V.
        double compMax = 0.0;
        for (const ComputePiece& pc : p.effectivePieces())
            compMax = std::max(compMax, pc.intercept + pc.slope * V);
        bigM += p.commStartup + p.commRate * V + compMax;
        maxR = std::max(maxR, p.releaseTime);
    }
    bigM += maxR + 1.0;

    // ---- column layout ---------------------------------------------------
    const int OFF_X     = 0;                 // x[i][k] at OFF_X + i*K + k
    const int OFF_ALPHA = N * K;             // alpha_k
    const int OFF_W     = N * K + K;         // w[i][k] at OFF_W + i*K + k
    const int OFF_T     = 2 * N * K + K;     // t_k
    const int OFF_CMAX  = 2 * N * K + 2 * K; // Cmax
    const int OFF_PROC  = OFF_CMAX + 1;      // proc_k (only when hasPieces)
    const int baseCols  = hasPieces ? OFF_PROC + K : OFF_CMAX + 1;
    // Energy block (only when addEnergy): used_i (proc i used), cx_i (= Cmax·used_i,
    // McCormick), er_k (running energy of slot k, >= its energy pieces).
    const int OFF_USED  = baseCols;          // used_i  at OFF_USED + i
    const int OFF_CX    = OFF_USED + N;       // cx_i    at OFF_CX   + i
    const int OFF_ER    = OFF_CX   + N;       // er_k    at OFF_ER   + k
    const int numCols   = addEnergy ? OFF_ER + K : baseCols;
    auto xCol     = [&](int i, int k) { return OFF_X + i * K + k; };
    auto alphaCol = [&](int k)        { return OFF_ALPHA + k; };
    auto wCol     = [&](int i, int k) { return OFF_W + i * K + k; };
    auto tCol     = [&](int k)        { return OFF_T + k; };
    auto procCol  = [&](int k)        { return OFF_PROC + k; };
    auto usedCol  = [&](int i)        { return OFF_USED + i; };
    auto cxCol    = [&](int i)        { return OFF_CX + i; };
    auto erCol    = [&](int k)        { return OFF_ER + k; };
    const int cmaxCol = OFF_CMAX;

    HighsModel model;
    HighsLp& lp = model.lp_;
    lp.num_col_ = numCols;
    lp.sense_   = ObjSense::kMinimize;
    lp.offset_  = 0.0;

    // Energy as a linear form over the MILP variables (single-installment, affine
    // time). Per processor i: startup P^S·S·used + network P^N·C·W − idle
    // P^I·(busy) + idle base P^I·cx, plus master E₀ = P^N₀·dist + P^I₀·(Cmax−dist).
    // busy = (S+p)·used + (C+A)·W ; dist = Σ(S·used + C·W). Mirrors energy_model.hpp.
    auto setEnergyObjective = [&](std::vector<double>& cost) {
        const double pNi0 = instance.originatorPowerIdle();      // master P^I₀
        const double pNn0 = instance.originatorPowerNetwork();   // master P^N₀
        for (int i = 0; i < N; ++i) {
            const Processor& p = instance.processors()[i];
            const double usedCoeff =  p.powerStartup * p.commStartup
                                    - p.powerIdle * (p.commStartup + p.computeStartup)
                                    + (pNn0 - pNi0) * p.commStartup;
            const double wCoeff    =  p.powerNetwork * p.commRate
                                    - p.powerIdle * (p.commRate + p.computeRate)
                                    + (pNn0 - pNi0) * p.commRate;
            cost[usedCol(i)] = usedCoeff;
            cost[cxCol(i)]   = p.powerIdle;                       // idle base P^I·(Cmax·used)
            for (int k = 0; k < K; ++k) cost[wCol(i, k)] += wCoeff;
        }
        for (int k = 0; k < K; ++k) cost[erCol(k)] = 1.0;        // running energy
        cost[cmaxCol] += pNi0;                                    // master P^I₀·Cmax
    };

    lp.col_cost_.assign(numCols, 0.0);
    if (params_.minimizeCost) {
        if (energyMode) {                            // minimize total energy E
            setEnergyObjective(lp.col_cost_);
        } else {                                     // minimize G = sum f*used + sum l*w
            for (int i = 0; i < N; ++i) {
                const Processor& p = instance.processors()[i];
                for (int k = 0; k < K; ++k) {
                    lp.col_cost_[xCol(i, k)] = p.fixedCost;
                    lp.col_cost_[wCol(i, k)] = p.linearCost;
                }
            }
        }
    } else {
        lp.col_cost_[cmaxCol] = 1.0;                  // minimize Cmax
    }
    lp.col_lower_.assign(numCols, 0.0);
    lp.col_upper_.assign(numCols, kHighsInf);
    lp.integrality_.assign(numCols, HighsVarType::kContinuous);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < K; ++k) {
            lp.col_upper_[xCol(i, k)] = 1.0;
            lp.integrality_[xCol(i, k)] = HighsVarType::kInteger;   // binary
        }
    for (int k = 0; k < K; ++k) {
        lp.col_upper_[alphaCol(k)] = U;
        for (int i = 0; i < N; ++i) lp.col_upper_[wCol(i, k)] = U;
    }
    lp.col_upper_[tCol(0)] = 0.0;              // t_0 = 0 (lower already 0)
    if (addEnergy) {
        lp.col_upper_[cmaxCol] = bigM;         // McCormick needs a finite Cmax bound
        for (int i = 0; i < N; ++i) {
            lp.col_upper_[usedCol(i)] = 1.0;   // used_i in [0,1] (= Σ_k x[i][k], 0/1)
            lp.col_upper_[cxCol(i)]   = bigM;  // cx_i = Cmax·used_i in [0, bigM]
        }
    }

    // ---- rows (built row-wise) ------------------------------------------
    std::vector<double> rlower, rupper;        // per-row bounds
    std::vector<HighsInt> a_start{0}, a_index; // CSR
    std::vector<double> a_value;
    auto addRow = [&](const std::vector<std::pair<int,double>>& terms, double lo, double hi) {
        for (const auto& t : terms) { a_index.push_back(t.first); a_value.push_back(t.second); }
        a_start.push_back(static_cast<HighsInt>(a_index.size()));
        rlower.push_back(lo); rupper.push_back(hi);
    };

    // Each processor used at most once.
    for (int i = 0; i < N; ++i) {
        std::vector<std::pair<int,double>> r;
        for (int k = 0; k < K; ++k) r.push_back({xCol(i, k), 1.0});
        addRow(r, -kHighsInf, 1.0);
    }
    // Each slot holds at most one processor.
    for (int k = 0; k < K; ++k) {
        std::vector<std::pair<int,double>> r;
        for (int i = 0; i < N; ++i) r.push_back({xCol(i, k), 1.0});
        addRow(r, -kHighsInf, 1.0);
    }
    // Memory + empty-slot forcing: alpha_k - sum_i B_i x[i][k] <= 0. The per-proc
    // cap is min(B_i, V): a single installment never exceeds V, so this keeps the
    // coefficient finite when memory is effectively unbounded (HiGHS rejects >1e15).
    for (int k = 0; k < K; ++k) {
        std::vector<std::pair<int,double>> r{{alphaCol(k), 1.0}};
        for (int i = 0; i < N; ++i) {
            const double B = instance.processors()[i].memoryLimit;
            const double effB = std::min((B > 0.0) ? B : V, V);
            r.push_back({xCol(i, k), -effB});
        }
        addRow(r, -kHighsInf, 0.0);
    }
    // alpha_k = sum_i w[i][k].
    for (int k = 0; k < K; ++k) {
        std::vector<std::pair<int,double>> r{{alphaCol(k), 1.0}};
        for (int i = 0; i < N; ++i) r.push_back({wCol(i, k), -1.0});
        addRow(r, 0.0, 0.0);
    }
    // McCormick linearization of w = x*alpha (x binary, alpha in [0,U]).
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < K; ++k) {
            addRow({{wCol(i,k), 1.0}, {xCol(i,k), -U}}, -kHighsInf, 0.0);                  // w <= U x
            addRow({{wCol(i,k), 1.0}, {alphaCol(k), -1.0}}, -kHighsInf, 0.0);              // w <= alpha
            addRow({{wCol(i,k), 1.0}, {alphaCol(k), -1.0}, {xCol(i,k), -U}}, -U, kHighsInf); // w >= alpha - U(1-x)
        }
    // Sequential single-port timing: t_{k+1} >= t_k + sum_i(S_i x + C_i w).
    for (int k = 0; k < K - 1; ++k) {
        std::vector<std::pair<int,double>> r{{tCol(k+1), 1.0}, {tCol(k), -1.0}};
        for (int i = 0; i < N; ++i) {
            r.push_back({xCol(i,k), -instance.processors()[i].commStartup});
            r.push_back({wCol(i,k), -instance.processors()[i].commRate});
        }
        addRow(r, 0.0, kHighsInf);
    }
    // Processing-time pieces (only when hasPieces): proc_k >= a*x[i][k] +
    // b*w[i][k] for each piece of each processor. Both x and w vanish when
    // processor i is not in slot k, so the row binds only for the assigned
    // processor (no big-M needed); for the assigned one it is a + b*alpha_k.
    if (hasPieces) for (int k = 0; k < K; ++k)
        for (int i = 0; i < N; ++i)
            for (const ComputePiece& pc : instance.processors()[i].effectivePieces())
                addRow({{procCol(k), 1.0}, {xCol(i,k), -pc.intercept}, {wCol(i,k), -pc.slope}},
                       0.0, kHighsInf);

    // Makespan: Cmax >= t_k + comm + processing. Processing is proc_k when
    // piecewise, else the inline sum_i(p_i x + A_i w).
    for (int k = 0; k < K; ++k) {
        std::vector<std::pair<int,double>> r{{cmaxCol, 1.0}, {tCol(k), -1.0}};
        if (hasPieces) r.push_back({procCol(k), -1.0});
        for (int i = 0; i < N; ++i) {
            const Processor& p = instance.processors()[i];
            r.push_back({xCol(i,k), -(p.commStartup + (hasPieces ? 0.0 : p.computeStartup))});
            r.push_back({wCol(i,k), -(p.commRate    + (hasPieces ? 0.0 : p.computeRate))});
        }
        addRow(r, 0.0, kHighsInf);
    }
    // Release: Cmax >= r_chosen + processing (only if any release set).
    if (anyRelease) for (int k = 0; k < K; ++k) {
        std::vector<std::pair<int,double>> r{{cmaxCol, 1.0}};
        if (hasPieces) r.push_back({procCol(k), -1.0});
        for (int i = 0; i < N; ++i) {
            const Processor& p = instance.processors()[i];
            r.push_back({xCol(i,k), -(p.releaseTime + (hasPieces ? 0.0 : p.computeStartup))});
            if (!hasPieces) r.push_back({wCol(i,k), -p.computeRate});
        }
        addRow(r, 0.0, kHighsInf);
    }
    // Deadline: completion_k <= d_i when x[i][k]=1, via big-M (per proc, d_i>0).
    if (anyDeadline) for (int k = 0; k < K; ++k) for (int i = 0; i < N; ++i) {
        const Processor& pi = instance.processors()[i];
        if (pi.deadline <= 0.0) continue;
        // comm-bound completion + M*x[i][k] <= d_i + M
        std::vector<std::pair<int,double>> rc{{tCol(k), 1.0}};
        if (hasPieces) rc.push_back({procCol(k), 1.0});
        for (int j = 0; j < N; ++j) {
            const Processor& pj = instance.processors()[j];
            double xc = pj.commStartup + (hasPieces ? 0.0 : pj.computeStartup) + (j == i ? bigM : 0.0);
            rc.push_back({xCol(j,k), xc});
            rc.push_back({wCol(j,k), pj.commRate + (hasPieces ? 0.0 : pj.computeRate)});
        }
        addRow(rc, -kHighsInf, pi.deadline + bigM);
        // release-bound completion + M*x[i][k] <= d_i + M
        if (anyRelease) {
            std::vector<std::pair<int,double>> rr;
            if (hasPieces) rr.push_back({procCol(k), 1.0});
            for (int j = 0; j < N; ++j) {
                const Processor& pj = instance.processors()[j];
                double xc = pj.releaseTime + (hasPieces ? 0.0 : pj.computeStartup) + (j == i ? bigM : 0.0);
                rr.push_back({xCol(j,k), xc});
                if (!hasPieces) rr.push_back({wCol(j,k), pj.computeRate});
            }
            addRow(rr, -kHighsInf, pi.deadline + bigM);
        }
    }
    // Load conservation: sum_k alpha_k = V.
    {
        std::vector<std::pair<int,double>> r;
        for (int k = 0; k < K; ++k) r.push_back({alphaCol(k), 1.0});
        addRow(r, V, V);
    }
    // Energy block (only when E enters the objective or a limit). used_i ties the
    // per-processor "used" flag to its slot assignment; cx_i = Cmax·used_i is the
    // idle-base linearization; er_k is the slot's running energy (convex pieces).
    if (addEnergy) {
        for (int i = 0; i < N; ++i) {
            // used_i - sum_k x[i][k] = 0.
            std::vector<std::pair<int,double>> ru{{usedCol(i), 1.0}};
            for (int k = 0; k < K; ++k) ru.push_back({xCol(i,k), -1.0});
            addRow(ru, 0.0, 0.0);
            // McCormick cx_i = Cmax·used_i (Cmax in [0,bigM], used_i in {0,1}).
            addRow({{cxCol(i), 1.0}, {usedCol(i), -bigM}}, -kHighsInf, 0.0);          // cx <= bigM·used
            addRow({{cxCol(i), 1.0}, {cmaxCol, -1.0}}, -kHighsInf, 0.0);              // cx <= Cmax
            addRow({{cxCol(i), 1.0}, {cmaxCol, -1.0}, {usedCol(i), -bigM}}, -bigM, kHighsInf); // cx >= Cmax-bigM(1-used)
        }
        // er_k >= l + k*alpha_k for each energy piece of the slot's processor:
        // er_k >= sum_i(l*x[i][k] + k*w[i][k]). x and w vanish off-assignment.
        for (int k = 0; k < K; ++k)
            for (int i = 0; i < N; ++i)
                for (const EnergyPiece& ep : instance.processors()[i].effectiveEnergyPieces())
                    addRow({{erCol(k), 1.0}, {xCol(i,k), -ep.intercept}, {wCol(i,k), -ep.slope}},
                           0.0, kHighsInf);
    }

    // Cost limit (bi-criteria second criterion). Linear cost G in the legacy
    // model; total energy E (via the linear form above) in energy mode.
    if (params_.costLimit < std::numeric_limits<double>::infinity()) {
        std::vector<std::pair<int,double>> r;
        if (energyMode) {                            // E(x) <= E-bar
            std::vector<double> ec(numCols, 0.0);
            setEnergyObjective(ec);
            for (int v = 0; v < numCols; ++v) if (ec[v] != 0.0) r.push_back({v, ec[v]});
        } else {                                     // G = sum f*used + sum l*w <= G-bar
            for (int i = 0; i < N; ++i) {
                const Processor& p = instance.processors()[i];
                for (int k = 0; k < K; ++k) {
                    r.push_back({xCol(i,k), p.fixedCost});   // fixed cost when used in any slot
                    r.push_back({wCol(i,k), p.linearCost});  // linear cost on the assigned load
                }
            }
        }
        addRow(r, -kHighsInf, params_.costLimit);
    }
    // Makespan limit (reverse bi-criteria): Cmax <= C-bar.
    if (params_.makespanLimit < std::numeric_limits<double>::infinity()) {
        addRow({{cmaxCol, 1.0}}, -kHighsInf, params_.makespanLimit);
    }
    // Symmetry break: pack used slots to the front.
    for (int k = 0; k < K - 1; ++k) {
        std::vector<std::pair<int,double>> r;
        for (int i = 0; i < N; ++i) { r.push_back({xCol(i,k), 1.0}); r.push_back({xCol(i,k+1), -1.0}); }
        addRow(r, 0.0, kHighsInf);
    }

    lp.num_row_   = static_cast<HighsInt>(rlower.size());
    lp.row_lower_ = rlower;
    lp.row_upper_ = rupper;
    HighsSparseMatrix& A = lp.a_matrix_;
    A.format_  = MatrixFormat::kRowwise;
    A.num_col_ = numCols;
    A.num_row_ = lp.num_row_;
    A.start_   = a_start;
    A.index_   = a_index;
    A.value_   = a_value;

    // ---- solve -----------------------------------------------------------
    Highs highs;
    highs.setOptionValue("output_flag", false);
    if (params_.timeLimitSec > 0.0) highs.setOptionValue("time_limit", params_.timeLimitSec);
    if (params_.mipGap       > 0.0) highs.setOptionValue("mip_rel_gap", params_.mipGap);
    if (highs.passModel(model) != HighsStatus::kOk) { sol.status = SolveStatus::Failure; return sol; }
    highs.run();

    const HighsModelStatus ms = highs.getModelStatus();
    const HighsSolution& s = highs.getSolution();
    const bool hasPrimal = s.value_valid;

    if (ms == HighsModelStatus::kOptimal && hasPrimal) {
        sol.status = SolveStatus::Optimal;
    } else if (hasPrimal && (ms == HighsModelStatus::kTimeLimit ||
                             ms == HighsModelStatus::kIterationLimit ||
                             ms == HighsModelStatus::kSolutionLimit)) {
        sol.status = SolveStatus::Feasible;        // incumbent, not proven optimal
    } else if (ms == HighsModelStatus::kInfeasible) {
        sol.status = SolveStatus::Infeasible; return sol;
    } else if (ms == HighsModelStatus::kUnbounded ||
               ms == HighsModelStatus::kUnboundedOrInfeasible) {
        sol.status = SolveStatus::Unbounded; return sol;
    } else {
        sol.status = SolveStatus::Failure; return sol;
    }

    // ---- translate the solution: read slots in order, skip empty ones ----
    sol.makespan = s.col_value[cmaxCol];
    for (int k = 0; k < K; ++k) {
        int chosen = -1;                            // processor in slot k, if any
        for (int i = 0; i < N; ++i)
            if (s.col_value[xCol(i, k)] > 0.5) { chosen = i; break; }
        if (chosen < 0) continue;                   // empty slot
        sol.sequence.push_back(chosen);
        LoadFragment f;
        f.processorId = chosen;
        f.loadSize    = s.col_value[alphaCol(k)];
        f.commStart   = s.col_value[tCol(k)];
        sol.fragments.push_back(f);
    }
    sol.cost   = scheduleCost(instance, sol.fragments);
    sol.energy = scheduleEnergy(instance, sol.fragments, sol.makespan);  // 0 if no energy model
    return sol;
}

}  // namespace dls
