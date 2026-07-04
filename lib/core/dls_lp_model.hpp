//---------------------------------------------------------------------------
// core/dls_lp_model.hpp
//
// Backend-neutral builder for the DLS per-sequence linear program. Given an
// instance and a FIXED activation sequence, buildDlsLp() produces the LP as a
// plain (objective + "<=" rows) structure that any backend can consume — so
// the intricate constraint logic lives in exactly ONE place and the CSimplex
// and HiGHS evaluators are guaranteed to solve the identical model.
//
// ---- variable layout (L = sequence length) -------------------------------
//   alpha_i = load assigned to installment i        index  i        (0..L-1)
//   t_i     = communication timing of installment i  index  i + L
//   y_i     = "carried" load term for installment i  index  i + 2L
//   proc_i  = processing time of installment i        index  i + 3L  (only when
//             some sequenced processor has a multi-piece convex processing time;
//             omitted otherwise so default instances keep the classic layout)
//   rc_i    = result-return completion of installment i (only when β > 0)
//   Cmax    = makespan (objective)                   last index
// All variables are non-negative (lower bound 0). The proc_i and rc_i blocks are
// appended only when their feature is active, so the default model is unchanged.
// Every row is an inequality  coeffs . x <= rhs.
//
// Piecewise-linear convex processing time: a processor's processing time of
// load q is max over its pieces of (intercept + slope·q). With the implicit
// single piece {computeStartup p, computeRate A} this is the classic p + A·q.
// In the LP it is linearized by proc_i >= a_k + b_k·(alpha_i + y_i) for each
// piece k; minimization drives proc_i to the binding (largest) piece, recovering
// the max. With one piece proc_i is pinned to p + A·(alpha_i+y_i) exactly.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_DLS_LP_MODEL_HPP
#define DLS_CORE_DLS_LP_MODEL_HPP

#include <limits>
#include <vector>

#include "dls_instance.hpp"
#include "dls_solution.hpp"

namespace dls {

// Goal:   compute the schedule cost G = Σ(fᵢ + αᵢ lᵢ) over used processors.
// Input:  instance and the solution's load fragments.
// Output: total cost (fixed cost once per distinct processor used, plus linear
//         cost times the load assigned to it).
inline double scheduleCost(const DLSInstance& instance,
                           const std::vector<LoadFragment>& fragments) {
    double cost = 0.0;
    std::vector<bool> counted(instance.numProcessors(), false);  // fixed cost once each
    for (const LoadFragment& f : fragments) {
        const Processor& p = instance.processors()[f.processorId];
        cost += p.linearCost * f.loadSize;
        if (!counted[f.processorId]) { cost += p.fixedCost; counted[f.processorId] = true; }
    }
    return cost;
}

// Outcome of building the model for a sequence.
enum class LpBuildStatus {
    Ok,               // model built
    EmptySequence,    // nothing to schedule (treat as Infeasible)
    IndexOutOfRange   // a sequence entry is not a valid processor (caller bug)
};

// The DLS LP in backend-neutral form (minimize x[cmaxVar] subject to the rows).
struct DlsLpModel {
    LpBuildStatus status = LpBuildStatus::EmptySequence;
    int numVars = 0;                          // number of decision variables
    int cmaxVar = 0;                          // index of the makespan variable (objective)
    std::vector<int> alphaVars;               // alphaVars[i] = load variable of installment i
    std::vector<int> timeVars;                // timeVars[i]  = timing variable of installment i
    std::vector<int> carryVars;               // carryVars[i] = y_i (carried load) variable
    std::vector<int> procVars;                // procVars[i]  = proc_i (run time), or -1 if absent
    std::vector<int> energyVars;              // energyVars[i]= e_i (running energy), or -1 if absent
    bool hasEnergy = false;                   // energy block present (instance.usesEnergyModel())
    // Total energy as a linear form over the LP variables (valid for the fixed
    // sequence): E(x) = energyObjConst + Σ energyObjCoeffs[v]·x[v]. Filled only
    // in energy mode; used both as the MinCost objective and the energy-limit row.
    std::vector<double> energyObjCoeffs;
    double              energyObjConst = 0.0;
    std::vector<std::vector<double>> rows;    // each row: coeffs over numVars
    std::vector<double> rhs;                  // row i constraint: rows[i] . x <= rhs[i]
};

// Goal:   build the DLS per-sequence LP (objective + "<=" rows) for a fixed
//         activation sequence. Mirrors the single-port, multi-installment,
//         affine-cost model with memory limits.
// Input:  instance - the DLS problem; sequence - ordered 0-based processor
//         indices, one per installment; freeLoad - when true the load is NOT
//         fixed to V (the Σαᵢ = V conservation rows are dropped) so a caller can
//         maximize Σαᵢ instead (the OptV / MaxLoad objective).
// Output: a DlsLpModel; on status==Ok the rows/rhs/var indices are filled.
inline DlsLpModel buildDlsLp(const DLSInstance& instance, const std::vector<int>& sequence,
                            double costLimit = std::numeric_limits<double>::infinity(),
                            double makespanLimit = std::numeric_limits<double>::infinity(),
                            bool freeLoad = false) {
    DlsLpModel m;

    const int L = static_cast<int>(sequence.size());          // installments
    const int P = static_cast<int>(instance.numProcessors()); // worker count
    if (L == 0 || P == 0) { m.status = LpBuildStatus::EmptySequence; return m; }
    for (int id : sequence)
        if (id < 0 || id >= P) { m.status = LpBuildStatus::IndexOutOfRange; return m; }

    // Result return (model A): when β > 0, add L return-completion variables
    // rc_i and shift Cmax after them. Default (β = 0) keeps the original layout.
    const double beta      = instance.resultFraction();
    const bool   hasReturn = beta > 0.0;

    // Active only when some sequenced processor has a multi-piece processing
    // time. Single-piece processors keep the classic inline term, so default
    // instances produce a byte-identical model (same vars, same rows).
    bool hasPieces = false;
    for (int i = 0; i < L; ++i)
        if (instance.processors()[sequence[i]].effectivePieces().size() > 1) { hasPieces = true; break; }

    // Energy mode: the proc_i (run-time) block is needed so the idle-energy term
    // T − busy can be expressed, and an e_i (running energy) block is added. The
    // run-time variables also let single-piece time instances reuse one path.
    const bool hasEnergy = instance.usesEnergyModel();
    const bool useProc   = hasPieces || hasEnergy;   // proc_i present?

    // Variable-block offsets (blocks appended only when their feature is active).
    int off = 0;
    const int ALPHA = off; off += L;          // alpha_i at ALPHA + i
    const int T     = off; off += L;          // t_i     at T + i
    const int Y     = off; off += L;          // y_i     at Y + i
    const int PROC  = useProc  ? off : -1; if (useProc)  off += L;  // proc_i at PROC + i
    const int ENER  = hasEnergy ? off : -1; if (hasEnergy) off += L; // e_i   at ENER + i
    const int RC    = hasReturn ? off : -1; if (hasReturn) off += L;  // rc_i   at RC + i
    const int CMAX  = off; off += 1;          // makespan
    const int nVars = off;

    m.numVars   = nVars;
    m.cmaxVar   = CMAX;
    m.hasEnergy = hasEnergy;
    m.alphaVars.resize(L);
    m.timeVars.resize(L);
    m.carryVars.resize(L);
    m.procVars.assign(L, -1);
    m.energyVars.assign(L, -1);
    for (int i = 0; i < L; ++i) {
        m.alphaVars[i] = ALPHA + i;
        m.timeVars[i]  = T + i;
        m.carryVars[i] = Y + i;
        if (useProc)   m.procVars[i]   = PROC + i;
        if (hasEnergy) m.energyVars[i] = ENER + i;
    }

    // appear[p][0] = times processor p occurs; appear[p][1..count] = positions.
    std::vector<std::vector<int>> appear(P, std::vector<int>(L + 1, 0));
    for (int i = 0; i < L; ++i) {
        int p = sequence[i];
        appear[p][0]++;
        appear[p][appear[p][0]] = i;
    }

    std::vector<double> c(nVars, 0.0);  // reusable row scratch
    auto zero  = [&] { std::fill(c.begin(), c.end(), 0.0); };
    auto addLE = [&](double rhs) { m.rows.push_back(c); m.rhs.push_back(rhs); };
    auto proc  = [&](int i) -> const Processor& { return instance.processors()[sequence[i]]; };

    // (1) Communication dependencies: t_{i+1} >= t_i + S + C*alpha_i.
    for (int i = 0; i < L - 1; ++i) {
        const Processor& p = proc(i);
        zero();
        c[ALPHA + i] = p.commRate;
        c[T + i]     = 1.0;
        c[T + i + 1] = -1.0;            // == t_{i+1}
        if (i == 0) c[T + i] = 0.0;     // t_0 handled by border constraint
        addLE(-p.commStartup);
    }

    // (3) Dependencies on the previous installment of the same processor.
    for (int i = 0; i < L; ++i) {
        const Processor& p = proc(i);
        int pid = sequence[i];
        zero();
        if (appear[pid][1] == i) {
            c[Y + i] = -1.0; addLE(0.0);   // first appearance: pin y_i = 0
            c[Y + i] =  1.0; addLE(0.0);
        } else {
            int j = 2;
            for (; j <= appear[pid][0]; ++j)
                if (appear[pid][j] == i) break;
            --j;
            int prev = appear[pid][j];      // immediately preceding appearance

            c[Y + prev]     = p.computeRate;
            c[Y + i]        = -p.computeRate;
            c[ALPHA + prev] = p.computeRate + p.commRate;
            c[T + i]        = -1.0;
            c[T + prev]     = 1.0;
            addLE(-p.commStartup);
        }
    }

    // (4) Buffer limit: carried + current installment must fit in memory.
    for (int i = 0; i < L; ++i) {
        const Processor& p = proc(i);
        zero();
        c[Y + i]     = 1.0;
        c[ALPHA + i] = 1.0;
        addLE(p.memoryLimit);
    }

    // (5) Carried-load term is non-negative: -y_i <= 0.
    for (int i = 0; i < L; ++i) {
        zero();
        c[Y + i] = -1.0;
        addLE(0.0);
    }

    // (6) All load is distributed: sum(alpha_i) == V, as two LE rows. Dropped
    //     when freeLoad (the MaxLoad/OptV objective leaves Σαᵢ free to maximize).
    if (!freeLoad) {
        zero();
        for (int i = 0; i < L; ++i) c[ALPHA + i] = 1.0;
        addLE(instance.totalLoad());
        zero();
        for (int i = 0; i < L; ++i) c[ALPHA + i] = -1.0;
        addLE(-instance.totalLoad());
    }

    // (6b) Processing-time pieces (only when useProc): for each installment i
    //      and each piece (a,b) of its processor,  proc_i >= a + b*(alpha_i+y_i).
    if (useProc) {
        for (int i = 0; i < L; ++i) {
            for (const ComputePiece& pc : proc(i).effectivePieces()) {
                zero();
                c[ALPHA + i] = pc.slope;
                c[Y + i]     = pc.slope;
                c[PROC + i]  = -1.0;
                addLE(-pc.intercept);
            }
        }
    }

    // (6c) Running-energy pieces (only in energy mode): for each installment i
    //      and each energy piece (l,k) of its processor,
    //      e_i >= l + k*(alpha_i+y_i). Minimizing energy drives e_i to the
    //      binding (largest) piece, recovering ε(α+y) = max_k(l_k+k_k(α+y)).
    if (hasEnergy) {
        for (int i = 0; i < L; ++i) {
            for (const EnergyPiece& ep : proc(i).effectiveEnergyPieces()) {
                zero();
                c[ALPHA + i] = ep.slope;
                c[Y + i]     = ep.slope;
                c[ENER + i]  = -1.0;
                addLE(-ep.intercept);
            }
        }
    }

    // (7) Each installment finishes computing before Cmax. The compute term is
    //     proc_i when piecewise, else the inline p + A*(alpha_i+y_i).
    //     Cmax >= t_i + S + C*alpha_i + (compute).
    for (int i = 0; i < L; ++i) {
        const Processor& p = proc(i);
        zero();
        c[T + i] = 1.0;
        c[CMAX]  = -1.0;
        if (useProc) {
            c[ALPHA + i] = p.commRate;
            c[PROC + i]  = 1.0;
            addLE(-p.commStartup);
        } else {
            c[ALPHA + i] = p.commRate + p.computeRate;
            c[Y + i]     = p.computeRate;
            addLE(-(p.commStartup + p.computeStartup));
        }
    }

    // (7') Availability constraints (release r, deadline d). Rows are added only
    //      for processors that set them, so default instances are unchanged.
    //      For single-installment (y_i = 0) these match the thesis SLSD model.
    for (int i = 0; i < L; ++i) {
        const Processor& p = proc(i);
        const bool hasDeadline = p.deadline > 0.0;   // <=0 means no deadline

        // Release: a processor cannot finish before r + compute.
        //   Cmax >= r + (compute).
        if (p.releaseTime > 0.0) {
            zero();
            c[CMAX] = -1.0;
            if (useProc) {
                c[PROC + i] = 1.0;
                addLE(-p.releaseTime);
            } else {
                c[ALPHA + i] = p.computeRate;
                c[Y + i]     = p.computeRate;
                addLE(-(p.releaseTime + p.computeStartup));
            }
        }
        // Deadline on the communication-bound completion:
        //   t_i + S + C*alpha_i + (compute) <= d.
        if (hasDeadline) {
            zero();
            c[T + i] = 1.0;
            if (useProc) {
                c[ALPHA + i] = p.commRate;
                c[PROC + i]  = 1.0;
                addLE(p.deadline - p.commStartup);
            } else {
                c[ALPHA + i] = p.commRate + p.computeRate;
                c[Y + i]     = p.computeRate;
                addLE(p.deadline - (p.commStartup + p.computeStartup));
            }
        }
        // Deadline on the release-bound completion (only binds when r > 0):
        //   r + (compute) <= d.
        if (hasDeadline && p.releaseTime > 0.0) {
            zero();
            if (useProc) {
                c[PROC + i] = 1.0;
                addLE(p.deadline - p.releaseTime);
            } else {
                c[ALPHA + i] = p.computeRate;
                c[Y + i]     = p.computeRate;
                addLE(p.deadline - (p.releaseTime + p.computeStartup));
            }
        }
    }

    // (8) Border condition: t_0 == 0, as two LE rows.
    zero();
    c[T + 0] = 1.0;  addLE(0.0);
    c[T + 0] = -1.0; addLE(0.0);

    // (8b) Energy objective as a linear form over the LP variables (energy mode
    //      only). For a fixed sequence the four-state total energy is linear in
    //      (alpha_i, e_i, proc_i, Cmax): the per-machine "used" flags x_p are
    //      constant, so startup energy and the master's distribution term reduce
    //      to constants/linear parts. This mirrors core/energy_model.hpp exactly
    //      (running e_i, network P^N·C·alpha, idle P^I·(T − busy), master E_0).
    if (hasEnergy) {
        std::vector<double>& eo = m.energyObjCoeffs;
        eo.assign(nVars, 0.0);
        double konst = 0.0;
        const double pNi = instance.originatorPowerIdle();      // master P^I_0
        const double pNn = instance.originatorPowerNetwork();   // master P^N_0
        for (int i = 0; i < L; ++i) {
            const Processor& p = proc(i);
            eo[ENER + i]  += 1.0;                       // running energy e_i
            eo[ALPHA + i] += p.powerNetwork * p.commRate;        // network P^N·C·alpha
            konst         += p.powerStartup * p.commStartup;     // startup P^S·S (per installment)
            // idle of this installment's machine: −P^I·busy contributions
            eo[ALPHA + i] += -p.powerIdle * p.commRate;          // −P^I·(C·alpha)
            eo[PROC + i]  += -p.powerIdle;                       // −P^I·run
            konst         += -p.powerIdle * p.commStartup;       // −P^I·S
            // master distribution term (P^N_0 − P^I_0)·(S + C·alpha)
            eo[ALPHA + i] += (pNn - pNi) * p.commRate;
            konst         += (pNn - pNi) * p.commStartup;
        }
        // idle base P^I·T, once per distinct used machine, plus master P^I_0·T
        for (int pp = 0; pp < P; ++pp)
            if (appear[pp][0] >= 1) eo[CMAX] += instance.processors()[pp].powerIdle;
        eo[CMAX] += pNi;
        m.energyObjConst = konst;
    }

    // (9) Cost limit (bi-criteria second criterion). In the legacy linear-cost
    //     model "min Cmax s.t. G <= G-bar": the fixed cost F of distinct
    //     processors used is constant for a fixed sequence, so the row constrains
    //     only the linear part sum_i l_{d_i} alpha_i <= G-bar - F. In energy mode
    //     the limit applies to total energy E(x) <= G-bar using the linear form
    //     above. Added only when a finite limit is given (default: none).
    if (costLimit < std::numeric_limits<double>::infinity()) {
        if (hasEnergy) {
            m.rows.push_back(m.energyObjCoeffs);
            m.rhs.push_back(costLimit - m.energyObjConst);
        } else {
            double F = 0.0;                              // fixed cost of distinct procs used
            for (int p = 0; p < P; ++p)
                if (appear[p][0] >= 1) F += instance.processors()[p].fixedCost;
            zero();
            for (int i = 0; i < L; ++i) c[ALPHA + i] = proc(i).linearCost;
            addLE(costLimit - F);
        }
    }

    // (10) Makespan limit (bi-criteria "min G s.t. Cmax <= C-bar"): Cmax <= C-bar.
    if (makespanLimit < std::numeric_limits<double>::infinity()) {
        zero();
        c[CMAX] = 1.0;
        addLE(makespanLimit);
    }

    // (11) Result return — MODEL A: non-overlapping, FIFO collection. After all
    //      distribution finishes, the master collects results in installment
    //      order over the single port. rc_i = return completion of installment i;
    //      return time of installment i is S_i + C_i * beta * alpha_i. The
    //      makespan becomes the last return completion.
    if (hasReturn) {
        for (int i = 0; i < L; ++i) {
            const Processor& pr = proc(i);
            // rc_i >= comp_i(comm) + return_i, with return_i = S + C*beta*alpha_i.
            zero();
            c[RC + i] = -1.0;
            c[T + i]  = 1.0;
            if (useProc) {
                c[ALPHA + i] = pr.commRate + pr.commRate * beta;
                c[PROC + i]  = 1.0;
                addLE(-(2.0 * pr.commStartup));
            } else {
                c[ALPHA + i] = pr.commRate + pr.computeRate + pr.commRate * beta;
                c[Y + i]     = pr.computeRate;
                addLE(-(2.0 * pr.commStartup + pr.computeStartup));
            }
            // rc_i >= comp_i(release) + return_i  (only when r_i > 0).
            if (pr.releaseTime > 0.0) {
                zero();
                c[RC + i] = -1.0;
                if (useProc) {
                    c[ALPHA + i] = pr.commRate * beta;
                    c[PROC + i]  = 1.0;
                    addLE(-(pr.releaseTime + pr.commStartup));
                } else {
                    c[ALPHA + i] = pr.computeRate + pr.commRate * beta;
                    c[Y + i]     = pr.computeRate;
                    addLE(-(pr.releaseTime + pr.computeStartup + pr.commStartup));
                }
            }
            // rc_i >= rc_{i-1} + return_i  (FIFO single-port serialization).
            if (i >= 1) {
                zero();
                c[RC + i]     = -1.0;
                c[RC + i - 1] = 1.0;
                c[ALPHA + i]  = pr.commRate * beta;
                addLE(-pr.commStartup);
            }
            // Cmax >= rc_i.
            zero();
            c[RC + i] = 1.0;
            c[CMAX]   = -1.0;
            addLE(0.0);
        }
        // rc_0 >= T_dist + return_0, with T_dist = sum_j (S_j + C_j alpha_j).
        zero();
        c[RC + 0] = -1.0;
        double startupSum = 0.0;
        for (int j = 0; j < L; ++j) { c[ALPHA + j] += proc(j).commRate; startupSum += proc(j).commStartup; }
        c[ALPHA + 0] += proc(0).commRate * beta;
        addLE(-(startupSum + proc(0).commStartup));
    }

    m.status = LpBuildStatus::Ok;
    return m;
}

}  // namespace dls

#endif  // DLS_CORE_DLS_LP_MODEL_HPP
