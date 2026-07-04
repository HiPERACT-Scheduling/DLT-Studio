//---------------------------------------------------------------------------
// core/energy_model.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Energy as a first-class scheduling criterion, after Marszałkowski (2020),
// "Scheduling divisible computations with energy constraints" (Poznań UT).
//
// The DLS makespan model tells you WHEN each installment communicates and
// computes; this module tells you how much ENERGY a schedule spends. Two
// contributions are modelled:
//
//   (#1) Piecewise running energy ε(α) — running energy is a convex
//        piecewise-linear function of the processed load, with an in-core
//        piece {l=0, k=k₁} and a steeper out-of-core piece {l=l₂<0, k=k₂}
//        meeting at the available-RAM size ρ. The thesis (§3.3, Tab. 3.4)
//        shows the classic proportional model l·α is badly wrong once a chunk
//        spills out of core, so energyPieces generalises it.
//
//   (#2) Four-state power accounting — every worker moves through Idle (P^I),
//        Startup (P^S), Networking (P^N) and Running states, and the master
//        M₀ draws P^N while distributing and P^I afterwards. Total energy is
//        E = E₀ + Σᵢ(E^S + E^N + E^R + E^I), with the idle term filling each
//        machine's slack up to the makespan T (thesis eq. 5.x / 6.4–6.5).
//
// Everything here is computed in closed form from a realised schedule and the
// instance coefficients (no LP, no timestamps) so it doubles as the ground
// truth the LP energy objective is validated against. It reduces EXACTLY to
// the thesis single-installment formulas when every processor appears once.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_ENERGY_MODEL_HPP
#define DLS_CORE_ENERGY_MODEL_HPP

#include <algorithm>
#include <vector>

#include "dls_instance.hpp"
#include "dls_solution.hpp"

namespace dls {

// Goal:   processing time τ(α) of load q on a processor (convex pieces).
// Input:  p - the processor; q - the processed load (fresh + carried).
// Output: max over effective pieces of (intercept + slope·q); 0 for q <= 0.
inline double processingTime(const Processor& p, double q) {
    if (q <= 0.0) return 0.0;
    double t = 0.0;
    bool first = true;
    for (const ComputePiece& pc : p.effectivePieces()) {
        double v = pc.intercept + pc.slope * q;
        t = first ? v : std::max(t, v);
        first = false;
    }
    return t < 0.0 ? 0.0 : t;
}

// Goal:   running energy ε(α) to process load q on a processor (#1).
// Input:  p - the processor; q - the processed load (fresh + carried).
// Output: max over effective energy pieces of (intercept + slope·q); 0 for
//         q <= 0. With the default single piece {0, linearCost} this is the
//         classic proportional energy l·q.
inline double runningEnergy(const Processor& p, double q) {
    if (q <= 0.0) return 0.0;
    double e = 0.0;
    bool first = true;
    for (const EnergyPiece& ep : p.effectiveEnergyPieces()) {
        double v = ep.intercept + ep.slope * q;
        e = first ? v : std::max(e, v);
        first = false;
    }
    return e < 0.0 ? 0.0 : e;
}

// Goal:   total four-state energy E of a realised schedule (#1 + #2).
// Input:  instance - the DLS problem (coefficients + power rates);
//         fragments - the schedule's installments (loadSize + carriedLoad);
//         makespan  - the schedule length T.
// Output: E = E₀(master) + Σ workers (startup + network + running + idle).
//         Returns 0 when the instance carries no energy model (so legacy
//         instances report energy 0). Each installment is split, consistently
//         with the makespan timing, into a startup span S (at P^S), a transfer
//         span C·α (at P^N) and a run span τ(α+y) (energy ε(α+y)); the rest of
//         T up to each used machine is idle (at P^I).
inline double scheduleEnergy(const DLSInstance& instance,
                             const std::vector<LoadFragment>& fragments,
                             double makespan) {
    if (!instance.usesEnergyModel()) return 0.0;
    const double T = makespan;
    const std::size_t P = instance.numProcessors();

    std::vector<double> busy(P, 0.0);   // total non-idle time per processor
    std::vector<bool>   used(P, false); // processor activated (has positive load)
    double distributeTime = 0.0;        // master's back-to-back distribution span
    double E = 0.0;

    for (const LoadFragment& f : fragments) {
        if (f.processorId < 0 || static_cast<std::size_t>(f.processorId) >= P) continue;
        const Processor& p = instance.processors()[f.processorId];
        const double alpha = f.loadSize;
        if (alpha <= 0.0) continue;                 // unused installment: xᵢ = 0
        const double q        = alpha + f.carriedLoad;  // processed load (fresh + carried)
        const double startup  = p.commStartup;          // S : wake-up / transfer latency
        const double transfer = p.commRate * alpha;     // C·α : load transfer time
        const double run      = processingTime(p, q);   // τ(α+y) : compute time

        E += p.powerStartup * startup;              // E^S : startup energy
        E += p.powerNetwork * transfer;             // E^N : networking energy
        E += runningEnergy(p, q);                   // E^R : running energy (#1)

        busy[f.processorId] += startup + transfer + run;
        used[f.processorId]  = true;
        distributeTime      += startup + transfer;  // master holds the port for S + C·α
    }

    // E^I : each used machine is idle for the rest of the schedule.
    for (std::size_t i = 0; i < P; ++i) {
        if (!used[i]) continue;                     // off machines draw nothing
        const double idle = std::max(0.0, T - busy[i]);
        E += instance.processors()[i].powerIdle * idle;
    }

    // E₀ : master draws P^N while distributing, then P^I until the makespan.
    E += instance.originatorPowerNetwork() * distributeTime;
    E += instance.originatorPowerIdle()    * std::max(0.0, T - distributeTime);

    return E;
}

// Goal:   convenience overload taking a solution directly.
// Input:  instance - the DLS problem; sol - a solved schedule.
// Output: scheduleEnergy(instance, sol.fragments, sol.makespan).
inline double scheduleEnergy(const DLSInstance& instance, const DLSSolution& sol) {
    return scheduleEnergy(instance, sol.fragments, sol.makespan);
}

}  // namespace dls

#endif  // DLS_CORE_ENERGY_MODEL_HPP
