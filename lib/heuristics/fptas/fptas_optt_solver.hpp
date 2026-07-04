//---------------------------------------------------------------------------
// heuristics/fptas/fptas_optt_solver.hpp
//
// FptasOptTSolver: a fully polynomial-time approximation scheme (FPTAS) for the
// OptT problem on DLS{Cᵢ=0} — minimize the time to process a given load V when
// bandwidths are infinite (per-byte communication Cᵢ=0, only startups Sᵢ). From
// Berlińska's thesis §2.3 (Algorithm 2.4). It guarantees
//
//     T_FPTAS <= (1 + ε)·T_OPT(V).
//
// Method (dual approximation + binary search). For a deadline T the OptV FPTAS
// (Algorithm 2.2, our FptasOptVSolver) tells how much load fits; DUAL-OPT-V
// (Algorithm 2.3) inflates that estimate to (1+ε)·V_FPTAS(T, ε/2), which is a
// superoptimal (feasible) load gauge. FPTAS-OPT-T binary-searches the deadline
// in [0, S_max + V·A_max]: if the dual gauge cannot reach V(1+ε) the deadline
// was too small (raise `lower`), else lower `upper`. The interval is shrunk
// relative to the positive lower bound LoBo = V·A_min/m until short enough; the
// answer is `upper`, with the processor subset from a final FPTAS-OPT-V call.
//
// We then report the *exact* finishing time of that subset for load V (an equal-
// completion closed form), which is no worse than `upper` and still within the
// (1+ε) guarantee, plus the matching loads (summing to exactly V).
//---------------------------------------------------------------------------

#ifndef DLS_HEURISTICS_FPTAS_OPTT_SOLVER_HPP
#define DLS_HEURISTICS_FPTAS_OPTT_SOLVER_HPP

#include <algorithm>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"
#include "heuristics/fptas/fptas_optv_solver.hpp"

namespace dls {

// Algorithm parameters for the OptT FPTAS. The load V is the instance totalLoad.
struct FptasOptTParams {
    double epsilon = 0.1;   // approximation precision 0 < ε < 1 (T <= (1+ε)·T_OPT)

    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (!(epsilon > 0.0 && epsilon < 1.0)) return fail("epsilon must satisfy 0 < ε < 1");
        if (error) error->clear();
        return true;
    }
};

class FptasOptTSolver : public DLSSolver {
public:
    explicit FptasOptTSolver(FptasOptTParams params) : params_(std::move(params)) {}

    std::string    name() const override     { return "fptas-optt"; }
    SolverCategory category() const override  { return SolverCategory::Heuristic; }  // (1+ε)-approx

    // Goal:   minimize the time to process the instance's load V on DLS{Cᵢ=0}.
    // Input:  instance - processors (commRate must be 0; computeRate > 0) and
    //         totalLoad V; config - runtime (unused; deterministic).
    // Output: a DLSSolution; Feasible with the chosen subset, per-processor loads
    //         summing to V, and makespan <= (1+ε)·T_OPT(V), else Failure.
    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override {
        (void)config;
        DLSSolution sol;
        std::string error;
        if (!instance.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }
        if (!params_.validate(&error))  { sol.status = SolveStatus::Failure;    return sol; }
        for (const Processor& p : instance.processors())
            if (p.commRate != 0.0) { sol.status = SolveStatus::Failure; return sol; }  // needs Cᵢ=0

        const double V   = instance.totalLoad();
        const double eps = params_.epsilon;
        const int    m   = static_cast<int>(instance.numProcessors());

        double aMax = 0.0, aMin = std::numeric_limits<double>::infinity(), sMax = 0.0;
        for (const Processor& p : instance.processors()) {
            aMax = std::max(aMax, p.computeRate);
            aMin = std::min(aMin, p.computeRate);
            sMax = std::max(sMax, p.commStartup);
        }

        double upper = sMax + V * aMax;          // trivial: one processor does all of V
        double lower = 0.0;
        const double loBo = V * aMin / m;        // positive lower bound on T_OPT
        const double tol  = eps * (1.0 - eps) / (2.0 - eps) * loBo;   // stop width (Alg. 2.4)

        // Binary search on the deadline using the dual gauge (1+ε)·V_FPTAS(Tp, ε/2).
        while (upper - lower > tol) {
            const double Tp = 0.5 * (upper + lower);
            const double vDual = (1.0 + eps) * maxLoadFor(instance, Tp, eps / 2.0);
            if (vDual < V * (1.0 + eps)) lower = Tp;   // deadline too small
            else                          upper = Tp;
        }

        // Final subset from FPTAS-OPT-V at the accepted deadline `upper`.
        FptasOptVParams fv; fv.deadline = upper; fv.epsilon = eps / 2.0;
        DLSSolution sub = FptasOptVSolver(fv).solve(instance, SolverConfig{});
        if (!sub.feasible() || sub.sequence.empty()) { sol.status = SolveStatus::Failure; return sol; }

        // Compute the equal-completion finishing time Tprime for load V on the
        // selected subset (Σ (Tprime − cumᵢ)/Aᵢ = V). The binary search may
        // converge to a deadline where OptV selects a processor that is infeasible
        // at the smaller Tprime (cumulative startup exceeds Tprime). Trim those
        // processors iteratively and recompute until all loads are positive.
        std::vector<int> used = sub.sequence;
        std::vector<double> cumOf;
        double Tprime = 0.0;
        bool trimmed = true;
        while (trimmed && !used.empty()) {
            trimmed = false;
            cumOf.clear();
            double cum = 0.0, sumInvA = 0.0, sumCumOverA = 0.0;
            for (int pid : used) {
                const Processor& p = instance.processors()[pid];
                cum         += p.commStartup;
                cumOf.push_back(cum);
                sumInvA     += 1.0 / p.computeRate;
                sumCumOverA += cum / p.computeRate;
            }
            Tprime = (V + sumCumOverA) / sumInvA;
            std::vector<int> keep;
            for (std::size_t k = 0; k < used.size(); ++k) {
                const Processor& p = instance.processors()[used[k]];
                if ((Tprime - cumOf[k]) / p.computeRate > 1e-9)
                    keep.push_back(used[k]);
                else
                    trimmed = true;
            }
            used = keep;
        }
        if (used.empty()) { sol.status = SolveStatus::Failure; return sol; }

        for (std::size_t k = 0; k < used.size(); ++k) {
            const int pid = used[k];
            const Processor& p = instance.processors()[pid];
            LoadFragment f;
            f.processorId   = pid;
            f.loadSize      = (Tprime - cumOf[k]) / p.computeRate;
            f.commStart     = cumOf[k] - p.commStartup;
            f.commFinish    = cumOf[k];
            f.computeStart  = cumOf[k];
            f.computeFinish = Tprime;
            sol.fragments.push_back(f);
            sol.sequence.push_back(pid);
        }
        sol.status   = SolveStatus::Feasible;     // (1+ε)-approximate
        sol.makespan = Tprime;
        return sol;
    }

private:
    FptasOptTParams params_;

    // Goal:   V_FPTAS(T, eps) — the OptV FPTAS load for deadline T (Algorithm 2.2).
    static double maxLoadFor(const DLSInstance& instance, double T, double eps) {
        FptasOptVParams fv; fv.deadline = T; fv.epsilon = eps;
        return FptasOptVSolver(fv).solve(instance, SolverConfig{}).totalAssignedLoad();
    }
};

}  // namespace dls

#endif  // DLS_HEURISTICS_FPTAS_OPTT_SOLVER_HPP
