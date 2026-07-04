//---------------------------------------------------------------------------
// heuristics/fptas/fptas_optv_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// FptasOptVSolver: a fully polynomial-time approximation scheme (FPTAS) for the
// OptV problem on DLS{Cᵢ=0} — maximize the load processed within a deadline T
// when bandwidths are infinite (per-byte communication Cᵢ=0, only startups Sᵢ).
// From Berlińska's thesis §2.2 (Algorithm 2.2), built on the Badics–Boros
// half-product FPTAS (Algorithm 2.1). It is the library's first solver with a
// *provable approximation guarantee*: it returns a load
//
//     V_FPTAS >= (1 − ε)·V_OPT(T).
//
// Structure (Proposition 2.1): for a fixed processor subset the load is
// maximized by activating in non-decreasing Sᵢ·Aᵢ order, and then choosing the
// subset reduces to minimizing the half-product
//
//     f(x) = −Σ pᵢxᵢ + Σ_{i<j} qᵢrⱼ xᵢxⱼ,   pᵢ=(T−Sᵢ)/Aᵢ, qᵢ=Sᵢ, rⱼ=1/Aⱼ,
//
// whose value is −V(x). The FPTAS is a DP over processors in that order, keeping
// one best (smallest partial f = gₖ) representative per geometric bucket of the
// partial Σqᵢxᵢ. The cross term telescopes (Σ_{i<k} qᵢrₖxᵢ = rₖ·Q_{k−1}), so the
// recurrence is O(1): selecting processor k adds −pₖ + rₖ·Q_{k−1} to gₖ.
//
// Input model: a DLSInstance with commRate (Cᵢ) == 0 for all processors,
// computeRate Aᵢ > 0, commStartup Sᵢ; totalLoad is ignored (T is the query).
//---------------------------------------------------------------------------

#ifndef DLS_HEURISTICS_FPTAS_OPTV_SOLVER_HPP
#define DLS_HEURISTICS_FPTAS_OPTV_SOLVER_HPP

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/dls_solver.hpp"

namespace dls {

// Algorithm parameters for the OptV FPTAS.
struct FptasOptVParams {
    double deadline = std::numeric_limits<double>::infinity();  // T (required, finite, > 0)
    double epsilon  = 0.1;   // approximation precision 0 < ε < 1 (V >= (1-ε)·V_OPT)

    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (!(deadline > 0.0 && deadline < std::numeric_limits<double>::infinity()))
            return fail("deadline (T) must be finite and positive");
        if (!(epsilon > 0.0 && epsilon < 1.0)) return fail("epsilon must satisfy 0 < ε < 1");
        if (error) error->clear();
        return true;
    }
};

class FptasOptVSolver : public DLSSolver {
public:
    explicit FptasOptVSolver(FptasOptVParams params) : params_(std::move(params)) {}

    std::string    name() const override     { return "fptas-optv"; }
    SolverCategory category() const override  { return SolverCategory::Heuristic; }  // (1-ε)-approx

    // Goal:   maximize the load processable within T on a DLS{Cᵢ=0} instance.
    // Input:  instance - processors (commRate must be 0; computeRate > 0);
    //         config   - runtime (unused; the FPTAS is deterministic).
    // Output: a DLSSolution; Feasible with the chosen subset, per-processor loads
    //         (Sᵢ·Aᵢ order) and totalAssignedLoad() >= (1-ε)·V_OPT, else Failure.
    DLSSolution solve(const DLSInstance& instance, const SolverConfig& config) override {
        (void)config;
        DLSSolution sol;
        std::string error;
        if (!instance.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }
        if (!params_.validate(&error))  { sol.status = SolveStatus::Failure;    return sol; }
        for (const Processor& p : instance.processors())
            if (p.commRate != 0.0) { sol.status = SolveStatus::Failure; return sol; }  // needs Cᵢ=0

        const double T = params_.deadline;

        // Usable processors: Sᵢ <= T and Aᵢ > 0. Order by non-decreasing Sᵢ·Aᵢ.
        std::vector<int> idx;
        for (int i = 0; i < static_cast<int>(instance.numProcessors()); ++i) {
            const Processor& p = instance.processors()[i];
            if (p.computeRate > 0.0 && p.commStartup <= T) idx.push_back(i);
        }
        const auto& procs = instance.processors();
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
            return procs[a].commStartup * procs[a].computeRate
                 < procs[b].commStartup * procs[b].computeRate;
        });

        const int m = static_cast<int>(idx.size());
        std::vector<char> pick = minimizeHalfProduct(instance, idx, T);   // selected mask

        // Build the schedule: used processors finish at T; Pᵢ starts computing
        // after the cumulative startups up to and including it (single port).
        double cumStartup = 0.0;
        for (int k = 0; k < m; ++k) {
            if (!pick[k]) continue;
            const Processor& p = procs[idx[k]];
            cumStartup += p.commStartup;
            const double load = (T - cumStartup) / p.computeRate;
            if (load <= 0.0) continue;                       // contributes nothing
            LoadFragment f;
            f.processorId   = idx[k];
            f.loadSize      = load;
            f.commStart     = cumStartup - p.commStartup;
            f.commFinish    = cumStartup;
            f.computeStart  = cumStartup;
            f.computeFinish = T;
            sol.fragments.push_back(f);
            sol.sequence.push_back(idx[k]);
        }
        sol.status = SolveStatus::Feasible;     // (1-ε)-approximate, not proven optimal

        // The DP allocates each processor its maximum possible load up to T, with no
        // cap on the total. When T is large, the sum can exceed V (the available work).
        // In that case scale every fragment down so that ΣαΙ = V exactly.
        const double V_avail = instance.totalLoad();
        double totalAssigned = 0.0;
        for (const LoadFragment& f : sol.fragments) totalAssigned += f.loadSize;

        if (!sol.fragments.empty() && totalAssigned > V_avail + 1e-9) {
            const double scale = V_avail / totalAssigned;
            double makespan = 0.0;
            for (LoadFragment& f : sol.fragments) {
                f.loadSize      *= scale;
                f.computeFinish  = f.computeStart + f.loadSize * procs[f.processorId].computeRate;
                makespan = std::max(makespan, f.computeFinish);
            }
            sol.makespan = makespan;
        } else {
            sol.makespan = sol.fragments.empty() ? 0.0 : T;
        }
        return sol;
    }

private:
    FptasOptVParams params_;

    // One DP representative: partial half-product value g, partial Σqᵢxᵢ, mask.
    struct Rep { double g; double Q; std::vector<char> x; };

    // Goal:   minimize the half-product f over subsets of the ordered processors
    //         (Badics–Boros FPTAS), returning the selected mask of length m.
    // Input:  instance, idx (processor indices in Sᵢ·Aᵢ order), T.
    // Output: a 0/1 vector marking the chosen processors.
    std::vector<char> minimizeHalfProduct(const DLSInstance& instance,
                                          const std::vector<int>& idx, double T) const {
        const int m = static_cast<int>(idx.size());
        if (m == 0) return {};
        const auto& procs = instance.processors();
        const double eps   = params_.epsilon;
        const double delta = std::pow(1.0 + eps, 1.0 / m) - 1.0;   // (1+δ)^m = 1+ε
        const double logBucket = std::log1p(delta);               // log(1+δ) > 0

        std::vector<Rep> reps;
        reps.push_back({0.0, 0.0, {}});                           // X₀ = { empty }

        for (int k = 0; k < m; ++k) {
            const Processor& p = procs[idx[k]];
            const double pk = (T - p.commStartup) / p.computeRate;   // pᵢ = (T-Sᵢ)/Aᵢ
            const double qk = p.commStartup;                          // qᵢ = Sᵢ
            const double rk = 1.0 / p.computeRate;                    // rᵢ = 1/Aᵢ

            // Extend every representative by xₖ ∈ {0,1}, then trim by Q-bucket.
            std::unordered_map<long, Rep> best;                      // bucket -> min-g rep
            auto consider = [&](Rep&& c) {
                const long b = (c.Q < 1.0) ? 0
                             : static_cast<long>(std::floor(std::log(c.Q) / logBucket)) + 1;
                auto it = best.find(b);
                if (it == best.end() || c.g < it->second.g) best[b] = std::move(c);
            };
            for (const Rep& r : reps) {
                Rep c0 = r; c0.x.push_back(0);                       // xₖ = 0: g, Q unchanged
                consider(std::move(c0));
                Rep c1{ r.g - pk + rk * r.Q, r.Q + qk, r.x };        // xₖ = 1
                c1.x.push_back(1);
                consider(std::move(c1));
            }
            reps.clear();
            for (auto& kv : best) reps.push_back(std::move(kv.second));
        }

        // Select the representative with the smallest full half-product gₘ = f.
        const Rep* bestRep = &reps.front();
        for (const Rep& r : reps) if (r.g < bestRep->g) bestRep = &r;
        return bestRep->x;
    }
};

}  // namespace dls

#endif  // DLS_HEURISTICS_FPTAS_OPTV_SOLVER_HPP
