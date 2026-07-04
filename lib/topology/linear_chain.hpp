//---------------------------------------------------------------------------
// topology/linear_chain.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Divisible load on a LINEAR DAISY CHAIN P0 — P1 — ... — P_{N-1} (a non-star
// topology). The whole load V originates at P0; each processor keeps a share αᵢ
// and forwards the remainder downstream over the link into the next processor.
// With front-ends (compute overlaps forwarding), the finish time of Pᵢ is
//
//   fᵢ = (Σ_{k=1..i} C_k · R_k)  +  A_i · αᵢ ,   R_k = Σ_{j>=k} αⱼ ,
//
// i.e. the cumulative link-transfer time for the load that reaches Pᵢ, plus its
// own computation. The optimum minimizes Cmax = maxᵢ fᵢ subject to Σαᵢ = V — and
// by the optimality principle all participating processors finish together.
//
// Because every fᵢ is *linear* in the αⱼ, this is a small LP (variables α₀..α_{N-1}
// and Cmax), solved exactly with the dependency-free CSimplex — the essay's
// "express the topology as an allocation LP" regime. (For a homogeneous chain the
// solution also has the classic closed form; the LP recovers it and generalizes
// to heterogeneous links/rates.)
//
// Model scope: link rate Cᵢ per hop (the link INTO Pᵢ; C₀ unused), compute rate
// Aᵢ per processor, no link startup (S = 0 — the canonical chain model; startups
// would need a used/unused indicator, i.e. a MILP, and are out of scope here).
//---------------------------------------------------------------------------

#ifndef DLS_TOPOLOGY_LINEAR_CHAIN_HPP
#define DLS_TOPOLOGY_LINEAR_CHAIN_HPP

#include <string>
#include <vector>

#include "core/simplex_lp_solver.hpp"   // solveLpViaSimplex
#include "core/dls_solution.hpp"        // SolveStatus

namespace dls {

// One processor in the chain: its compute rate and the rate of the link feeding
// it from upstream (linkRate is ignored for the originator P0).
struct ChainProcessor {
    double computeRate = 0.0;   // Aᵢ : computation time per unit load
    double linkRate    = 0.0;   // Cᵢ : transfer time per unit load over the link into Pᵢ
};

class LinearChainInstance {
public:
    LinearChainInstance() = default;
    LinearChainInstance(std::vector<ChainProcessor> procs, double totalLoad)
        : processors_(std::move(procs)), totalLoad_(totalLoad) {}

    std::vector<ChainProcessor>&       processors()       { return processors_; }
    const std::vector<ChainProcessor>& processors() const { return processors_; }
    std::size_t numProcessors() const { return processors_.size(); }
    double totalLoad() const      { return totalLoad_; }
    void   setTotalLoad(double v)  { totalLoad_ = v; }

    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (processors_.empty()) return fail("chain has no processors");
        if (totalLoad_ <= 0.0)   return fail("totalLoad must be positive");
        for (std::size_t i = 0; i < processors_.size(); ++i) {
            if (processors_[i].computeRate < 0.0 || processors_[i].linkRate < 0.0)
                return fail("processor " + std::to_string(i) + " has a negative rate");
            if (i > 0 && processors_[i].computeRate + processors_[i].linkRate <= 0.0)
                return fail("processor " + std::to_string(i) + " has zero compute and link rate");
        }
        if (processors_[0].computeRate <= 0.0) return fail("originator P0 must have a positive compute rate");
        if (error) error->clear();
        return true;
    }

private:
    std::vector<ChainProcessor> processors_;
    double                      totalLoad_ = 0.0;
};

struct ChainSolution {
    SolveStatus         status   = SolveStatus::NotSolved;
    double              makespan = 0.0;   // Cmax
    std::vector<double> loads;            // αᵢ kept by each processor (chain order)

    double totalAssignedLoad() const { double s = 0.0; for (double a : loads) s += a; return s; }
    bool feasible() const { return status == SolveStatus::Optimal || status == SolveStatus::Feasible; }
};

class LinearChainSolver {
public:
    std::string name() const { return "linear-chain"; }

    // Goal:   optimally partition the load along the chain (minimize Cmax).
    // Input:  the chain instance.
    // Output: a ChainSolution; Optimal with per-processor loads + makespan.
    ChainSolution solve(const LinearChainInstance& inst, std::uint64_t seed = 0) const {
        ChainSolution sol;
        std::string error;
        if (!inst.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }

        const int N = static_cast<int>(inst.numProcessors());
        const auto& P = inst.processors();
        const double V = inst.totalLoad();

        // Cumulative link rate PC[k] = sum_{l=1..k} C_l (PC[0] = 0).
        std::vector<double> PC(N, 0.0);
        for (int k = 1; k < N; ++k) PC[k] = PC[k - 1] + P[k].linkRate;

        // Variables: alpha_0..alpha_{N-1} (0..N-1), Cmax (N).
        const int nVars = N + 1;
        const int CMAX = N;

        std::vector<std::vector<double>> rows;
        std::vector<double> rhs;
        std::vector<double> c(nVars, 0.0);
        auto zero  = [&] { std::fill(c.begin(), c.end(), 0.0); };
        auto addLE = [&](double r) { rows.push_back(c); rhs.push_back(r); };

        // Completion: f_i <= Cmax, with coef(alpha_m in f_i) = PC[min(i,m)] + A_i*[m==i].
        for (int i = 0; i < N; ++i) {
            zero();
            for (int m = 0; m < N; ++m) c[m] = PC[std::min(i, m)];
            c[i] += P[i].computeRate;
            c[CMAX] = -1.0;
            addLE(0.0);
        }
        // Load conservation: sum alpha = V (two LE rows).
        zero(); for (int m = 0; m < N; ++m) c[m] = 1.0;  addLE(V);
        zero(); for (int m = 0; m < N; ++m) c[m] = -1.0; addLE(-V);

        std::vector<double> objective(nVars, 0.0);
        objective[CMAX] = 1.0;                        // minimize Cmax

        SimplexLpResult res = solveLpViaSimplex(nVars, std::move(objective), std::move(rows), std::move(rhs), seed);
        sol.status = res.status;
        if (res.status == SolveStatus::Optimal) {
            sol.makespan = res.values[CMAX];
            sol.loads.assign(res.values.begin(), res.values.begin() + N);
        }
        return sol;
    }

    // Goal:   the finish time fᵢ of processor i for a given load vector (the
    //         closed-form completion expression — used to verify the optimum).
    static double finishTime(const LinearChainInstance& inst, const std::vector<double>& loads, int i) {
        const auto& P = inst.processors();
        const int N = static_cast<int>(inst.numProcessors());
        double comm = 0.0;
        for (int k = 1; k <= i; ++k) {                // cumulative link transfer of load reaching P_i
            double Rk = 0.0; for (int j = k; j < N; ++j) Rk += loads[j];
            comm += P[k].linkRate * Rk;
        }
        return comm + P[i].computeRate * loads[i];
    }
};

}  // namespace dls

#endif  // DLS_TOPOLOGY_LINEAR_CHAIN_HPP
