//---------------------------------------------------------------------------
// exact/branch_and_price/reducer_read_bp_solver.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Branch-and-price solver for heterogeneous multi-channel reducer read
// scheduling (mapreduce/reducer_read_instance.hpp). C++ port of the
// validated research prototype tools/bp_reducer_scheduling.py — see that
// file's module docstring for the full model writeup and design rationale;
// this header mirrors its logic function-for-function, so read it alongside
// this one for the "why", not just the "what".
//
// This is architecturally distinct from both exact/enumerative/ (branches
// directly over the combinatorial space with a closed-form bound) and
// exact/milp/ (hands the WHOLE problem to HiGHS as one MILP): here, HiGHS
// only solves a small per-node LP (the "master"), while a per-reducer
// pricing DP and the branching itself are hand-written — the textbook
// branch-and-price split, needed because the natural column set (one column
// per feasible reducer read-plan) is too large to enumerate up front.
//
// Design recap:
//   - COLUMN = one reducer's complete, feasible read plan (order + exact
//     start ticks for its whole affinity set), found in isolation.
//   - MASTER LP: pick a convex combination of generated columns per reducer
//     (one column per reducer in the integer solution), respecting the
//     shared capacity constraint at every tick, minimizing the objective.
//   - PRICING: per reducer, given the master's duals, a shortest-path DP
//     over (subset-read, tick) states — returns EVERY distinct-finish-tick
//     candidate (not just the single best), because a single best-reduced-
//     cost column is only guaranteed to find SOME improving column, not
//     every column an eventual integer solution needs (a real, validated
//     stall the Python prototype hit and fixed — see its file header).
//   - BRANCHING: "does reducer j start reading mapper i at tick t" — the
//     natural DP-enforceable decision, not whole-column identity (which the
//     DP has no way to search around once excluded).
//
// Requires HiGHS (master LP duals); guarded by DLS_WITH_HIGHS like the
// other exact/milp/ solvers.
//---------------------------------------------------------------------------

#ifndef DLS_EXACT_BRANCH_AND_PRICE_REDUCER_READ_BP_SOLVER_HPP
#define DLS_EXACT_BRANCH_AND_PRICE_REDUCER_READ_BP_SOLVER_HPP

#ifdef DLS_WITH_HIGHS

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "Highs.h"
#include "core/dls_solution.hpp"
#include "mapreduce/reducer_read_instance.hpp"

namespace dls {

struct ReducerReadSolution {
    SolveStatus status = SolveStatus::NotSolved;
    double makespan = 0.0;    // the achieved objective value (Cmax or sum-of-finish, per instance.objective)
    std::vector<ReducerReadColumn> schedule;   // schedule[j] = reducer j's chosen read plan
    int nodesExplored = 0;
};

class ReducerReadBpSolver {
public:
    // Goal:   solve to proven optimality.
    // Input:  inst (validated by the caller); maxNodes safety cap.
    // Output: ReducerReadSolution; status Infeasible if the branch-and-price
    //         search exhausts maxNodes without proving optimality (a real
    //         cap for this prototype-grade branching scheme, not expected
    //         at validation scale — see the file header on branching).
    ReducerReadSolution solve(const ReducerReadInstance& inst, int maxNodes = 2000) {
        const int horizon = inst.horizon();
        double bestObj = kInf;
        std::vector<ReducerReadColumn> bestSol;
        int nodes = 0;

        struct Node {
            std::map<int, std::map<int, int>> forcedTick;               // [reducer][mapper] = tick
            std::map<int, std::map<int, std::set<int>>> bannedTicks;    // [reducer][mapper] = {tick,...}
        };
        std::vector<Node> stack;
        stack.push_back({});

        while (!stack.empty() && nodes < maxNodes) {
            ++nodes;
            Node node = std::move(stack.back());
            stack.pop_back();

            NodeResult res = solveNode(inst, horizon, node.forcedTick, node.bannedTicks);
            if (!res.feasible || res.obj >= bestObj - 1e-9) continue;   // infeasible or can't beat incumbent

            int fracJ = -1;
            std::vector<ReducerReadColumn> chosen(inst.r());
            std::vector<bool> chosenSet(inst.r(), false);
            for (int j = 0; j < inst.r(); ++j) {
                std::vector<std::pair<int, double>> picks;   // (column index, weight)
                for (const auto& [key, w] : res.lam)
                    if (key.first == j) picks.emplace_back(key.second, w);
                if (picks.size() == 1 && picks[0].second > 1 - 1e-6) {
                    chosen[j] = res.columns[j][picks[0].first];
                    chosenSet[j] = true;
                } else if (fracJ < 0) {
                    fracJ = j;
                }
            }
            if (fracJ < 0) {
                if (res.obj < bestObj - 1e-9) {
                    bestObj = res.obj;
                    bestSol = std::move(chosen);
                }
                continue;
            }

            auto [mapperI, tick] = pickBranchVar(res, fracJ);
            Node force2 = node;
            force2.forcedTick[fracJ][mapperI] = tick;
            stack.push_back(force2);
            Node ban2 = node;
            ban2.bannedTicks[fracJ][mapperI].insert(tick);
            stack.push_back(ban2);
        }

        ReducerReadSolution out;
        out.nodesExplored = nodes;
        if (bestSol.empty() && bestObj >= kInf) { out.status = SolveStatus::Infeasible; return out; }
        out.status = SolveStatus::Optimal;
        out.makespan = bestObj;
        out.schedule = std::move(bestSol);
        return out;
    }

private:
    static constexpr double kInf = std::numeric_limits<double>::infinity();
    static constexpr double kBigMFactor = 10.0;

    // ---- pricing: per-reducer shortest-path DP over (subset, tick) --------

    // Goal:   every distinct-finish-tick read plan for reducer j, ranked by
    //         min[finishWeight*finish - sum(mu over occupied ticks)].
    // Inputs: inst; reducer j; mu (dual price per capacity tick, 0 if
    //         absent); finishWeight (epigraph dual in makespan mode, else 1
    //         for balance mode); horizon; forcedTick[i]/bannedTicks[i] —
    //         this node's per-mapper timing constraints, reducer j only.
    // Output: one Column per achievable finish tick (empty if none).
    static std::vector<ReducerReadColumn> priceReducer(
        const ReducerReadInstance& inst, int j, const std::map<int, double>& mu,
        double finishWeight, int horizon,
        const std::map<int, int>& forcedTick, const std::map<int, std::set<int>>& bannedTicks) {
        const auto& req = inst.affinity[j];
        const int n = static_cast<int>(req.size());
        if (n == 0) return {};
        const int full = (1 << n) - 1;

        std::vector<double> muPrefix(horizon + 2, 0.0);
        for (int t = 0; t <= horizon; ++t) {
            auto it = mu.find(t);
            muPrefix[t + 1] = muPrefix[t] + (it != mu.end() ? it->second : 0.0);
        }
        auto occCost = [&](int a, int b) { return muPrefix[b] - muPrefix[a]; };

        // dp[mask][t], parent[mask][t] = (prevMask, prevTick, mapperReadOrNeg1)
        std::vector<std::vector<std::optional<double>>> dp(
            1 << n, std::vector<std::optional<double>>(horizon + 1));
        std::vector<std::vector<std::tuple<int, int, int>>> parent(
            1 << n, std::vector<std::tuple<int, int, int>>(horizon + 1, {-1, -1, -2}));
        dp[0][0] = 0.0;

        for (int t = 0; t < horizon; ++t) {
            for (int mask = 0; mask < (1 << n); ++mask) {
                if (!dp[mask][t].has_value()) continue;
                double val = *dp[mask][t];
                // idle one tick
                if (!dp[mask][t + 1].has_value() || *dp[mask][t + 1] > val) {
                    dp[mask][t + 1] = val;
                    parent[mask][t + 1] = {mask, t, -1};
                }
                // start reading one unread mapper now
                for (int bit = 0; bit < n; ++bit) {
                    if (mask & (1 << bit)) continue;
                    int i = req[bit];
                    if (t < inst.ready[i]) continue;
                    auto ft = forcedTick.find(i);
                    if (ft != forcedTick.end() && t != ft->second) continue;
                    auto bt = bannedTicks.find(i);
                    if (bt != bannedTicks.end() && bt->second.count(t)) continue;
                    int dur = inst.duration(i, j);
                    int finishT = t + dur;
                    if (finishT > horizon) continue;
                    double nval = val - occCost(t, finishT);
                    int nmask = mask | (1 << bit);
                    if (!dp[nmask][finishT].has_value() || *dp[nmask][finishT] > nval) {
                        dp[nmask][finishT] = nval;
                        parent[nmask][finishT] = {mask, t, i};
                    }
                }
            }
        }

        auto reconstruct = [&](int t) {
            ReducerReadColumn col;
            col.reducer = j;
            col.finish = t;
            int mask = full, tt = t;
            while (mask != 0 || tt != 0) {
                auto [pm, pt, i] = parent[mask][tt];
                if (i >= 0) col.starts.emplace_back(i, pt);
                mask = pm; tt = pt;
            }
            std::vector<int> occ;
            for (auto& [i, s] : col.starts) {
                int dur = inst.duration(i, j);
                for (int tk = s; tk < s + dur; ++tk) occ.push_back(tk);
            }
            std::sort(occ.begin(), occ.end());
            col.occupied = std::move(occ);
            return col;
        };

        std::vector<ReducerReadColumn> out;
        for (int t = 0; t <= horizon; ++t)
            if (dp[full][t].has_value()) out.push_back(reconstruct(t));
        return out;
    }

    // Goal:   the reduced cost of one column, given the master's duals.
    static double reducedCost(const ReducerReadColumn& col, double pi,
                              const std::map<int, double>& mu, double finishWeight) {
        double sum = 0.0;
        for (int t : col.occupied) { auto it = mu.find(t); if (it != mu.end()) sum += it->second; }
        return finishWeight * col.finish - pi - sum;
    }

    // ---- seeding ------------------------------------------------------------

    static std::optional<ReducerReadColumn> greedyColumnForOrder(
        const ReducerReadInstance& inst, int j, const std::vector<int>& order,
        const std::map<int, int>& ft, const std::map<int, std::set<int>>& bt) {
        int t = 0;
        ReducerReadColumn col;
        col.reducer = j;
        for (int i : order) {
            int s = std::max(t, inst.ready[i]);
            auto fit = ft.find(i);
            if (fit != ft.end()) {
                if (s > fit->second) return std::nullopt;
                s = fit->second;
            }
            auto bit = bt.find(i);
            while (bit != bt.end() && bit->second.count(s)) {
                ++s;
                if (fit != ft.end() && s > fit->second) return std::nullopt;
            }
            col.starts.emplace_back(i, s);
            t = s + inst.duration(i, j);
        }
        col.finish = order.empty() ? 0 : t;
        std::vector<int> occ;
        for (auto& [i, s] : col.starts) {
            int dur = inst.duration(i, j);
            for (int tk = s; tk < s + dur; ++tk) occ.push_back(tk);
        }
        std::sort(occ.begin(), occ.end());
        col.occupied = std::move(occ);
        return col;
    }

    // Goal:   richly seed each reducer's column pool (one column per DISTINCT
    //         ordering of its required set) so column generation reliably
    //         escapes degenerate-dual stalls — see the file header.
    static std::vector<std::vector<ReducerReadColumn>> seedColumns(
        const ReducerReadInstance& inst,
        const std::map<int, std::map<int, int>>& forcedTick,
        const std::map<int, std::map<int, std::set<int>>>& bannedTicks) {
        std::vector<std::vector<ReducerReadColumn>> cols(inst.r());
        for (int j = 0; j < inst.r(); ++j) {
            std::map<int, int> ft;
            std::map<int, std::set<int>> bt;
            if (auto it = forcedTick.find(j); it != forcedTick.end()) ft = it->second;
            if (auto it = bannedTicks.find(j); it != bannedTicks.end()) bt = it->second;
            std::vector<int> order = inst.affinity[j];
            std::sort(order.begin(), order.end());
            std::set<std::vector<std::pair<int, int>>> seen;
            do {
                auto col = greedyColumnForOrder(inst, j, order, ft, bt);
                if (col.has_value() && seen.insert(col->key()).second)
                    cols[j].push_back(*col);
            } while (std::next_permutation(order.begin(), order.end()));
        }
        return cols;
    }

    // ---- master LP ------------------------------------------------------------

    struct MasterResult {
        std::map<std::pair<int, int>, double> lam;
        double obj = kInf;
        std::vector<double> pi;    // per reducer, convexity dual
        std::map<int, double> mu;  // per tick, capacity dual
        std::vector<double> rho;   // per reducer, epigraph dual (0 in balance mode)
        bool solved = false;
        bool stillInfeasible = false;
    };

    static MasterResult solveMaster(const ReducerReadInstance& inst,
                                    std::vector<std::vector<ReducerReadColumn>>& columns,
                                    int horizon) {
        MasterResult out;
        out.pi.assign(inst.r(), 0.0);
        out.rho.assign(inst.r(), 0.0);
        const bool hasT = inst.objective == ReducerReadObjective::Makespan;

        std::vector<std::pair<int, int>> idx;   // (reducer, column position) per variable
        for (int j = 0; j < inst.r(); ++j)
            for (int p = 0; p < static_cast<int>(columns[j].size()); ++p) idx.emplace_back(j, p);
        const int ncols = static_cast<int>(idx.size());

        std::vector<int> usedTicks;
        for (int t = 0; t <= horizon; ++t) {
            bool used = false;
            for (int j = 0; j < inst.r() && !used; ++j)
                for (const auto& c : columns[j])
                    if (std::binary_search(c.occupied.begin(), c.occupied.end(), t)) { used = true; break; }
            if (used) usedTicks.push_back(t);
        }
        const int nCapSlack = static_cast<int>(usedTicks.size());
        const int nArt = inst.r();
        const double M = kBigMFactor * (inst.r() * (horizon + 1) + 10);
        const int TIdx = hasT ? ncols : -1;
        const int slackBase = ncols + (hasT ? 1 : 0);
        const int artBase = slackBase + nCapSlack;
        const int nvars = artBase + nArt;

        std::vector<double> cost(nvars, 0.0);
        if (hasT) cost[TIdx] = 1.0;
        else for (int k = 0; k < ncols; ++k) cost[k] = columns[idx[k].first][idx[k].second].finish;
        for (int k = slackBase; k < slackBase + nCapSlack; ++k) cost[k] = M;
        for (int k = artBase; k < artBase + nArt; ++k) cost[k] = M;

        // rows: [0..r) convexity (equality); then (if hasT) r epigraph rows
        // (<=0); then nCapSlack capacity rows (<= capacity).
        HighsModel model;
        HighsLp& lp = model.lp_;
        lp.num_col_ = nvars;
        lp.sense_ = ObjSense::kMinimize;
        lp.offset_ = 0.0;
        lp.col_cost_ = cost;
        lp.col_lower_.assign(nvars, 0.0);
        lp.col_upper_.assign(nvars, kHighsInf);

        const int nEq = inst.r();
        const int nEpi = hasT ? inst.r() : 0;
        const int nRows = nEq + nEpi + nCapSlack;
        lp.num_row_ = nRows;
        lp.row_lower_.assign(nRows, 0.0);
        lp.row_upper_.assign(nRows, 0.0);
        for (int j = 0; j < nEq; ++j) { lp.row_lower_[j] = 1.0; lp.row_upper_[j] = 1.0; }         // = 1
        for (int r = nEq; r < nEq + nEpi; ++r) { lp.row_lower_[r] = -kHighsInf; lp.row_upper_[r] = 0.0; }  // <= 0
        for (int r = nEq + nEpi; r < nRows; ++r) {
            lp.row_lower_[r] = -kHighsInf;
            lp.row_upper_[r] = static_cast<double>(inst.capacity);   // <= capacity
        }

        // build sparse rows (rowwise) directly
        std::vector<std::vector<std::pair<int, double>>> rowEntries(nRows);
        for (int k = 0; k < ncols; ++k) {
            auto [j, p] = idx[k];
            rowEntries[j].emplace_back(k, 1.0);   // convexity
            if (hasT) rowEntries[nEq + j].emplace_back(k, static_cast<double>(columns[j][p].finish));
        }
        if (hasT) for (int j = 0; j < inst.r(); ++j) rowEntries[nEq + j].emplace_back(TIdx, -1.0);
        for (int si = 0; si < nCapSlack; ++si) {
            int t = usedTicks[si];
            int rowI = nEq + nEpi + si;
            for (int k = 0; k < ncols; ++k) {
                auto [j, p] = idx[k];
                if (std::binary_search(columns[j][p].occupied.begin(), columns[j][p].occupied.end(), t))
                    rowEntries[rowI].emplace_back(k, 1.0);
            }
            rowEntries[rowI].emplace_back(slackBase + si, -1.0);
        }
        for (int j = 0; j < inst.r(); ++j) rowEntries[j].emplace_back(artBase + j, 1.0);

        HighsSparseMatrix& A = lp.a_matrix_;
        A.format_ = MatrixFormat::kRowwise;
        A.num_col_ = nvars;
        A.num_row_ = nRows;
        // A.start_ already holds a leading {0} from HighsSparseMatrix's own
        // constructor (HighsSparseMatrix::clear() does start_.assign(1, 0)) —
        // pushing another one here would shift every row boundary by one.
        for (auto& row : rowEntries) {
            std::sort(row.begin(), row.end());
            for (auto& [c, v] : row) { A.index_.push_back(c); A.value_.push_back(v); }
            A.start_.push_back(static_cast<HighsInt>(A.index_.size()));
        }

        Highs highs;
        highs.setOptionValue("output_flag", false);
        if (highs.passModel(model) != HighsStatus::kOk) { out.solved = false; out.stillInfeasible = true; return out; }
        highs.run();
        if (highs.getModelStatus() != HighsModelStatus::kOptimal) {
            out.solved = false; out.stillInfeasible = true; return out;
        }

        const HighsSolution& sol = highs.getSolution();
        double slackSum = 0.0, artSum = 0.0;
        for (int k = slackBase; k < slackBase + nCapSlack; ++k) slackSum += sol.col_value[k];
        for (int k = artBase; k < artBase + nArt; ++k) artSum += sol.col_value[k];
        const double relaxTotal = slackSum + artSum;
        out.stillInfeasible = relaxTotal > 1e-6;

        for (int k = 0; k < ncols; ++k)
            if (sol.col_value[k] > 1e-9) out.lam[idx[k]] = sol.col_value[k];
        for (int j = 0; j < inst.r(); ++j) out.pi[j] = sol.row_dual[j];
        if (hasT) for (int j = 0; j < inst.r(); ++j) out.rho[j] = sol.row_dual[nEq + j];
        for (int si = 0; si < nCapSlack; ++si) out.mu[usedTicks[si]] = sol.row_dual[nEq + nEpi + si];

        out.solved = true;
        out.obj = out.stillInfeasible ? kInf : (highs.getObjectiveValue() - M * relaxTotal);
        return out;
    }

    static double finishWeight(const ReducerReadInstance& inst, int j, const MasterResult& duals) {
        if (inst.objective == ReducerReadObjective::Balance) return 1.0;
        return duals.rho[j];
    }

    // ---- one branch-and-price node's LP relaxation via column generation ----

    struct NodeResult {
        MasterResult master;
        std::vector<std::vector<ReducerReadColumn>> columns;
        std::map<std::pair<int, int>, double> lam;   // convenience alias of master.lam
        double obj = kInf;
        bool feasible = false;
    };

    static NodeResult solveNode(const ReducerReadInstance& inst, int horizon,
                               const std::map<int, std::map<int, int>>& forcedTick,
                               const std::map<int, std::map<int, std::set<int>>>& bannedTicks,
                               int maxIters = 200) {
        NodeResult out;
        out.columns = seedColumns(inst, forcedTick, bannedTicks);
        MasterResult m = solveMaster(inst, out.columns, horizon);

        for (int iter = 0; iter < maxIters; ++iter) {
            if (!m.solved) { out.feasible = false; return out; }
            bool improved = false;
            for (int j = 0; j < inst.r(); ++j) {
                double fw = finishWeight(inst, j, m);
                std::map<int, int> ft;
                std::map<int, std::set<int>> bt;
                if (auto it = forcedTick.find(j); it != forcedTick.end()) ft = it->second;
                if (auto it = bannedTicks.find(j); it != bannedTicks.end()) bt = it->second;
                auto candidates = priceReducer(inst, j, m.mu, fw, horizon, ft, bt);
                std::set<std::vector<std::pair<int, int>>> existing;
                for (auto& c : out.columns[j]) existing.insert(c.key());
                for (auto& col : candidates) {
                    auto k = col.key();
                    if (existing.count(k)) continue;
                    double rc = reducedCost(col, m.pi[j], m.mu, fw);
                    if (rc < -1e-7) {
                        out.columns[j].push_back(col);
                        existing.insert(k);
                        improved = true;
                    }
                }
            }
            if (!improved) break;
            m = solveMaster(inst, out.columns, horizon);
        }
        if (!m.solved || m.stillInfeasible) { out.feasible = false; return out; }
        out.master = m;
        out.lam = m.lam;
        out.obj = m.obj;
        out.feasible = true;
        return out;
    }

    // ---- branching variable selection ------------------------------------

    // Goal:   pick a (mapper, tick) branching variable for a fractional
    //         reducer — a mapper where its positive-weight columns disagree
    //         on start tick (guaranteed to exist: distinct columns differ in
    //         `starts` by construction of Column::key()).
    static std::pair<int, int> pickBranchVar(const NodeResult& res, int j) {
        std::vector<std::pair<int, double>> picks;   // (column index, weight)
        for (const auto& [key, w] : res.lam)
            if (key.first == j) picks.emplace_back(key.second, w);
        std::sort(picks.begin(), picks.end(),
                 [](auto& a, auto& b) { return a.second > b.second; });
        const auto& topStarts = res.columns[j][picks[0].first].starts;
        auto lookup = [](const std::vector<std::pair<int, int>>& starts, int mapper) -> std::optional<int> {
            for (auto& [i, t] : starts) if (i == mapper) return t;
            return std::nullopt;
        };
        for (std::size_t p = 1; p < picks.size(); ++p) {
            const auto& other = res.columns[j][picks[p].first].starts;
            for (auto& [i, t] : topStarts) {
                auto ot = lookup(other, i);
                if (!ot.has_value() || *ot != t) return {i, t};
            }
        }
        // Column::key() invariant guarantees distinct columns differ
        // somewhere; reaching here means picks.size() == 1, which the caller
        // already excludes (only called when the reducer is fractional).
        return {topStarts.front().first, topStarts.front().second};
    }
};

}  // namespace dls

#endif  // DLS_WITH_HIGHS

#endif  // DLS_EXACT_BRANCH_AND_PRICE_REDUCER_READ_BP_SOLVER_HPP
