//---------------------------------------------------------------------------
// topology/tree.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Divisible load on a MULTI-LEVEL TREE — the natural generalization of both the
// single-level star (this library's base topology) and the linear daisy chain.
// The whole load V originates at the root; every node keeps a share αᵢ and
// forwards each child's subtree-load down the link to that child, sending to its
// children sequentially over a single port (in node-index order). With front-
// ends, a node computes while it forwards, so it finishes at
//
//   fᵢ = rᵢ + Aᵢ·αᵢ ,   rᵢ = r_{parent} + Σ_{siblings c<=i} C_c · L_c ,
//
// where rᵢ is when node i finishes RECEIVING its subtree-load, L_c = Σ_{j in
// subtree(c)} αⱼ, and r_root = 0. Minimizing Cmax = maxᵢ fᵢ subject to Σαᵢ = V is
// an LP (variables αᵢ, rᵢ, Cmax), solved exactly with the dependency-free
// CSimplex — the "express the topology as an allocation LP" regime.
//
// Specializations recovered exactly: a root with N−1 leaf children == the star
// single-port model (S=0); a path (each node one child) == the linear chain.
// Scope: link rate Cᵢ per edge (into node i), compute rate Aᵢ per node, no link
// startup (S=0, as for the chain); the sibling send order is the node index
// order (optimizing it is a combinatorial search, left as follow-up).
//---------------------------------------------------------------------------

#ifndef DLS_TOPOLOGY_TREE_HPP
#define DLS_TOPOLOGY_TREE_HPP

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "core/simplex_lp_solver.hpp"   // solveLpViaSimplex
#include "core/dls_solution.hpp"        // SolveStatus

namespace dls {

// One node of the tree: its compute rate, the rate of the link from its parent,
// and its parent's index (-1 for the root). Parents must precede children
// (parent < index), so node 0 is the root.
struct TreeNode {
    double computeRate = 0.0;   // Aᵢ : computation time per unit load
    double linkRate    = 0.0;   // Cᵢ : transfer time per unit load over the link into node i
    int    parent      = -1;    // index of the parent (< this node's index); -1 for the root
};

class TreeInstance {
public:
    TreeInstance() = default;
    TreeInstance(std::vector<TreeNode> nodes, double totalLoad)
        : nodes_(std::move(nodes)), totalLoad_(totalLoad) {}

    std::vector<TreeNode>&       nodes()       { return nodes_; }
    const std::vector<TreeNode>& nodes() const { return nodes_; }
    std::size_t numNodes() const { return nodes_.size(); }
    double totalLoad() const     { return totalLoad_; }
    void   setTotalLoad(double v) { totalLoad_ = v; }

    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (nodes_.empty()) return fail("tree has no nodes");
        if (totalLoad_ <= 0.0) return fail("totalLoad must be positive");
        if (nodes_[0].parent != -1) return fail("node 0 must be the root (parent = -1)");
        if (nodes_[0].computeRate <= 0.0) return fail("the root must have a positive compute rate");
        for (std::size_t i = 1; i < nodes_.size(); ++i) {
            if (nodes_[i].parent < 0 || nodes_[i].parent >= static_cast<int>(i))
                return fail("node " + std::to_string(i) + " has an invalid parent (must be < its index)");
            if (nodes_[i].computeRate < 0.0 || nodes_[i].linkRate < 0.0)
                return fail("node " + std::to_string(i) + " has a negative rate");
        }
        if (error) error->clear();
        return true;
    }

private:
    std::vector<TreeNode> nodes_;
    double                totalLoad_ = 0.0;
};

struct TreeSolution {
    SolveStatus         status   = SolveStatus::NotSolved;
    double              makespan = 0.0;   // Cmax
    std::vector<double> loads;            // αᵢ kept by each node

    double totalAssignedLoad() const { double s = 0.0; for (double a : loads) s += a; return s; }
    bool feasible() const { return status == SolveStatus::Optimal || status == SolveStatus::Feasible; }
};

class TreeSolver {
public:
    std::string name() const { return "tree"; }

    // Goal:   optimally partition the load over the tree (minimize Cmax).
    // Input:  the tree instance; seed - LP anti-cycling seed.
    // Output: a TreeSolution; Optimal with per-node loads + makespan.
    TreeSolution solve(const TreeInstance& inst, std::uint64_t seed = 0) const {
        TreeSolution sol;
        std::string error;
        if (!inst.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }

        const int N = static_cast<int>(inst.numNodes());
        const auto& nd = inst.nodes();
        std::vector<std::vector<int>> children, subtree;
        buildStructure(nd, children, subtree);

        // Variables: alpha_i (0..N-1), r_i (N..2N-1), Cmax (2N).
        const int nVars = 2 * N + 1;
        const int R = N, CMAX = 2 * N;

        std::vector<std::vector<double>> rows;
        std::vector<double> rhs;
        std::vector<double> c(nVars, 0.0);
        auto zero  = [&] { std::fill(c.begin(), c.end(), 0.0); };
        auto addLE = [&](double r) { rows.push_back(c); rhs.push_back(r); };
        auto addEq = [&](double r) { addLE(r); for (double& x : c) x = -x; addLE(-r); };

        // r_0 = 0.
        zero(); c[R + 0] = 1.0; addEq(0.0);

        // r_i = r_parent + sum_{siblings c <= i} C_c * L_c  (L_c = sum over subtree(c) of alpha).
        for (int i = 1; i < N; ++i) {
            const int p = nd[i].parent;
            zero();
            c[R + i] = 1.0;
            c[R + p] = -1.0;
            for (int sib : children[p]) {
                if (sib > i) break;                          // siblings are sent in index order
                for (int j : subtree[sib]) c[j] -= nd[sib].linkRate;
            }
            addEq(0.0);
        }

        // finish_i = r_i + A_i * alpha_i <= Cmax.
        for (int i = 0; i < N; ++i) {
            zero(); c[R + i] = 1.0; c[i] = nd[i].computeRate; c[CMAX] = -1.0; addLE(0.0);
        }
        // Load conservation: sum alpha = V.
        zero(); for (int i = 0; i < N; ++i) c[i] = 1.0; addEq(inst.totalLoad());

        std::vector<double> objective(nVars, 0.0);
        objective[CMAX] = 1.0;

        SimplexLpResult res = solveLpViaSimplex(nVars, std::move(objective), std::move(rows), std::move(rhs), seed, 2000);
        sol.status = res.status;
        if (res.status == SolveStatus::Optimal) {
            sol.makespan = res.values[CMAX];
            sol.loads.assign(res.values.begin(), res.values.begin() + N);
        }
        return sol;
    }

    // Goal:   the finish time fᵢ of node i for a given load vector (used to
    //         verify the optimum / equal-finish principle).
    static double finishTime(const TreeInstance& inst, const std::vector<double>& loads, int node) {
        const int N = static_cast<int>(inst.numNodes());
        const auto& nd = inst.nodes();
        std::vector<std::vector<int>> children, subtree;
        buildStructure(nd, children, subtree);
        std::vector<double> r(N, 0.0);
        for (int i = 1; i < N; ++i) {                        // ancestors precede i, so r is ready
            const int p = nd[i].parent;
            double add = 0.0;
            for (int sib : children[p]) {
                if (sib > i) break;
                double Lc = 0.0; for (int j : subtree[sib]) Lc += loads[j];
                add += nd[sib].linkRate * Lc;
            }
            r[i] = r[p] + add;
        }
        return r[node] + nd[node].computeRate * loads[node];
    }

private:
    // Goal:   children[p] = children of p in index order; subtree[i] = the nodes
    //         in i's subtree. (Parents precede children, so subtrees fill in
    //         from the highest index down.)
    static void buildStructure(const std::vector<TreeNode>& nd,
                               std::vector<std::vector<int>>& children,
                               std::vector<std::vector<int>>& subtree) {
        const int N = static_cast<int>(nd.size());
        children.assign(N, {});
        subtree.assign(N, {});
        for (int i = 0; i < N; ++i) if (nd[i].parent >= 0) children[nd[i].parent].push_back(i);
        for (int i = N - 1; i >= 0; --i) {
            subtree[i].push_back(i);
            for (int c : children[i]) subtree[i].insert(subtree[i].end(), subtree[c].begin(), subtree[c].end());
        }
    }
};

}  // namespace dls

#endif  // DLS_TOPOLOGY_TREE_HPP
