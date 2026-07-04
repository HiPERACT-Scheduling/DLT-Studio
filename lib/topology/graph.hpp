//---------------------------------------------------------------------------
// topology/graph.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Divisible load on a GENERAL GRAPH (mesh / arbitrary network) — the most
// general regime-2 topology. The load originates at a source node and must reach
// every computing node over the graph's edges.
//
// Principle: in the single-port store-and-forward model an optimal divisible-load
// distribution uses a *spanning arborescence* rooted at the source (extra paths
// only add contention; a tree suffices for the optimum). So the general graph
// reduces to "choose the best rooted spanning tree, then solve it with the tree
// allocation LP" (topology/tree.hpp). For small graphs this is solved exactly by
// enumerating the arborescences and keeping the one with the smallest makespan;
// it reduces to TreeSolver when the graph already is a tree.
//
// Scope: undirected edges with a per-edge rate Cₑ, compute rate Aᵢ per node, no
// link startup (S = 0, as for the chain/tree). The sibling send order within the
// chosen tree is the node-index order (as in TreeSolver). Enumeration is capped,
// so this is for small graphs / validation, not large meshes.
//---------------------------------------------------------------------------

#ifndef DLS_TOPOLOGY_GRAPH_HPP
#define DLS_TOPOLOGY_GRAPH_HPP

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "core/dls_solution.hpp"   // SolveStatus
#include "topology/tree.hpp"

namespace dls {

struct GraphEdge { int u = 0; int v = 0; double rate = 0.0; };   // undirected link u—v with rate Cₑ

class GraphInstance {
public:
    GraphInstance() = default;
    GraphInstance(std::vector<double> computeRates, std::vector<GraphEdge> edges, double totalLoad)
        : computeRates_(std::move(computeRates)), edges_(std::move(edges)), totalLoad_(totalLoad) {}

    std::vector<double>&       computeRates()       { return computeRates_; }
    const std::vector<double>& computeRates() const { return computeRates_; }
    std::vector<GraphEdge>&        edges()       { return edges_; }
    const std::vector<GraphEdge>&  edges() const { return edges_; }
    std::size_t numNodes() const { return computeRates_.size(); }
    double totalLoad() const      { return totalLoad_; }
    void   setTotalLoad(double v)  { totalLoad_ = v; }

    // Goal: well-formedness + connectivity from the source (node 0).
    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        const int N = static_cast<int>(computeRates_.size());
        if (N == 0) return fail("graph has no nodes");
        if (totalLoad_ <= 0.0) return fail("totalLoad must be positive");
        if (computeRates_[0] <= 0.0) return fail("the source (node 0) must have a positive compute rate");
        for (int i = 0; i < N; ++i) if (computeRates_[i] < 0.0) return fail("node " + std::to_string(i) + " has a negative compute rate");
        std::vector<std::vector<int>> adj(N);
        for (const GraphEdge& e : edges_) {
            if (e.u < 0 || e.u >= N || e.v < 0 || e.v >= N) return fail("edge endpoint out of range");
            if (e.rate < 0.0) return fail("edge has a negative rate");
            adj[e.u].push_back(e.v); adj[e.v].push_back(e.u);
        }
        std::vector<char> seen(N, 0); std::vector<int> st{0}; seen[0] = 1; int cnt = 1;   // BFS/DFS from source
        while (!st.empty()) { int u = st.back(); st.pop_back(); for (int w : adj[u]) if (!seen[w]) { seen[w] = 1; ++cnt; st.push_back(w); } }
        if (cnt != N) return fail("graph is not connected from the source");
        if (error) error->clear();
        return true;
    }

private:
    std::vector<double>    computeRates_;
    std::vector<GraphEdge> edges_;
    double                 totalLoad_ = 0.0;
};

struct GraphSolution {
    SolveStatus         status   = SolveStatus::NotSolved;
    double              makespan = 0.0;
    std::vector<double> loads;            // αᵢ per ORIGINAL node index
    std::vector<int>    parent;           // chosen arborescence parent per node (-1 for source)

    double totalAssignedLoad() const { double s = 0.0; for (double a : loads) s += a; return s; }
    bool feasible() const { return status == SolveStatus::Optimal || status == SolveStatus::Feasible; }
};

class GraphSolver {
public:
    explicit GraphSolver(long maxArborescences = 200000) : cap_(maxArborescences) {}
    std::string name() const { return "graph"; }

    // Goal:   minimize Cmax over the graph by choosing the best spanning
    //         arborescence rooted at the source and solving its tree LP.
    // Input:  the graph instance; seed - LP anti-cycling seed.
    // Output: a GraphSolution; Optimal with loads (per original node), the chosen
    //         parents, and the makespan. Failure if the enumeration cap is hit.
    GraphSolution solve(const GraphInstance& inst, std::uint64_t seed = 0) const {
        GraphSolution sol;
        std::string error;
        if (!inst.validate(&error)) { sol.status = SolveStatus::Infeasible; return sol; }

        const int N = static_cast<int>(inst.numNodes());
        // Adjacency with rates: cand[v] = {(neighbor, rate)} (candidate parents of v).
        std::vector<std::vector<std::pair<int,double>>> cand(N);
        for (const GraphEdge& e : inst.edges()) {
            cand[e.u].push_back({e.v, e.rate});
            cand[e.v].push_back({e.u, e.rate});
        }

        std::vector<int>    parent(N, -1);   // current assignment (parent[0] stays -1)
        std::vector<double> plink(N, 0.0);   // rate of the edge to the chosen parent
        long tried = 0;
        bool capHit = false;
        double bestMk = std::numeric_limits<double>::infinity();

        // Recurse over nodes 1..N-1, choosing each one's parent edge.
        std::function<void(int)> choose = [&](int v) {
            if (capHit) return;
            if (v == N) {                                  // a full parent assignment
                if (++tried > cap_) { capHit = true; return; }
                evaluateArborescence(inst, parent, plink, seed, bestMk, sol);
                return;
            }
            for (const auto& [par, rate] : cand[v]) {
                parent[v] = par; plink[v] = rate;
                choose(v + 1);
                if (capHit) return;
            }
        };
        choose(1);

        if (capHit && !sol.feasible()) { sol.status = SolveStatus::Failure; return sol; }  // too big for exact
        if (!sol.feasible())           { sol.status = SolveStatus::Failure; return sol; }
        return sol;
    }

private:
    long cap_;

    // Evaluate one candidate arborescence (if valid): re-root into a TreeInstance
    // and solve; keep it if it beats the best makespan so far.
    static void evaluateArborescence(const GraphInstance& inst, const std::vector<int>& parent,
                                     const std::vector<double>& plink, std::uint64_t seed,
                                     double& bestMk, GraphSolution& best) {
        const int N = static_cast<int>(inst.numNodes());
        // Validity: every node reaches the source via parents (acyclic in-tree at 0).
        std::vector<int> order; order.reserve(N);
        std::vector<std::vector<int>> children(N);
        for (int v = 1; v < N; ++v) children[parent[v]].push_back(v);
        std::vector<char> seen(N, 0); std::vector<int> bfs{0}; seen[0] = 1;
        for (std::size_t h = 0; h < bfs.size(); ++h) {
            int u = bfs[h]; order.push_back(u);
            for (int c : children[u]) if (!seen[c]) { seen[c] = 1; bfs.push_back(c); }
        }
        if (static_cast<int>(order.size()) != N) return;     // not a spanning arborescence (cycle / unreached)

        // Re-index in BFS order (parents precede children) and build the tree.
        std::vector<int> newIdx(N, -1);
        for (int k = 0; k < N; ++k) newIdx[order[k]] = k;
        std::vector<TreeNode> tn(N);
        for (int k = 0; k < N; ++k) {
            int orig = order[k];
            tn[k].computeRate = inst.computeRates()[orig];
            tn[k].linkRate    = (orig == 0) ? 0.0 : plink[orig];
            tn[k].parent      = (orig == 0) ? -1  : newIdx[parent[orig]];
        }
        TreeInstance tree(std::move(tn), inst.totalLoad());
        TreeSolution ts = TreeSolver().solve(tree, seed);
        if (ts.status != SolveStatus::Optimal) return;

        if (ts.makespan < bestMk - 1e-12) {
            bestMk = ts.makespan;
            best.status   = SolveStatus::Optimal;
            best.makespan = ts.makespan;
            best.loads.assign(N, 0.0);
            best.parent.assign(N, -1);
            for (int k = 0; k < N; ++k) {
                best.loads[order[k]]  = ts.loads[k];
                best.parent[order[k]] = (order[k] == 0) ? -1 : parent[order[k]];
            }
        }
    }
};

}  // namespace dls

#endif  // DLS_TOPOLOGY_GRAPH_HPP
