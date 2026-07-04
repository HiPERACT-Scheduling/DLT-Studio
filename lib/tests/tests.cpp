//---------------------------------------------------------------------------
// tests/tests.cpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Unit/behaviour tests for the DLS library, using the vendored doctest header.
// Built and run via `make test`. Single translation unit: it pulls in the
// header-only core and the GA (whose ga.hpp defines functions inline within
// this TU), plus doctest's main.
//---------------------------------------------------------------------------

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.h"

#include <cstdio>
#include <fstream>
#include <string>

#include <functional>
#include <vector>

#include "core/dls_instance.hpp"
#include "core/energy_model.hpp"
#include "core/simplex_schedule_evaluator.hpp"
#include "core/caching_schedule_evaluator.hpp"
#include "core/evaluator_factory.hpp"
#include "mlsd/mlsd_solver.hpp"
#include "mlsd/mlsd_ga_solver.hpp"
#include "mapreduce/mapreduce_solver.hpp"
#include "mapreduce/multilayer_solver.hpp"
#include "topology/linear_chain.hpp"
#include "topology/tree.hpp"
#include "topology/graph.hpp"
#include "heuristics/ga/ga_solver.hpp"
#include "heuristics/best_rate/best_rate_solver.hpp"
#include "heuristics/online/online_solver.hpp"
#include "heuristics/single_round/single_round_solver.hpp"
#include "core/homogeneous.hpp"
#include "core/bounds.hpp"
#include "exact/enumerative/exact_solver.hpp"
#include "exact/optv/optv_solver.hpp"
#include "exact/dual/dual_bisection_solver.hpp"
#include "core/solver_registry.hpp"
#include "core/instance_io.hpp"
#include "core/schedule_expand.hpp"
#include "core/json_io.hpp"
#include "core/pareto.hpp"
#include "cli/class_io.hpp"
#include "bench/benchmark.hpp"
#include "heuristics/fptas/fptas_optv_solver.hpp"
#include "heuristics/fptas/fptas_optt_solver.hpp"
#ifdef DLS_WITH_HIGHS
#include "exact/milp/milp_solver.hpp"
#include "exact/milp/multi_milp_solver.hpp"
#include "exact/milp/mlsd_milp_solver.hpp"
#endif

using namespace dls;

// The canonical 5-processor instance (S, C, A, B) whose fixed-order LP optimum
// is the makespan 125689 (matches the standalone evaluator check).
static DLSInstance canonical5() {
    return DLSInstance({{0.0,0.23,0.12,1e9},{0.0,0.11,0.23,1e9},{0.0,0.11,0.11,1e9},
                        {0.0,0.11,0.31,1e9},{0.0,0.11,0.11,1e9}}, 1e6);
}

// Helper: every installment's load must respect its processor's memory limit.
static bool fragmentsWithinMemory(const DLSSolution& s, const DLSInstance& inst, double tol = 1.0) {
    for (const LoadFragment& f : s.fragments)
        if (f.loadSize > inst.processors()[f.processorId].memoryLimit + tol) return false;
    return true;
}

// An identical MLSD processor (S=C=A=1), used by the MLSD tests.
static Processor identicalProc() {
    Processor p; p.commStartup=1; p.commRate=1; p.computeRate=1; p.memoryLimit=1e18; return p;
}

// The thesis's worked m=4 single-installment example, all fields {S,C,A,B,p,r,d,f,l}.
static DLSInstance thesisM4() {
    return DLSInstance({
        {1.0, 1.0,  2.0, 10.0, 0.0, 0.0,  10.0,  1.0, 0.5},   // P1
        {1.0, 0.1,  0.5, 10.0, 1.0, 10.0, 20.0,  5.0, 1.0},   // P2
        {1.0, 2.0,  1.0, 10.0, 1.0, 20.0, 30.0,  3.0, 0.3},   // P3
        {2.0, 2.0,  2.0, 20.0, 0.0, 20.0, 200.0, 2.0, 1.0},   // P4
    }, 20.0);
}

//=================== core: instance validation =============================
TEST_CASE("DLSInstance::validate accepts a good instance and rejects bad ones") {
    std::string err;
    CHECK(canonical5().validate(&err));

    DLSInstance empty;
    CHECK_FALSE(empty.validate(&err));                       // no processors

    DLSInstance noLoad({{0.0,0.1,0.1,1e9}}, 0.0);
    CHECK_FALSE(noLoad.validate(&err));                      // totalLoad <= 0

    DLSInstance neg({{0.0,-0.1,0.1,1e9}}, 100.0);
    CHECK_FALSE(neg.validate(&err));                         // negative cost

    DLSInstance zeroCost({{0.0,0.0,0.0,1e9}}, 100.0);
    CHECK_FALSE(zeroCost.validate(&err));                    // no per-unit cost
}

//=================== core: simplex evaluator ===============================
TEST_CASE("SimplexScheduleEvaluator solves the canonical instance optimally") {
    SimplexScheduleEvaluator ev;
    EvaluatorConfig cfg; cfg.seed = 1;
    DLSInstance inst = canonical5();
    DLSSolution s = ev.evaluate(inst, {0,1,2,3,4}, cfg);

    CHECK(s.status == SolveStatus::Optimal);
    CHECK(s.makespan == doctest::Approx(125689).epsilon(0.001));
    CHECK(s.conservesLoad(inst.totalLoad(), 1.0));
}

TEST_CASE("SimplexScheduleEvaluator handles multi-installment sequences") {
    SimplexScheduleEvaluator ev;
    EvaluatorConfig cfg; cfg.seed = 1;
    DLSInstance inst({{0.1,0.11,0.52,400000},{0.2,0.21,0.22,50000},{0.3,0.31,0.32,15000}}, 100000);
    DLSSolution s = ev.evaluate(inst, {0,1,2,0,1}, cfg);   // P1 and P2 twice

    CHECK(s.status == SolveStatus::Optimal);
    CHECK(s.conservesLoad(inst.totalLoad(), 1.0));
    CHECK(s.fragments.size() == 5);
}

TEST_CASE("SimplexScheduleEvaluator rejects an out-of-range sequence index") {
    SimplexScheduleEvaluator ev;
    DLSInstance inst = canonical5();
    DLSSolution s = ev.evaluate(inst, {0,1,99}, EvaluatorConfig{});
    CHECK(s.status == SolveStatus::Failure);
}

//=================== core: caching decorator ===============================
TEST_CASE("CachingScheduleEvaluator memoizes by sequence with identical results") {
    SimplexScheduleEvaluator simplex;
    CachingScheduleEvaluator cache(simplex);
    EvaluatorConfig cfg; cfg.seed = 1;
    DLSInstance inst = canonical5();

    DLSSolution a = cache.evaluate(inst, {0,1,2,3,4}, cfg);
    DLSSolution b = cache.evaluate(inst, {0,1,2,3,4}, cfg);   // same -> cache hit
    DLSSolution c = cache.evaluate(inst, {4,3,2,1,0}, cfg);   // different -> miss

    CHECK(a.makespan == b.makespan);
    CHECK(a.makespan == cache.evaluate(inst, {0,1,2,3,4}, cfg).makespan);
    CHECK(cache.cachedCount() == 2);                          // only 2 distinct solves
    CHECK(c.status == SolveStatus::Optimal);
}

//=================== evaluator factory =====================================
TEST_CASE("makeScheduleEvaluator returns simplex and matches direct construction") {
    auto ev = makeScheduleEvaluator("simplex");
    REQUIRE(ev != nullptr);
    CHECK(ev->name() == "simplex");

    DLSInstance inst = canonical5();
    EvaluatorConfig cfg; cfg.seed = 1;
    SimplexScheduleEvaluator direct;
    DLSSolution viaFactory = ev->evaluate(inst, {0,1,2,3,4}, cfg);
    DLSSolution viaDirect  = direct.evaluate(inst, {0,1,2,3,4}, cfg);
    CHECK(viaFactory.makespan == doctest::Approx(viaDirect.makespan));

    CHECK(makeScheduleEvaluator("nonexistent") == nullptr);   // unknown backend
}

TEST_CASE("GASolver rejects an unknown evaluator backend") {
    GAParams p; p.populationSize=8; p.maxGenerations=10; p.installments=4;
    p.evaluatorBackend = "nope";
    CHECK(GASolver(p).solve(canonical5(), SolverConfig{}).status == SolveStatus::Failure);
}

//=================== HiGHS backend (only when compiled in) =================
#ifdef DLS_WITH_HIGHS
TEST_CASE("HiGHS backend is registered and agrees with CSimplex on the LP optimum") {
    auto simplex = makeScheduleEvaluator("simplex");
    auto highs   = makeScheduleEvaluator("highs");
    REQUIRE(highs != nullptr);
    CHECK(highs->name() == "highs");

    DLSInstance inst = canonical5();
    EvaluatorConfig cfg; cfg.seed = 1;
    std::vector<std::vector<int>> seqs = {{0,1,2,3,4}, {2,0,4,1,3,2}, {0,0,1}, {4,3,2,1,0}};
    for (const auto& seq : seqs) {
        DLSSolution a = simplex->evaluate(inst, seq, cfg);
        DLSSolution b = highs->evaluate(inst, seq, cfg);
        CHECK(a.status == b.status);
        if (a.status == SolveStatus::Optimal) {
            CHECK(b.makespan == doctest::Approx(a.makespan).epsilon(1e-3));  // same LP, agree closely
            CHECK(b.conservesLoad(inst.totalLoad(), 1.0));
        }
    }
}

TEST_CASE("solveLpViaHighs trivial: min x s.t. x>=5 => 5") {
    HighsLpSolution r = solveLpViaHighs(1, {1.0}, {{-1.0}}, {-5.0});  // -x <= -5
    CHECK(static_cast<int>(r.status) == static_cast<int>(SolveStatus::Optimal));
    if (r.status == SolveStatus::Optimal) CHECK(r.values[0] == doctest::Approx(5.0));
}

TEST_CASE("MLSD HiGHS backend agrees with CSimplex (thesis Example 1)") {
    MlsdInstance inst({{32.0,0.0},{2.0,0.0}}, {identicalProc(), identicalProc(), identicalProc()});
    MlsdStructure st; st.taskOrder={0,1}; st.procSeq={{0,1,2},{0,1}};
    MlsdScheduleEvaluator simplex("simplex");
    MlsdScheduleEvaluator highs("highs");
    double a = simplex.evaluate(inst, st).makespan;
    MlsdSolution hs = highs.evaluate(inst, st);
    CHECK(static_cast<int>(hs.status) == static_cast<int>(SolveStatus::Optimal));
    double b = hs.makespan;
    CHECK(b == doctest::Approx(a).epsilon(1e-4));
    CHECK(b == doctest::Approx(39.3333).epsilon(1e-4));
}

TEST_CASE("HiGHS backend honours availability conditions (p) like CSimplex") {
    // Same p=5 case as the dependency-free test: Cmax = p + A*V = 15.
    DLSInstance inst({{0.0, 0.0, 1.0, 1e9, 5.0, 0.0, 0.0}}, 10.0);
    auto highs = makeScheduleEvaluator("highs");
    REQUIRE(highs != nullptr);
    CHECK(highs->evaluate(inst, {0}, EvaluatorConfig{}).makespan == doctest::Approx(15.0));
}

TEST_CASE("HiGHS backend reproduces the result-return closed form (beta)") {
    DLSInstance inst({{0.0, 1.0, 1.0, 1e9}}, 10.0);
    inst.setResultFraction(0.5);
    auto highs = makeScheduleEvaluator("highs");
    REQUIRE(highs != nullptr);
    CHECK(highs->evaluate(inst, {0}, EvaluatorConfig{}).makespan == doctest::Approx(25.0));
}

TEST_CASE("HiGHS backend reproduces the piecewise swap-point and agrees with CSimplex") {
    Processor p; p.commStartup = 0.0; p.commRate = 0.0; p.computeRate = 0.0;
    p.memoryLimit = 1e9; p.computePieces = {{0.0, 1.0}, {-10.0, 3.0}};   // swap at q=5
    auto simplex = makeScheduleEvaluator("simplex");
    auto highs   = makeScheduleEvaluator("highs");
    REQUIRE(highs != nullptr);
    for (double V : {4.0, 5.0, 10.0}) {
        DLSInstance inst({p}, V);
        double a = simplex->evaluate(inst, {0}, EvaluatorConfig{}).makespan;
        double b = highs->evaluate(inst, {0}, EvaluatorConfig{}).makespan;
        CHECK(b == doctest::Approx(a));
    }
    DLSInstance above({p}, 10.0);
    CHECK(highs->evaluate(above, {0}, EvaluatorConfig{}).makespan == doctest::Approx(20.0));
}

TEST_CASE("OptV MaxLoad: HiGHS and CSimplex agree on the max load") {
    DLSInstance inst({{0.0,1.0,1.0,1e9},{0.1,0.2,0.3,1e9}}, 1.0);
    auto simplex = makeScheduleEvaluator("simplex");
    auto highs   = makeScheduleEvaluator("highs");
    REQUIRE(highs != nullptr);
    EvaluatorConfig c; c.objective = EvalObjective::MaxLoad; c.makespanLimit = 30.0;
    DLSSolution s = simplex->evaluate(inst, {0,1}, c);
    DLSSolution h = highs->evaluate(inst, {0,1}, c);
    REQUIRE(s.status == SolveStatus::Optimal);
    REQUIRE(h.status == SolveStatus::Optimal);
    CHECK(h.totalAssignedLoad() == doctest::Approx(s.totalAssignedLoad()).epsilon(1e-4));
}

TEST_CASE("BestRateSolver agrees on the HiGHS and CSimplex backends") {
    DLSInstance inst = canonical5();
    BestRateParams sp; sp.evaluatorBackend = "simplex";
    BestRateParams hp; hp.evaluatorBackend = "highs";
    DLSSolution s = BestRateSolver(sp).solve(inst, SolverConfig{});
    DLSSolution h = BestRateSolver(hp).solve(inst, SolverConfig{});
    REQUIRE(s.status == SolveStatus::Feasible);
    REQUIRE(h.status == SolveStatus::Feasible);
    CHECK(h.makespan == doctest::Approx(s.makespan).epsilon(1e-3));   // same sequence + model
}

TEST_CASE("GASolver runs on the HiGHS backend (feasible + conserves load)") {
    DLSInstance inst = canonical5();
    GAParams p; p.installments=5; p.populationSize=12; p.maxGenerations=100;
    p.noImprovementLimit=50; p.evaluatorBackend = "highs";
    SolverConfig c; c.seed = 5;
    DLSSolution s = GASolver(p).solve(inst, c);
    CHECK(s.feasible());
    CHECK(s.conservesLoad(inst.totalLoad(), 1.0));
}

TEST_CASE("ExactSolver finds the same optimum under either backend") {
    DLSInstance inst = canonical5();
    ExactParams simplex; simplex.maxInstallments=4; simplex.evaluatorBackend="simplex";
    ExactParams highs;   highs.maxInstallments=4;   highs.evaluatorBackend="highs";
    double so = ExactSolver(simplex).solve(inst, SolverConfig{}).makespan;
    double ho = ExactSolver(highs).solve(inst, SolverConfig{}).makespan;
    CHECK(ho == doctest::Approx(so).epsilon(1e-3));   // same model, both exact
}

//=================== MILP solver (cross-validated vs B&B) ==================
TEST_CASE("MilpSolver matches the single-installment B&B optimum") {
    // Instances where single-installment (each processor once) is feasible.
    std::vector<DLSInstance> insts = {
        canonical5(),
        DLSInstance({{0.5,0.20,0.30,1e9},{0.7,0.15,0.25,1e9},{0.3,0.25,0.40,1e9}}, 10000),
        DLSInstance({{0.0,0.23,0.12,1e6},{0.1,0.11,0.23,1e6},{0.2,0.11,0.31,1e6}}, 5000),
    };
    for (DLSInstance& inst : insts) {
        MilpParams mp; mp.maxInstallments = (int)inst.numProcessors();
        DLSSolution milp = MilpSolver(mp).solve(inst, SolverConfig{});

        ExactParams ep; ep.maxInstallments = (int)inst.numProcessors(); ep.allowRepeats = false;
        DLSSolution bnb = ExactSolver(ep).solve(inst, SolverConfig{});

        CHECK(milp.status == SolveStatus::Optimal);
        CHECK(milp.conservesLoad(inst.totalLoad(), 1.0));
        CHECK(milp.makespan == doctest::Approx(bnb.makespan).epsilon(1e-3));  // two exact methods agree
    }
}

TEST_CASE("MultiMilpSolver matches the MULTI-installment B&B (ample memory)") {
    // Independent exact cross-check of ExactSolver's multi-installment optimum.
    // Ample memory (B >> V) => the per-chunk buffer is the only memory limit, so
    // the MILP's pipelined finish times equal the B&B's carried-load makespan.
    std::vector<DLSInstance> insts = {
        DLSInstance({{0.1,0.2,0.3,1e9},{0.2,0.15,0.25,1e9}}, 100.0),
        DLSInstance({{0.0,0.20,0.30,1e9},{0.0,0.15,0.25,1e9},{0.0,0.25,0.40,1e9}}, 1000.0),
        DLSInstance({{0.5,0.20,0.30,1e9},{0.3,0.25,0.40,1e9}}, 50.0),
    };
    for (DLSInstance& inst : insts) {
        MultiMilpParams mp; mp.maxInstallments = 3;
        DLSSolution milp = MultiMilpSolver(mp).solve(inst, SolverConfig{});

        ExactParams ep; ep.maxInstallments = 3; ep.allowRepeats = true;   // multi-installment
        DLSSolution bnb = ExactSolver(ep).solve(inst, SolverConfig{});

        REQUIRE(milp.status == SolveStatus::Optimal);
        CHECK(milp.conservesLoad(inst.totalLoad(), 1.0));
        CHECK(milp.makespan == doctest::Approx(bnb.makespan).epsilon(1e-3));  // independent exact methods agree
    }
}

TEST_CASE("MultiMilpSolver can beat single-installment when multi-installment helps") {
    // A 2-processor instance where pipelining (multi-installment) lowers Cmax:
    // milp-multi must be <= the single-installment MILP optimum.
    DLSInstance inst({{0.0,0.1,0.5,1e9},{0.0,0.1,0.5,1e9}}, 100.0);
    MultiMilpParams mm; mm.maxInstallments = 4;
    double multi = MultiMilpSolver(mm).solve(inst, SolverConfig{}).makespan;
    MilpParams ms; ms.maxInstallments = 2;
    double single = MilpSolver(ms).solve(inst, SolverConfig{}).makespan;
    CHECK(multi <= single + 1e-6);
}

TEST_CASE("MilpSolver reproduces the piecewise swap-point and matches B&B(single)") {
    // Single processor, swap at q=5: V=10 spills to the disk piece => Cmax=20.
    Processor p; p.commStartup = 0.0; p.commRate = 0.0; p.computeRate = 0.0;
    p.memoryLimit = 1e9; p.computePieces = {{0.0, 1.0}, {-10.0, 3.0}};
    DLSInstance one({p}, 10.0);
    MilpParams m1; m1.maxInstallments = 1;
    CHECK(MilpSolver(m1).solve(one, SolverConfig{}).makespan == doctest::Approx(20.0));

    // Two memory-hierarchy processors: the MILP optimum must equal the
    // single-installment B&B optimum (whose evaluator also models the pieces).
    Processor a; a.commStartup = 0.1; a.commRate = 0.1; a.memoryLimit = 1e9;
    a.computePieces = {{0.0, 0.2}, {-30.0, 0.7}};   // swap at 60
    Processor b; b.commStartup = 0.2; b.commRate = 0.15; b.memoryLimit = 1e9;
    b.computePieces = {{0.0, 0.3}, {-40.0, 0.9}};   // swap at ~66.7
    DLSInstance inst({a, b}, 200.0);

    MilpParams mp; mp.maxInstallments = 2;
    DLSSolution milp = MilpSolver(mp).solve(inst, SolverConfig{});
    ExactParams ep; ep.maxInstallments = 2; ep.allowRepeats = false;
    DLSSolution bnb = ExactSolver(ep).solve(inst, SolverConfig{});

    CHECK(milp.status == SolveStatus::Optimal);
    CHECK(milp.conservesLoad(inst.totalLoad(), 1.0));
    CHECK(milp.makespan == doctest::Approx(bnb.makespan).epsilon(1e-3));
}

TEST_CASE("MilpSolver: infeasible when total memory < V") {
    // 3 processors, buffer 1000 each -> 3000 < V=10000.
    DLSInstance inst({{0,0.2,0.3,1000.0},{0,0.15,0.25,1000.0},{0,0.25,0.4,1000.0}}, 10000);
    MilpParams mp; mp.maxInstallments = 3;
    CHECK(MilpSolver(mp).solve(inst, SolverConfig{}).status == SolveStatus::Infeasible);
}

TEST_CASE("MILP honours p/r/d and matches B&B(single) with availability conditions") {
    // 3 processors with compute startup, a release, and a deadline.
    DLSInstance inst({{0.1,0.2,0.3,1e9, 1.0, 0.0, 0.0},     // p=1
                      {0.2,0.15,0.25,1e9, 0.0, 5.0, 0.0},   // r=5
                      {0.3,0.25,0.4,1e9, 0.0, 0.0, 5000.0}} , 6000.0);  // d=5000
    MilpParams mp; mp.maxInstallments = 3;
    ExactParams ep; ep.maxInstallments = 3; ep.allowRepeats = false;
    DLSSolution milp = MilpSolver(mp).solve(inst, SolverConfig{});
    DLSSolution bnb  = ExactSolver(ep).solve(inst, SolverConfig{});
    CHECK(milp.status == bnb.status);
    if (milp.status == SolveStatus::Optimal)
        CHECK(milp.makespan == doctest::Approx(bnb.makespan).epsilon(1e-3));
}

TEST_CASE("MILP bi-criteria matches B&B and respects the cost limit") {
    DLSInstance inst = thesisM4();
    const double Gbar = 24.5;
    MilpParams mp; mp.maxInstallments = 4; mp.costLimit = Gbar;
    ExactParams ep; ep.maxInstallments = 4; ep.allowRepeats = false; ep.costLimit = Gbar;
    DLSSolution milp = MilpSolver(mp).solve(inst, SolverConfig{});
    DLSSolution bnb  = ExactSolver(ep).solve(inst, SolverConfig{});
    REQUIRE(milp.status == SolveStatus::Optimal);
    CHECK(milp.cost <= Gbar + 1e-3);
    CHECK(milp.makespan == doctest::Approx(bnb.makespan).epsilon(1e-3));
}

TEST_CASE("MILP reverse bi-criteria: minimize cost subject to a makespan limit") {
    DLSInstance inst = thesisM4();
    MilpParams mp; mp.maxInstallments = 4; mp.minimizeCost = true; mp.makespanLimit = 41.5;
    DLSSolution s = MilpSolver(mp).solve(inst, SolverConfig{});
    REQUIRE(s.status == SolveStatus::Optimal);
    CHECK(s.makespan <= 41.5 + 1e-3);            // makespan limit respected
    CHECK(s.cost <= 24.25 + 1e-3);               // <= the full-sequence frontier cost
}

TEST_CASE("MilpSolver rejects invalid params") {
    MilpParams bad; bad.maxInstallments = 0;
    CHECK(MilpSolver(bad).solve(canonical5(), SolverConfig{}).status == SolveStatus::Failure);
}

TEST_CASE("MILP == B&B(single) across sizes and the memory boundary") {
    std::vector<DLSInstance> insts = {
        DLSInstance({{0.1,0.20,0.30,1e9},{0.2,0.15,0.25,1e9}}, 5000.0),                 // N=2
        DLSInstance({{0.1,0.20,0.30,1e9},{0.2,0.15,0.25,1e9},{0.3,0.25,0.40,1e9}}, 8000.0), // N=3
        DLSInstance({{0.0,0.20,0.30,3000.0},{0.0,0.15,0.25,3000.0},{0.0,0.25,0.40,3000.0}}, 7000.0), // N=3, B binds
    };
    for (DLSInstance& inst : insts) {
        const int K = (int)inst.numProcessors();
        MilpParams mp; mp.maxInstallments = K;
        ExactParams ep; ep.maxInstallments = K; ep.allowRepeats = false;
        DLSSolution milp = MilpSolver(mp).solve(inst, SolverConfig{});
        DLSSolution bnb  = ExactSolver(ep).solve(inst, SolverConfig{});
        CHECK(milp.status == bnb.status);
        if (milp.status == SolveStatus::Optimal) {
            CHECK(milp.makespan == doctest::Approx(bnb.makespan).epsilon(1e-3));
            CHECK(milp.conservesLoad(inst.totalLoad(), 1.0));
            CHECK(fragmentsWithinMemory(milp, inst));
        }
    }
}
#endif  // DLS_WITH_HIGHS

//=================== GA: reproducibility & strategies ======================
static DLSSolution solveGA(unsigned long seed, GAParams p) {
    DLSInstance inst = canonical5();
    GASolver g(p);
    SolverConfig c; c.seed = seed;
    return g.solve(inst, c);
}

TEST_CASE("GASolver is reproducible for a fixed seed") {
    GAParams p; p.populationSize=12; p.maxGenerations=120; p.noImprovementLimit=50; p.installments=6;
    DLSSolution a = solveGA(42, p);
    DLSSolution b = solveGA(42, p);
    CHECK(a.status == SolveStatus::Feasible);
    CHECK(a.makespan == b.makespan);
    CHECK(a.sequence == b.sequence);
    CHECK(a.usedSeed == 42);
}

TEST_CASE("GASolver: every strategy combo stays feasible and conserves load") {
    GAParams base; base.populationSize=12; base.maxGenerations=120;
    base.noImprovementLimit=50; base.installments=6;

    for (Selection sel : {Selection::Roulette, Selection::Tournament})
    for (Crossover cx  : {Crossover::SinglePoint, Crossover::TwoPoint})
    for (Mutation  mu  : {Mutation::PerGene, Mutation::Swap}) {
        GAParams p = base; p.selection=sel; p.crossover=cx; p.mutation=mu; p.tournamentSize=3;
        DLSInstance inst = canonical5();
        GASolver g(p); SolverConfig c; c.seed = 7;
        DLSSolution s = g.solve(inst, c);
        CHECK(s.feasible());
        CHECK(s.conservesLoad(inst.totalLoad(), 1.0));
    }
}

TEST_CASE("GASolver records usedSeed so an unseeded run is replayable") {
    GAParams p; p.populationSize=10; p.maxGenerations=80; p.noImprovementLimit=40; p.installments=5;
    DLSInstance inst = canonical5();
    GASolver g(p);
    DLSSolution u = g.solve(inst, SolverConfig{});           // unseeded
    REQUIRE(u.feasible());
    DLSSolution replay = solveGA(u.usedSeed, p);
    CHECK(replay.makespan == u.makespan);
}

//=================== validation paths ======================================
TEST_CASE("GASolver rejects an invalid instance and invalid params") {
    GAParams good; good.populationSize=8; good.maxGenerations=10; good.installments=4;

    DLSInstance bad;                                          // empty -> infeasible
    CHECK(GASolver(good).solve(bad, SolverConfig{}).status == SolveStatus::Infeasible);

    GAParams badParams = good; badParams.populationSize = 0;  // invalid params -> failure
    CHECK(GASolver(badParams).solve(canonical5(), SolverConfig{}).status == SolveStatus::Failure);

    std::string err;
    CHECK_FALSE(badParams.validate(&err));
}

//=================== file I/O error handling ===============================
TEST_CASE("Readers return false on a missing file and true on a valid one") {
    ProgrammeParameters pp;
    std::string err;
    CHECK_FALSE(pp.ReadProgrammeParam("/no/such/dls_file.txt", &err));   // missing
    CHECK_FALSE(err.empty());

    // Write a valid programme file and read it back.
    const char* path = "/tmp/dls_test_prog.txt";
    { std::ofstream f(path); f << "5 4 6 12 500 task.txt out.txt 1\n"; }
    ProgrammeParameters ok;
    REQUIRE(ok.ReadProgrammeParam(path, &err));
    CHECK(ok.maxProcNum == 5);
    CHECK(ok.numProcParams == 4);
    CHECK(ok.solverMaxIter == 500);
    CHECK(ok.fileLP == "task.txt");
    std::remove(path);
}

//=================== exact solver ==========================================

// Reference brute force: minimum makespan over all sequences of length 1..L
// (no pruning), used to validate the branch-and-bound bounds.
static double bruteForceOptimum(const DLSInstance& inst, int L, bool repeats) {
    SimplexScheduleEvaluator ev;
    EvaluatorConfig cfg; cfg.seed = 1;
    cfg.seed = 0;                       // match ExactSolver's fixed LP seed for exact comparison
    double best = 1e300;
    std::vector<int> seq;
    std::vector<bool> used(inst.numProcessors(), false);
    std::function<void()> rec = [&]() {
        if (!seq.empty()) {
            DLSSolution s = ev.evaluate(inst, seq, cfg);
            if (s.status == SolveStatus::Optimal && s.makespan < best) best = s.makespan;
        }
        if ((int)seq.size() >= L) return;
        for (int i = 0; i < (int)inst.numProcessors(); ++i) {
            if (!repeats && used[i]) continue;
            seq.push_back(i); used[i] = true;
            rec();
            used[i] = false; seq.pop_back();
        }
    };
    rec();
    return best;
}

TEST_CASE("ExactSolver matches brute force (validates B&B pruning) and is optimal") {
    // A small instance with NON-zero startups so the startup bound actually prunes.
    DLSInstance inst({{0.5,0.20,0.30,1e9},{0.7,0.15,0.25,1e9},{0.3,0.25,0.40,1e9}}, 10000);

    for (bool repeats : {true, false}) {
        ExactParams p; p.maxInstallments = 4; p.allowRepeats = repeats;
        DLSSolution s = ExactSolver(p).solve(inst, SolverConfig{});
        CHECK(s.status == SolveStatus::Optimal);
        CHECK(s.conservesLoad(inst.totalLoad(), 1.0));
        CHECK(s.makespan == doctest::Approx(bruteForceOptimum(inst, 4, repeats)).epsilon(1e-4));
    }
}

TEST_CASE("ExactSolver: multi-installment optimum <= single-installment optimum") {
    // B large enough that single-installment (<=3 procs) can still hold V=10000.
    DLSInstance inst({{0.5,0.20,0.30,1e9},{0.7,0.15,0.25,1e9},{0.3,0.25,0.40,1e9}}, 10000);
    ExactParams multi;  multi.maxInstallments = 4; multi.allowRepeats = true;
    ExactParams single; single.maxInstallments = 4; single.allowRepeats = false;
    double mo = ExactSolver(multi).solve(inst, SolverConfig{}).makespan;
    double so = ExactSolver(single).solve(inst, SolverConfig{}).makespan;
    CHECK(mo <= so + 1e-6);   // repeats add options, never hurt
}

TEST_CASE("ExactSolver optimum is a lower bound on the GA result") {
    DLSInstance inst = canonical5();
    const int L = 5;
    ExactParams ep; ep.maxInstallments = L; ep.allowRepeats = true;
    double exactCmax = ExactSolver(ep).solve(inst, SolverConfig{}).makespan;

    GAParams gp; gp.populationSize=12; gp.maxGenerations=120; gp.noImprovementLimit=50; gp.installments=L;
    SolverConfig c; c.seed = 3;
    double gaCmax = GASolver(gp).solve(inst, c).makespan;

    CHECK(exactCmax <= gaCmax + 1e-6);   // exact is optimal over the same space
}

TEST_CASE("ExactSolver is deterministic regardless of seed") {
    DLSInstance inst = canonical5();
    ExactParams p; p.maxInstallments = 4; p.allowRepeats = true;
    SolverConfig c1; c1.seed = 1;
    SolverConfig c2; c2.seed = 999;
    DLSSolution a = ExactSolver(p).solve(inst, c1);
    DLSSolution b = ExactSolver(p).solve(inst, c2);
    CHECK(a.makespan == b.makespan);
    CHECK(a.sequence == b.sequence);
}

TEST_CASE("ExactSolver respects a node budget (returns best-so-far as Feasible)") {
    DLSInstance inst = canonical5();
    ExactParams p; p.maxInstallments = 6; p.allowRepeats = true; p.nodeBudget = 5;
    DLSSolution s = ExactSolver(p).solve(inst, SolverConfig{});
    CHECK(s.status == SolveStatus::Feasible);     // not proven optimal
    CHECK(s.feasible());
    CHECK(s.conservesLoad(inst.totalLoad(), 1.0));
    CHECK(s.iterations <= 5);
}

TEST_CASE("ExactSolver rejects invalid instance and params") {
    ExactParams ok; ok.maxInstallments = 3;
    DLSInstance empty;
    CHECK(ExactSolver(ok).solve(empty, SolverConfig{}).status == SolveStatus::Infeasible);
    ExactParams bad; bad.maxInstallments = 0;
    CHECK(ExactSolver(bad).solve(canonical5(), SolverConfig{}).status == SolveStatus::Failure);
}

//=================== validation: analytic ground truth =====================
// (model-validation rule: assert exact results against hand-derived values)

//=================== Auto meta-solver (feature-based dispatch) =============

TEST_CASE("AutoSolver: small instance -> exact (proven optimum)") {
    DLSInstance inst({{0.0,0.20,0.30,1e9},{0.1,0.15,0.25,1e9},{0.2,0.25,0.40,1e9}}, 1000.0);
    AutoSolver various;                       // default smallThreshold = 6
    DLSSolution s = various.solve(inst, SolverConfig{});
    CHECK(various.chosenSolver() == "exact");
    CHECK(s.status == SolveStatus::Optimal);
    ExactParams ep; ep.maxInstallments = 5; ep.allowRepeats = true;   // the depth auto uses
    CHECK(s.makespan == doctest::Approx(ExactSolver(ep).solve(inst, SolverConfig{}).makespan).epsilon(1e-3));
}

TEST_CASE("AutoSolver: ample memory + no startups -> single-round") {
    std::vector<Processor> ps;
    for (int i = 0; i < 7; ++i) { Processor p; p.commRate = 0.1 + 0.01*i; p.computeRate = 0.2; p.memoryLimit = 1e9; ps.push_back(p); }
    DLSInstance inst(ps, 1000.0);             // N=7 > threshold, all S=0, ΣB >> V
    AutoSolver a;
    DLSSolution s = a.solve(inst, SolverConfig{});
    CHECK(a.chosenSolver() == "single-round");
    CHECK(s.feasible());
}

TEST_CASE("AutoSolver: memory-limited (ΣB < V) -> best-rate") {
    std::vector<Processor> ps;
    for (int i = 0; i < 7; ++i) { Processor p; p.commStartup=0.1; p.commRate=0.2; p.computeRate=0.3; p.memoryLimit=1000; ps.push_back(p); }
    DLSInstance inst(ps, 10000.0);            // ΣB = 7000 < V = 10000
    AutoSolver a;
    DLSSolution s = a.solve(inst, SolverConfig{});
    CHECK(a.chosenSolver() == "best-rate");
    CHECK(s.feasible());
    CHECK(s.conservesLoad(10000.0, 1.0));
}

TEST_CASE("AutoSolver: ample memory + startups -> ga") {
    std::vector<Processor> ps;
    for (int i = 0; i < 7; ++i) { Processor p; p.commStartup=0.5; p.commRate=0.2; p.computeRate=0.3; p.memoryLimit=1e9; ps.push_back(p); }
    DLSInstance inst(ps, 1000.0);             // N=7, S>0, ample memory
    AutoSolver a;
    DLSSolution s = a.solve(inst, SolverConfig{42});
    CHECK(a.chosenSolver() == "ga");
    CHECK(s.feasible());
}

TEST_CASE("AutoSolver: reachable via the registry and listed first") {
    CHECK(availableSolvers().front() == "auto");
    DLSInstance inst({{0.0,0.2,0.3,1e9},{0.1,0.15,0.25,1e9}}, 1000.0);
    DLSSolution s = makeSolver("auto")->solve(inst, SolverConfig{});
    CHECK(s.feasible());
}

//=================== Solver registry (uniform portfolio access) ===========

TEST_CASE("Registry: lists solvers and rejects unknown names") {
    std::vector<std::string> names = availableSolvers();
    CHECK(!names.empty());
    auto has = [&](const std::string& n) {
        return std::find(names.begin(), names.end(), n) != names.end();
    };
    CHECK(has("ga")); CHECK(has("best-rate")); CHECK(has("single-round"));
    CHECK(has("exact")); CHECK(has("optv")); CHECK(has("fptas-optv")); CHECK(has("fptas-optt"));
    CHECK(makeSolver("does-not-exist") == nullptr);
}

TEST_CASE("Registry: every listed solver constructs and reports a category") {
    for (const std::string& n : availableSolvers()) {
        std::unique_ptr<DLSSolver> s = makeSolver(n);
        REQUIRE(s != nullptr);
        CHECK(s->name() == n);
        CHECK((s->category() == SolverCategory::Exact || s->category() == SolverCategory::Heuristic));
    }
}

TEST_CASE("Registry: makeSolver('exact') matches a directly-built ExactSolver") {
    DLSInstance inst = canonical5();
    SolverOptions opt; opt.maxInstallments = 4; opt.allowRepeats = true;
    DLSSolution viaReg = makeSolver("exact", opt)->solve(inst, SolverConfig{});

    ExactParams ep; ep.maxInstallments = 4; ep.allowRepeats = true;
    DLSSolution direct = ExactSolver(ep).solve(inst, SolverConfig{});
    CHECK(viaReg.makespan == doctest::Approx(direct.makespan));
}

TEST_CASE("Registry: default-configurable solvers run feasibly via the registry") {
    DLSInstance inst({{0.0,0.20,0.30,1e9},{0.1,0.15,0.25,1e9},{0.2,0.25,0.40,1e9}}, 1000.0);
    for (const char* n : {"ga", "best-rate", "single-round", "exact"}) {
        SolverOptions opt; opt.maxInstallments = 3;
        DLSSolution s = makeSolver(n, opt)->solve(inst, SolverConfig{42});
        CHECK(s.feasible());
        CHECK(s.conservesLoad(1000.0, 1.0));
    }
    SolverOptions ov; ov.deadline = 500.0; ov.maxInstallments = 3;   // optv needs a deadline
    CHECK(makeSolver("optv", ov)->solve(inst, SolverConfig{}).feasible());
}

//=================== Instance I/O (rich text format) =======================

TEST_CASE("Instance I/O: legacy 'S C A B' rows still parse") {
    std::istringstream in("V 1000\n0.1 0.2 0.3 1e9\n0.0 0.15 0.25 1e9\n");
    DLSInstance inst; std::string err;
    REQUIRE(readInstance(in, inst, err));
    CHECK(inst.totalLoad() == doctest::Approx(1000.0));
    REQUIRE(inst.numProcessors() == 2);
    CHECK(inst.processors()[0].commStartup == doctest::Approx(0.1));
    CHECK(inst.processors()[1].computeRate == doctest::Approx(0.25));
    CHECK(inst.resultFraction() == 0.0);
}

TEST_CASE("Instance I/O: rich fields, beta, and convex pieces parse") {
    std::istringstream in(
        "# rich instance\n"
        "V 500\n"
        "beta 0.5\n"
        "0.1 0.2 0.3 5000 5 10 200 1.0 0.5\n"   // S C A B p r d f l
        "0.0 0.0 0.0 5000\n"
        "pieces 0,0.2 -30,0.7\n");
    DLSInstance inst; std::string err;
    REQUIRE(readInstance(in, inst, err));
    CHECK(inst.totalLoad() == doctest::Approx(500.0));
    CHECK(inst.resultFraction() == doctest::Approx(0.5));
    const Processor& p0 = inst.processors()[0];
    CHECK(p0.computeStartup == doctest::Approx(5.0));
    CHECK(p0.releaseTime    == doctest::Approx(10.0));
    CHECK(p0.deadline       == doctest::Approx(200.0));
    CHECK(p0.fixedCost      == doctest::Approx(1.0));
    CHECK(p0.linearCost     == doctest::Approx(0.5));
    REQUIRE(inst.processors()[1].computePieces.size() == 2);
    CHECK(inst.processors()[1].computePieces[1].intercept == doctest::Approx(-30.0));
    CHECK(inst.processors()[1].computePieces[1].slope     == doctest::Approx(0.7));
}

TEST_CASE("Instance I/O: write then read round-trips every field") {
    Processor a; a.commStartup=0.1; a.commRate=0.2; a.computeRate=0.3; a.memoryLimit=5000;
    a.computeStartup=5; a.releaseTime=10; a.deadline=200; a.fixedCost=1.5; a.linearCost=0.25;
    Processor b; b.commRate=0.0; b.computeRate=0.0; b.memoryLimit=5000;
    b.computePieces = {{0.0,0.2},{-30.0,0.7}};
    DLSInstance orig({a, b}, 1234.5);
    orig.setResultFraction(0.4);

    std::ostringstream out; writeInstance(out, orig);
    std::istringstream in(out.str());
    DLSInstance rt; std::string err;
    REQUIRE(readInstance(in, rt, err));

    CHECK(rt.totalLoad() == doctest::Approx(orig.totalLoad()));
    CHECK(rt.resultFraction() == doctest::Approx(orig.resultFraction()));
    REQUIRE(rt.numProcessors() == 2);
    const Processor& r0 = rt.processors()[0];
    CHECK(r0.commStartup == doctest::Approx(a.commStartup));
    CHECK(r0.deadline    == doctest::Approx(a.deadline));
    CHECK(r0.linearCost  == doctest::Approx(a.linearCost));
    REQUIRE(rt.processors()[1].computePieces.size() == 2);
    CHECK(rt.processors()[1].computePieces[0].slope == doctest::Approx(0.2));
}

TEST_CASE("Instance I/O: structural errors are reported") {
    auto fails = [](const std::string& text) {
        std::istringstream in(text); DLSInstance inst; std::string err;
        return !readInstance(in, inst, err);
    };
    CHECK(fails("V 100\n0.1 0.2\n"));                 // too few columns
    CHECK(fails("V 100\npieces 0,1\n"));              // pieces before any processor
    CHECK(fails("V 100\n0.1 0.2 0.3 1e9\npieces 7\n")); // malformed piece (no comma)
    CHECK(fails("V 100\n"));                          // no processors
}

//=================== Per-class instance I/O (CLI --class) ==================

TEST_CASE("Class I/O: MLSD text parses and reaches the Example 1 optimum") {
    std::istringstream in("# Example 1\ntask 32\ntask 2\n"
                          "proc 1 1 1 1e18\nproc 1 1 1 1e18\nproc 1 1 1 1e18\n");
    MlsdInstance inst; std::string err;
    REQUIRE(readMlsdInstance(in, inst, err));
    CHECK(inst.numTasks() == 2);
    CHECK(inst.numProcessors() == 3);
    CHECK(inst.tasks()[0].size == doctest::Approx(32.0));
    CHECK(MlsdSolver().solve(inst).makespan == doctest::Approx(39.3333).epsilon(1e-3));
}

TEST_CASE("Class I/O: MapReduce text parses every field") {
    std::istringstream in("V 100\nstartup 1\nreadrate 2\ngamma0 0.5\nreducers 4\n"
                          "reducer_startup 0.1\nreducer_rate 1e-6\nmapper 1\nmapper 2\nmapper 3\n");
    MapReduceInstance inst; std::string err;
    REQUIRE(readMapReduceInstance(in, inst, err));
    CHECK(inst.totalLoad() == doctest::Approx(100.0));
    CHECK(inst.numMappers() == 3);
    CHECK(inst.readRate() == doctest::Approx(2.0));
    CHECK(inst.resultFraction() == doctest::Approx(0.5));
    CHECK(inst.numReducers() == 4);
    CHECK(inst.mapperRates()[2] == doctest::Approx(3.0));
    CHECK(MapReduceSolver().solve(inst).status == SolveStatus::Optimal);
}

TEST_CASE("Class I/O: Multilayer text parses fields and layers") {
    std::istringstream in("V 1024\nmappers 4\nmapper_rate 0.01\nreadrate 0.001\n"
                          "gamma0 1.0\nbisection 4\nlayer 2 0.1 1e-4 0.5\nlayer 3 0.2 2e-4 1.0\n");
    MultilayerInstance inst; std::string err;
    REQUIRE(readMultilayerInstance(in, inst, err));
    CHECK(inst.totalLoad == doctest::Approx(1024.0));
    CHECK(inst.numMappers == 4);
    CHECK(inst.bisectionWidth == 4);
    REQUIRE(inst.layers.size() == 2);
    CHECK(inst.layers[0].count == 2);
    CHECK(inst.layers[1].resultFraction == doctest::Approx(1.0));
    CHECK(MultilayerSolver().solve(inst).status == SolveStatus::Feasible);
}

TEST_CASE("Class I/O: structural errors are reported") {
    auto mlsdFails = [](const std::string& t) { std::istringstream in(t); MlsdInstance i; std::string e; return !readMlsdInstance(in, i, e); };
    auto mrFails   = [](const std::string& t) { std::istringstream in(t); MapReduceInstance i; std::string e; return !readMapReduceInstance(in, i, e); };
    auto mlFails   = [](const std::string& t) { std::istringstream in(t); MultilayerInstance i; std::string e; return !readMultilayerInstance(in, i, e); };
    CHECK(mlsdFails("task 5\n"));                 // no processors
    CHECK(mlsdFails("proc 1 1 1 1\n"));           // no tasks
    CHECK(mlsdFails("bogus 1\n"));                // unknown directive
    CHECK(mrFails("V 100\n"));                    // no mappers
    CHECK(mlFails("V 100\nmappers 2\n"));         // no layers
}

TEST_CASE("Class I/O: topology readers (chain / tree / graph)") {
    // Chain: parsed instance solves to the known anchor (α0=20, α1=10, T=20).
    {
        std::istringstream in("V 30\nnode 1 0\nnode 1 1\n");
        LinearChainInstance inst; std::string err;
        REQUIRE(readChainInstance(in, inst, err));
        CHECK(inst.numProcessors() == 2);
        ChainSolution s = LinearChainSolver().solve(inst);
        CHECK(s.makespan == doctest::Approx(20.0));
    }
    // Tree: parsed nodes match a directly-built tree's makespan.
    {
        std::istringstream in("V 1000\nnode 0.3 0 -1\nnode 0.25 0.2 0\nnode 0.4 0.15 0\nnode 0.5 0.3 1\n");
        TreeInstance inst; std::string err;
        REQUIRE(readTreeInstance(in, inst, err));
        REQUIRE(inst.numNodes() == 4);
        CHECK(inst.nodes()[3].parent == 1);
        double parsed = TreeSolver().solve(inst).makespan;
        double direct = TreeSolver().solve(TreeInstance({{0.3,0,-1},{0.25,0.2,0},{0.4,0.15,0},{0.5,0.3,1}}, 1000.0)).makespan;
        CHECK(parsed == doctest::Approx(direct));
    }
    // Graph: parsed mesh routes node 2 through node 1 (the cheap two-hop path).
    {
        std::istringstream in("V 30\nnode 1\nnode 1\nnode 1\nedge 0 1 0.1\nedge 1 2 0.1\nedge 0 2 10\n");
        GraphInstance inst; std::string err;
        REQUIRE(readGraphInstance(in, inst, err));
        CHECK(inst.numNodes() == 3);
        CHECK(inst.edges().size() == 3);
        GraphSolution s = GraphSolver().solve(inst);
        REQUIRE(s.status == SolveStatus::Optimal);
        CHECK(s.parent[2] == 1);
    }
    // Structural errors.
    auto chFails = [](const std::string& t) { std::istringstream in(t); LinearChainInstance i; std::string e; return !readChainInstance(in, i, e); };
    auto trFails = [](const std::string& t) { std::istringstream in(t); TreeInstance i; std::string e; return !readTreeInstance(in, i, e); };
    auto grFails = [](const std::string& t) { std::istringstream in(t); GraphInstance i; std::string e; return !readGraphInstance(in, i, e); };
    CHECK(chFails("V 10\n"));                      // no nodes
    CHECK(trFails("V 10\nnode 1 0\n"));            // node needs A C parent
    CHECK(grFails("V 10\nedge 0 1 1\n"));          // no nodes
}

//=================== Benchmark harness =====================================

static const dls::bench::SolverBenchResult& findResult(
        const dls::bench::BenchReport& rep, const std::string& name) {
    for (const auto& r : rep.solvers) if (r.name == name) return r;
    FAIL("benchmark result not found: ", name);
    return rep.solvers.front();
}

TEST_CASE("Benchmark: deterministic generation and aggregation") {
    using namespace dls::bench;
    BenchConfig cfg; cfg.numProcessors = 3; cfg.numInstances = 6; cfg.seed = 99;
    cfg.options.maxInstallments = 3;

    // Same seed => identical processors generated.
    util::Rng a(cfg.seed), b(cfg.seed);
    DLSInstance ia = generateInstance(a, cfg), ib = generateInstance(b, cfg);
    REQUIRE(ia.numProcessors() == ib.numProcessors());
    for (std::size_t i = 0; i < ia.numProcessors(); ++i)
        CHECK(ia.processors()[i].computeRate == ib.processors()[i].computeRate);

    // Same config => identical aggregated results (no hidden global RNG).
    auto r1 = runBenchmark({"single-round", "exact"}, cfg);
    auto r2 = runBenchmark({"single-round", "exact"}, cfg);
    CHECK(findResult(r1, "exact").avgMakespan == doctest::Approx(findResult(r2, "exact").avgMakespan));
    CHECK(findResult(r1, "single-round").avgRelGap == doctest::Approx(findResult(r2, "single-round").avgRelGap));
}

TEST_CASE("Benchmark: exact is no worse than the heuristics, all feasible, gaps >= 0") {
    using namespace dls::bench;
    BenchConfig cfg; cfg.numProcessors = 3; cfg.numInstances = 8; cfg.seed = 5;
    cfg.memoryFactor = 2.0;                      // unbounded => heuristics live in exact's space
    cfg.options.maxInstallments = 3; cfg.options.allowRepeats = true;

    auto rs = runBenchmark({"single-round", "best-rate", "ga", "exact"}, cfg);
    const double exMk = findResult(rs, "exact").avgMakespan;
    for (const char* n : {"single-round", "best-rate", "ga"})
        CHECK(exMk <= findResult(rs, n).avgMakespan + 1e-6);   // per-instance optimality holds on average

    // Exact is optimal over the searched space, so it IS the per-instance best:
    // its distance from the virtual best is zero, and every other gap is >= 0.
    CHECK(findResult(rs, "exact").avgGapToBest == doctest::Approx(0.0));
    for (const auto& r : rs.solvers) {
        CHECK(r.feasible == r.total);                          // every instance solved
        CHECK(r.avgRelGap   >= -1e-9);                         // Cmax >= LB (valid bound)
        CHECK(r.avgGapToBest >= -1e-9);                        // nobody beats the virtual best
    }
}

TEST_CASE("Benchmark: an unavailable solver is reported with total = 0") {
    using namespace dls::bench;
    BenchConfig cfg; cfg.numProcessors = 2; cfg.numInstances = 2;
    auto rs = runBenchmark({"single-round", "not-a-solver"}, cfg);
    CHECK(findResult(rs, "single-round").total == 2);
    CHECK(findResult(rs, "not-a-solver").total == 0);
}

TEST_CASE("Benchmark: proven-optimum reference is status-aware") {
    using namespace dls::bench;
    BenchConfig cfg; cfg.numProcessors = 3; cfg.numInstances = 6; cfg.seed = 11;
    cfg.options.maxInstallments = 3; cfg.options.allowRepeats = true;

    // With an exact solver in the run, every instance has a PROVEN optimum, and
    // the exact solver's gap to that optimum is zero.
    BenchReport withExact = runBenchmark({"single-round", "exact"}, cfg);
    CHECK(withExact.provenOptimal == withExact.instances);
    CHECK(findResult(withExact, "exact").provenScored == withExact.instances);
    CHECK(findResult(withExact, "exact").avgGapToOpt == doctest::Approx(0.0));
    CHECK(findResult(withExact, "single-round").avgGapToOpt >= -1e-9);   // rigorous gap from optimum

    // With only heuristics, nothing is proven: no proven references, no gap→opt.
    BenchReport heurOnly = runBenchmark({"single-round", "best-rate"}, cfg);
    CHECK(heurOnly.provenOptimal == 0);
    CHECK(findResult(heurOnly, "single-round").provenScored == 0);
}

TEST_CASE("Benchmark: on canonical (homogeneous) instances the closed form is optimal") {
    using namespace dls::bench;
    BenchConfig cfg; cfg.numProcessors = 4; cfg.numInstances = 8; cfg.seed = 31;
    cfg.homogeneous = true;                       // regime 1: closed-form-tractable
    cfg.memoryFactor = 2.0;                       // ample memory
    cfg.options.maxInstallments = 4; cfg.options.allowRepeats = true;

    // Generated processors are identical.
    util::Rng rng(cfg.seed);
    DLSInstance inst = generateInstance(rng, cfg);
    for (std::size_t i = 1; i < inst.numProcessors(); ++i) {
        CHECK(inst.processors()[i].commStartup == inst.processors()[0].commStartup);
        CHECK(inst.processors()[i].computeRate == inst.processors()[0].computeRate);
    }

    // The exact B&B proves the optimum; the closed-form single-round recovers it
    // (the "validate on tractable cases first" check) — gap to optimum ~ 0.
    BenchReport rep = runBenchmark({"single-round", "exact"}, cfg);
    CHECK(rep.provenOptimal == rep.instances);
    CHECK(rep.regime1 == rep.instances);                         // tagged regime 1
    CHECK(rep.regime2 == 0);
    CHECK(findResult(rep, "exact").avgGapToOpt == doctest::Approx(0.0));
    CHECK(findResult(rep, "single-round").avgGapToOpt < 1e-6);   // closed form == proven optimum
}

TEST_CASE("Benchmark: regime 1 is proven by the closed form WITHOUT running the B&B") {
    using namespace dls::bench;
    BenchConfig cfg; cfg.numProcessors = 4; cfg.numInstances = 6; cfg.seed = 41;
    cfg.homogeneous = true; cfg.memoryFactor = 2.0;
    cfg.options.maxInstallments = 4; cfg.options.allowRepeats = true;

    // No exact solver in the run — only heuristics. On regime-1 the closed form
    // still supplies the proven optimum, so gap→opt is reported cheaply.
    BenchReport rep = runBenchmark({"single-round", "best-rate", "ga"}, cfg);
    CHECK(rep.regime1 == rep.instances);
    CHECK(rep.provenOptimal == rep.instances);                   // proven without the B&B
    CHECK(findResult(rep, "single-round").provenScored == rep.instances);
    CHECK(findResult(rep, "single-round").avgGapToOpt == doctest::Approx(0.0));
    CHECK(findResult(rep, "ga").avgGapToOpt >= -1e-9);

    // And the closed-form reference equals the exact B&B optimum (so trusting it
    // is correct): exact's gap→opt against that same reference is ~0.
    BenchReport withExact = runBenchmark({"single-round", "exact"}, cfg);
    CHECK(findResult(withExact, "exact").avgGapToOpt == doctest::Approx(0.0));

    // Heterogeneous (regime 2): no closed-form proof; nothing proven without exact.
    BenchConfig het = cfg; het.homogeneous = false;
    BenchReport rh = runBenchmark({"single-round", "best-rate"}, het);
    CHECK(rh.regime2 == rh.instances);
    CHECK(rh.regime1 == 0);
    CHECK(rh.provenOptimal == 0);
}

TEST_CASE("Benchmark: labels + policy scoring (certify against the optimum / LB)") {
    using namespace dls::bench;
    BenchConfig cfg; cfg.numProcessors = 3; cfg.numInstances = 6; cfg.seed = 21;
    cfg.options.maxInstallments = 3; cfg.options.allowRepeats = true;

    std::vector<InstanceLabel> labels = generateLabeledInstances(cfg, "exact");
    REQUIRE(labels.size() == 6);
    for (const InstanceLabel& l : labels) {
        CHECK(l.proven);                                   // exact + ample memory => proved
        CHECK(l.optMakespan > 0.0);
        CHECK(l.lowerBound <= l.optMakespan + 1e-6);       // valid floor
    }

    // A policy that returns exactly M* scores a zero optimality gap.
    std::vector<double> optimalPolicy;
    for (const InstanceLabel& l : labels) optimalPolicy.push_back(l.optMakespan);
    PolicyScore s0 = scorePolicy(labels, optimalPolicy);
    CHECK(s0.scoredAgainstOpt == 6);
    CHECK(s0.avgGapToOpt == doctest::Approx(0.0));

    // A policy 10% above optimal scores ~10% gap to opt, and a positive LB gap.
    std::vector<double> worse;
    for (const InstanceLabel& l : labels) worse.push_back(l.optMakespan * 1.10);
    PolicyScore s1 = scorePolicy(labels, worse);
    CHECK(s1.avgGapToOpt == doctest::Approx(0.10).epsilon(1e-6));
    CHECK(s1.maxGapToOpt == doctest::Approx(0.10).epsilon(1e-6));
    CHECK(s1.avgGapToLB >= s1.avgGapToOpt - 1e-9);          // LB gap is >= the (tighter) opt gap
}

TEST_CASE("Benchmark: label CSV round-trips, and the score path is end-to-end") {
    using namespace dls::bench;
    BenchConfig cfg; cfg.numProcessors = 3; cfg.numInstances = 5; cfg.seed = 33;
    cfg.options.maxInstallments = 3; cfg.options.allowRepeats = true;

    std::vector<InstanceLabel> labels = generateLabeledInstances(cfg, "exact");

    // writeLabels -> readLabels recovers M*, proven, LB.
    std::ostringstream out; writeLabels(out, labels);
    std::istringstream in(out.str());
    std::vector<InstanceLabel> back = readLabels(in);
    REQUIRE(back.size() == labels.size());
    for (std::size_t k = 0; k < labels.size(); ++k) {
        CHECK(back[k].optMakespan == doctest::Approx(labels[k].optMakespan));
        CHECK(back[k].proven == labels[k].proven);
        CHECK(back[k].lowerBound == doctest::Approx(labels[k].lowerBound));
    }

    // readPolicyMakespans: takes the last comma field, skips a header line.
    std::istringstream pin("makespan\n5.5\n0,6.25\n");
    std::vector<double> pm = readPolicyMakespans(pin);
    REQUIRE(pm.size() == 2);
    CHECK(pm[0] == doctest::Approx(5.5));
    CHECK(pm[1] == doctest::Approx(6.25));

    // End-to-end through the disk format: optimal policy => 0%, 8%-above => 8%.
    std::vector<double> opt, hi;
    for (const InstanceLabel& l : back) { opt.push_back(l.optMakespan); hi.push_back(l.optMakespan * 1.08); }
    CHECK(scorePolicy(back, opt).avgGapToOpt == doctest::Approx(0.0));
    CHECK(scorePolicy(back, hi).avgGapToOpt == doctest::Approx(0.08).epsilon(1e-6));
}

TEST_CASE("Benchmark: gap->LB is a valid certificate (>= the true optimality gap)") {
    using namespace dls::bench;
    BenchConfig cfg; cfg.numProcessors = 4; cfg.numInstances = 12; cfg.seed = 51;
    cfg.options.maxInstallments = 4; cfg.options.allowRepeats = true;

    // On instances with a proven optimum, the certified bound (gap->LB) must be
    // at least the true optimality gap (gap->opt) for every solver — that is the
    // whole point: it bounds the optimum gap from above without knowing it.
    BenchReport rep = runBenchmark({"single-round", "best-rate", "ga", "exact"}, cfg);
    REQUIRE(rep.provenOptimal == rep.instances);
    for (const SolverBenchResult& r : rep.solvers) {
        CHECK(r.maxGapToLB >= -1e-9);                       // a valid (non-negative) bound
        CHECK(r.maxGapToLB >= r.avgRelGap - 1e-9);          // worst >= average
        CHECK(r.maxGapToLB >= r.avgGapToOpt - 1e-9);        // certificate dominates the true gap
    }

    // Policy certificate: maxGapToLB >= maxGapToOpt (the certified worst case is
    // never below the true worst-case optimality gap).
    std::vector<InstanceLabel> labels = generateLabeledInstances(cfg, "exact");
    std::vector<double> policy; for (const InstanceLabel& l : labels) policy.push_back(l.optMakespan * 1.05);
    PolicyScore sc = scorePolicy(labels, policy);
    CHECK(sc.maxGapToLB >= sc.maxGapToOpt - 1e-9);
    CHECK(sc.maxGapToLB >= sc.avgGapToLB - 1e-9);
}

//=================== Single-round closed form (classic DLT / §2.4) =========

TEST_CASE("SingleRound: analytic anchors (one and two processors, S=0)") {
    DLSInstance one({{0.0, 0.11, 0.12, 1e9}}, 1000.0);          // (C+A)·V = 230
    DLSSolution s1 = SingleRoundSolver().solve(one, SolverConfig{});
    REQUIRE(s1.status == SolveStatus::Feasible);
    CHECK(s1.makespan == doctest::Approx(230.0));
    CHECK(s1.conservesLoad(1000.0, 1e-4));

    const double C = 0.1, A = 0.2, V = 1000.0;                  // (C+A)²V/(C+2A) = 180
    DLSInstance two({{0.0, C, A, 1e9}, {0.0, C, A, 1e9}}, V);
    DLSSolution s2 = SingleRoundSolver().solve(two, SolverConfig{});
    CHECK(s2.makespan == doctest::Approx((C + A) * (C + A) * V / (C + 2 * A)));
}

TEST_CASE("SingleRound equals the exact B&B optimum when S=0") {
    std::vector<DLSInstance> insts = {
        DLSInstance({{0.0,0.23,0.12,1e9},{0.0,0.11,0.23,1e9},{0.0,0.11,0.31,1e9}}, 5000.0),
        DLSInstance({{0.0,0.20,0.30,1e9},{0.0,0.15,0.25,1e9},{0.0,0.25,0.40,1e9}}, 8000.0),
    };
    for (DLSInstance& inst : insts) {
        DLSSolution sr = SingleRoundSolver().solve(inst, SolverConfig{});
        ExactParams ep; ep.maxInstallments = (int)inst.numProcessors(); ep.allowRepeats = false;
        DLSSolution ex = ExactSolver(ep).solve(inst, SolverConfig{});
        REQUIRE(sr.status == SolveStatus::Feasible);
        CHECK(sr.conservesLoad(inst.totalLoad(), 1e-3));
        CHECK(sr.makespan == doctest::Approx(ex.makespan).epsilon(1e-4));   // exact for S=0
    }
}

TEST_CASE("SingleRound: its loads match the LP on the same ordered sequence") {
    DLSInstance inst({{0.0,0.23,0.12,1e9},{0.0,0.11,0.23,1e9},{0.0,0.11,0.31,1e9}}, 5000.0);
    DLSSolution sr = SingleRoundSolver().solve(inst, SolverConfig{});
    SimplexScheduleEvaluator ev;
    DLSSolution lp = ev.evaluate(inst, sr.sequence, EvaluatorConfig{});   // optimal loads for that order
    REQUIRE(lp.status == SolveStatus::Optimal);
    CHECK(sr.makespan == doctest::Approx(lp.makespan).epsilon(1e-6));     // equal-completion == LP optimum
}

TEST_CASE("SingleRound: non-decreasing Cᵢ order beats the reverse") {
    DLSInstance inst({{0.0, 0.5, 0.2, 1e9}, {0.0, 0.1, 0.2, 1e9}}, 100.0);   // P1 slow comm
    DLSSolution sr = SingleRoundSolver().solve(inst, SolverConfig{});
    SimplexScheduleEvaluator ev;
    std::vector<int> rev(sr.sequence.rbegin(), sr.sequence.rend());
    double revMakespan = ev.evaluate(inst, rev, EvaluatorConfig{}).makespan;
    CHECK(sr.sequence.front() == 1);                              // smaller Cᵢ activated first
    CHECK(sr.makespan <= revMakespan + 1e-9);
}

//=================== Homogeneous closed form (optimal k*) =================

TEST_CASE("Homogeneous: closed-form makespan matches the analytic anchors") {
    // 1 processor S=0, C=0.11, A=0.12, V=1000 => (C+A)·V = 230.
    CHECK(homogeneousSingleRound(0.0, 0.11, 0.12, 1000.0, 1).makespan == doctest::Approx(230.0));
    // 2 identical S=0, C=0.1, A=0.2, V=1000 => (C+A)²V/(C+2A) = 180.
    CHECK(homogeneousSingleRound(0.0, 0.1, 0.2, 1000.0, 2).makespan == doctest::Approx(180.0));
    // C = 0 branch: 2 procs S=0, A=1, V=10 => each computes 5 in parallel => 5.
    CHECK(homogeneousSingleRound(0.0, 0.0, 1.0, 10.0, 2).makespan == doctest::Approx(5.0));
}

TEST_CASE("Homogeneous: more processors help, then the marginal load goes negative") {
    const double S = 1.0, C = 0.1, A = 0.2, V = 100.0;
    double prev = homogeneousSingleRound(S, C, A, V, 1).makespan;
    for (int k = 2; k <= 5; ++k) {
        HomogeneousRound r = homogeneousSingleRound(S, C, A, V, k);
        if (r.minLoad < 0.0) break;                       // beyond k*: infeasible
        CHECK(r.makespan <= prev + 1e-9);                 // using more processors never hurts
        prev = r.makespan;
    }
    HomogeneousOptimum opt = homogeneousOptimalProcessors(S, C, A, V, 100);
    CHECK(opt.kStar >= 1);
    CHECK(homogeneousSingleRound(S, C, A, V, opt.kStar).minLoad >= -1e-9);   // k* is feasible
}

TEST_CASE("Homogeneous closed form == SingleRound == exact B&B on identical processors") {
    const double S = 0.2, C = 0.15, A = 0.25, V = 5000.0;
    const int m = 6;
    HomogeneousOptimum opt = homogeneousOptimalProcessors(S, C, A, V, m);

    std::vector<Processor> ps(m, Processor{S, C, A, 1e18, 0,0,0,0,0, {}, 0,0,0, {}, ""});
    DLSInstance inst(ps, V);
    DLSSolution sr = SingleRoundSolver().solve(inst, SolverConfig{});
    ExactParams ep; ep.maxInstallments = m; ep.allowRepeats = false;
    DLSSolution bnb = ExactSolver(ep).solve(inst, SolverConfig{});

    CHECK(opt.makespan == doctest::Approx(sr.makespan).epsilon(1e-4));   // closed form == solver
    CHECK(opt.makespan == doctest::Approx(bnb.makespan).epsilon(1e-3));  // == exact optimum
    CHECK(opt.kStar == (int)sr.sequence.size());                         // same processor count
}

//=================== Best Rate heuristic + lower bound (Berlińska) =========

TEST_CASE("Lower bound (eq. 3.23) is analytic on a pure-compute instance") {
    // S=C=0, two processors A=1, V=10, ample memory. No communication, so the
    // optimum is the fluid bound V/(Σ 1/Aᵢ) = 10/2 = 5, and LB must equal it.
    DLSInstance inst({{0.0,0.0,1.0,1e9},{0.0,0.0,1.0,1e9}}, 10.0);
    CHECK(divisibleLoadLowerBound(inst) == doctest::Approx(5.0));
}

TEST_CASE("Lower bound never exceeds the exact optimum") {
    std::vector<DLSInstance> insts = {
        canonical5(),
        DLSInstance({{0.5,0.20,0.30,1e9},{0.7,0.15,0.25,1e9},{0.3,0.25,0.40,1e9}}, 10000),
        DLSInstance({{0.0,0.23,0.12,1e6},{0.1,0.11,0.23,1e6},{0.2,0.11,0.31,1e6}}, 5000),
    };
    for (DLSInstance& inst : insts) {
        ExactParams ep; ep.maxInstallments = (int)inst.numProcessors(); ep.allowRepeats = false;
        DLSSolution opt = ExactSolver(ep).solve(inst, SolverConfig{});
        REQUIRE(opt.feasible());
        CHECK(divisibleLoadLowerBound(inst) <= opt.makespan + 1e-6);        // valid (eq. 3.23)
        CHECK(divisibleLoadLowerBoundTight(inst) <= opt.makespan + 1e-6);   // valid (tight)
    }
}

TEST_CASE("Tight lower bound is valid, >= eq. 3.23, and analytic on a clean case") {
    // Single processor S=2, C=0, A=1, V=10: no port contention to relax away, so
    // the port-relaxed bound is exactly the optimum S + V·A = 12.
    DLSInstance one({{2.0, 0.0, 1.0, 1e9}}, 10.0);
    CHECK(portRelaxedLowerBound(one) == doctest::Approx(12.0));
    CHECK(divisibleLoadLowerBoundTight(one) == doctest::Approx(12.0));

    // The tight bound dominates eq. 3.23 and the fluid bound everywhere.
    std::vector<DLSInstance> insts = {
        canonical5(),
        DLSInstance({{0.5,0.20,0.30,1e9},{0.7,0.15,0.25,1e9},{0.3,0.25,0.40,1e9}}, 10000),
    };
    for (DLSInstance& inst : insts) {
        const double tight = divisibleLoadLowerBoundTight(inst);
        CHECK(tight >= divisibleLoadLowerBound(inst) - 1e-9);
        CHECK(tight >= fluidLowerBound(inst) - 1e-9);
        ExactParams ep; ep.maxInstallments = (int)inst.numProcessors(); ep.allowRepeats = false;
        CHECK(tight <= ExactSolver(ep).solve(inst, SolverConfig{}).makespan + 1e-6);
    }
}

TEST_CASE("BestRateSolver: feasible, conserves load, and lies in [LB, exact]") {
    std::vector<DLSInstance> insts = {
        canonical5(),
        DLSInstance({{0.5,0.20,0.30,1e9},{0.7,0.15,0.25,1e9},{0.3,0.25,0.40,1e9}}, 10000),
    };
    for (DLSInstance& inst : insts) {
        DLSSolution br = BestRateSolver().solve(inst, SolverConfig{});
        CHECK(br.status == SolveStatus::Feasible);
        CHECK(br.conservesLoad(inst.totalLoad(), 1.0));
        // Bracketed by the lower bound below and the (multi-installment) exact
        // optimum above — the heuristic is valid but not necessarily optimal.
        ExactParams ep; ep.maxInstallments = (int)inst.numProcessors(); ep.allowRepeats = true;
        double opt = ExactSolver(ep).solve(inst, SolverConfig{}).makespan;
        CHECK(br.makespan >= divisibleLoadLowerBound(inst) - 1e-6);
        CHECK(br.makespan >= opt - 1e-6);     // exact optimum can't be beaten
    }
}

TEST_CASE("BestRateSolver is deterministic and respects memory limits") {
    // Small buffers force a multi-chunk sequence; every fragment must fit Bᵢ.
    DLSInstance inst({{0.1,0.2,0.3,1500.0},{0.2,0.15,0.25,1500.0},{0.0,0.25,0.4,1500.0}}, 4000.0);
    DLSSolution a = BestRateSolver().solve(inst, SolverConfig{});
    DLSSolution b = BestRateSolver().solve(inst, SolverConfig{});
    REQUIRE(a.status == SolveStatus::Feasible);
    CHECK(a.makespan == doctest::Approx(b.makespan));            // no RNG => identical
    CHECK(a.conservesLoad(inst.totalLoad(), 1.0));
    for (const LoadFragment& f : a.fragments)
        CHECK(f.loadSize <= inst.processors()[f.processorId].memoryLimit + 1e-6);
}

TEST_CASE("BestRateSolver drops a uselessly slow-to-start processor") {
    // P1 has a huge startup; for a small load the greedy/LP must avoid it.
    DLSInstance inst({{100.0, 1.0, 1.0, 1e9}, {0.0, 1.0, 1.0, 1e9}}, 10.0);
    DLSSolution br = BestRateSolver().solve(inst, SolverConfig{});
    CHECK(br.status == SolveStatus::Feasible);
    CHECK(br.makespan == doctest::Approx(20.0));                 // only P2: (C+A)*V
}

TEST_CASE("Analytic: one processor, one installment => Cmax = (C+A)*V") {
    // S=0, single installment: comm C*V then compute A*V, no overlap.
    DLSInstance inst({{0.0, 0.11, 0.12, 1e9}}, 1000.0);
    SimplexScheduleEvaluator ev; EvaluatorConfig cfg;
    DLSSolution s = ev.evaluate(inst, {0}, cfg);
    CHECK(s.status == SolveStatus::Optimal);
    CHECK(s.makespan == doctest::Approx((0.11 + 0.12) * 1000.0));   // 230
}

TEST_CASE("Analytic: two identical processors, S=0 => closed-form split") {
    // Balanced single-port split: Cmax = (C+A)^2 * V / (C + 2A).
    const double C = 0.1, A = 0.2, V = 1000.0;
    DLSInstance inst({{0.0, C, A, 1e9}, {0.0, C, A, 1e9}}, V);
    const double expected = (C + A) * (C + A) * V / (C + 2 * A);   // = 180

    SimplexScheduleEvaluator ev; EvaluatorConfig cfg;
    CHECK(ev.evaluate(inst, {0,1}, cfg).makespan == doctest::Approx(expected));

    ExactParams ep; ep.maxInstallments = 2; ep.allowRepeats = false;   // single-installment
    DLSSolution s = ExactSolver(ep).solve(inst, SolverConfig{});
    CHECK(s.status == SolveStatus::Optimal);
    CHECK(s.makespan == doctest::Approx(expected));
}

//=================== validation: piecewise-linear convex processing ========
// Processing time of load q is max over pieces of (intercept + slope*q),
// modeling a memory hierarchy (fast core piece + steeper disk piece).

TEST_CASE("Piecewise: swap-point closed form on a single processor") {
    // S=C=0. Core piece: 0 + 1*q. Disk piece: -10 + 3*q. They cross at q=5.
    // Below the swap the core piece binds; above it the disk piece binds.
    Processor p; p.commStartup = 0.0; p.commRate = 0.0; p.computeRate = 0.0;
    p.memoryLimit = 1e9; p.computePieces = {{0.0, 1.0}, {-10.0, 3.0}};
    SimplexScheduleEvaluator ev; EvaluatorConfig cfg;

    DLSInstance below({p}, 4.0);     // q=4 < 5 => proc = max(4, 2) = 4
    CHECK(ev.evaluate(below, {0}, cfg).makespan == doctest::Approx(4.0));

    DLSInstance above({p}, 10.0);    // q=10 > 5 => proc = max(10, 20) = 20
    CHECK(ev.evaluate(above, {0}, cfg).makespan == doctest::Approx(20.0));

    DLSInstance atSwap({p}, 5.0);    // exactly at swap => both give 5
    CHECK(ev.evaluate(atSwap, {0}, cfg).makespan == doctest::Approx(5.0));
}

TEST_CASE("Piecewise: duplicated single piece reproduces the classic model") {
    // The proc path (active with >1 piece) must reduce to the inline affine
    // model when the pieces coincide — checked through the y-coupling of a
    // multi-installment sequence so the (alpha+y) handling is exercised.
    SimplexScheduleEvaluator ev; EvaluatorConfig cfg;
    const std::vector<int> seq = {0, 1, 0};   // P0 carries load across installments

    Processor base; base.commStartup = 0.1; base.commRate = 0.1;
    base.computeRate = 0.2; base.computeStartup = 0.5; base.memoryLimit = 1e9;
    DLSInstance classic({base, base}, 1000.0);
    const double ref = ev.evaluate(classic, seq, cfg).makespan;

    Processor piece = base;                    // same line, stated as two pieces
    piece.computePieces = {{base.computeStartup, base.computeRate},
                           {base.computeStartup, base.computeRate}};
    DLSInstance pieced({piece, piece}, 1000.0);
    CHECK(ev.evaluate(pieced, seq, cfg).makespan == doctest::Approx(ref));
}

TEST_CASE("Piecewise: a dominated extra piece never changes the result") {
    // Adding a piece that lies strictly below the core line for all q>=0
    // (slope 0, intercept 0) leaves the makespan equal to the affine model.
    SimplexScheduleEvaluator ev; EvaluatorConfig cfg;
    DLSInstance affine({{0.0, 0.0, 1.0, 1e9}}, 10.0);          // A=1 => Cmax=10
    CHECK(ev.evaluate(affine, {0}, cfg).makespan == doctest::Approx(10.0));

    Processor p; p.commStartup = 0.0; p.commRate = 0.0; p.computeRate = 0.0;
    p.memoryLimit = 1e9; p.computePieces = {{0.0, 1.0}, {0.0, 0.0}};  // 2nd piece dominated
    DLSInstance pieced({p}, 10.0);
    CHECK(ev.evaluate(pieced, {0}, cfg).makespan == doctest::Approx(10.0));
}

TEST_CASE("Piecewise: validate rejects a negative-slope piece") {
    Processor p; p.commRate = 1.0;
    p.computePieces = {{0.0, 1.0}, {5.0, -0.5}};   // decreasing piece is invalid
    DLSInstance inst({p}, 10.0);
    std::string err;
    CHECK_FALSE(inst.validate(&err));
    CHECK(err.find("negative slope") != std::string::npos);
}

//=================== validation: worst-case heterogeneity (thesis) =========
// The thesis shows naive strategies are arbitrarily bad in heterogeneous
// systems. These assert the exact solver achieves the GOOD (optimal) value,
// validating that load sizing, processor selection, and ordering all matter.

TEST_CASE("Worst-case sizing: optimal beats equal partitioning") {
    // P1 slow (A=10), P2 fast (A=1), no comm. Equal split => Cmax=5000;
    // optimal balances => Cmax = A1*V/(A1+1) = 10000/11.
    DLSInstance inst({{0.0, 0.0, 10.0, 1e9}, {0.0, 0.0, 1.0, 1e9}}, 1000.0);
    ExactParams ep; ep.maxInstallments = 2; ep.allowRepeats = false;
    DLSSolution s = ExactSolver(ep).solve(inst, SolverConfig{});
    CHECK(s.makespan == doctest::Approx(10.0 * 1000.0 / 11.0));   // ~909.09, << 5000
}

TEST_CASE("Worst-case selection: optimal drops a uselessly slow-to-start processor") {
    // P1 has huge startup S=100; for small V it must not be used.
    // Optimal uses only P2: Cmax = (C+A)*V = 20.
    DLSInstance inst({{100.0, 1.0, 1.0, 1e9}, {0.0, 1.0, 1.0, 1e9}}, 10.0);
    ExactParams ep; ep.maxInstallments = 2; ep.allowRepeats = false;
    DLSSolution s = ExactSolver(ep).solve(inst, SolverConfig{});
    CHECK(s.makespan == doctest::Approx(20.0));                   // only P2, not >= 100
}

TEST_CASE("Worst-case ordering: optimal picks the better activation order") {
    // P1 has high comm rate C=10. Order (P2,P1) is far better than (P1,P2).
    // Optimal Cmax = (2C1+2)/(C1+1.5) with C1=10 => 22/11.5.
    DLSInstance inst({{0.0, 10.0, 1.0, 1e9}, {0.0, 0.5, 0.5, 1e9}}, 2.0);
    ExactParams ep; ep.maxInstallments = 2; ep.allowRepeats = false;
    DLSSolution s = ExactSolver(ep).solve(inst, SolverConfig{});
    CHECK(s.makespan == doctest::Approx((2 * 10.0 + 2) / (10.0 + 1.5)));  // ~1.913, << 11
}

//=================== validation: availability conditions (p, r, d) =========
// Processor fields {S,C,A,B,p,r,d}. Single-processor cases give exact answers.

TEST_CASE("Condition p: computation startup adds to the makespan") {
    // One processor, S=C=0, A=1, p=5, V=10 => Cmax = p + A*V = 15.
    DLSInstance inst({{0.0, 0.0, 1.0, 1e9, 5.0, 0.0, 0.0}}, 10.0);
    SimplexScheduleEvaluator ev; EvaluatorConfig cfg;
    CHECK(ev.evaluate(inst, {0}, cfg).makespan == doctest::Approx(15.0));
}

TEST_CASE("Condition r: release time delays a processor that would finish earlier") {
    // S=C=0, A=1, V=10 => data ready at 0, compute 10. With release r=7 the
    // processor cannot start before 7, so Cmax = r + A*V = 17 (not 10).
    DLSInstance inst({{0.0, 0.0, 1.0, 1e9, 0.0, 7.0, 0.0}}, 10.0);
    SimplexScheduleEvaluator ev; EvaluatorConfig cfg;
    CHECK(ev.evaluate(inst, {0}, cfg).makespan == doctest::Approx(17.0));
}

TEST_CASE("Condition d: a deadline forces load away from the constrained processor") {
    // P0 (A=1) with deadline d=4 can take at most 4 units; P1 (A=2) takes the
    // rest. No deadline => balanced Cmax=20/3; with d=4 => alpha0=4, Cmax=12.
    SimplexScheduleEvaluator ev; EvaluatorConfig cfg;
    DLSInstance free({{0.0,0.0,1.0,1e9,0.0,0.0,0.0},{0.0,0.0,2.0,1e9,0.0,0.0,0.0}}, 10.0);
    CHECK(ev.evaluate(free, {0,1}, cfg).makespan == doctest::Approx(20.0 / 3.0));

    DLSInstance dl({{0.0,0.0,1.0,1e9,0.0,0.0,4.0},{0.0,0.0,2.0,1e9,0.0,0.0,0.0}}, 10.0);
    CHECK(ev.evaluate(dl, {0,1}, cfg).makespan == doctest::Approx(12.0));
}

//=================== validation: cost / bi-criteria (thesis m=4) ===========
// The thesis's worked single-installment example: 4 processors with all fields
// {S,C,A,B,p,r,d,f,l}, V=20, sequence (P1,P2,P3,P4). Lowering the cost limit
// G-bar raises Cmax (known reference values).

TEST_CASE("Cost/bi-criteria: thesis m=4 example reproduces reference Cmax per G-bar") {
    DLSInstance inst = thesisM4();
    SimplexScheduleEvaluator ev;
    const std::vector<int> seq = {0,1,2,3};

    auto cmaxFor = [&](double Gbar) {
        EvaluatorConfig cfg; cfg.costLimit = Gbar;
        return ev.evaluate(inst, seq, cfg);
    };

    // Cost limit inactive (large G-bar): the min-Cmax schedule.
    DLSSolution s0 = cmaxFor(1e9);
    CHECK(s0.status == SolveStatus::Optimal);
    CHECK(s0.makespan == doctest::Approx(26.3333).epsilon(1e-3));

    // Tightening the cost limit raises Cmax (reference values from the thesis).
    CHECK(cmaxFor(24.25).makespan   == doctest::Approx(41.5).epsilon(1e-3));
    CHECK(cmaxFor(24.1334).makespan == doctest::Approx(60.656).epsilon(1e-3));

    // Below the feasibility threshold the schedule is infeasible.
    CHECK(cmaxFor(24.0).status == SolveStatus::Infeasible);
}

TEST_CASE("Bi-criteria ExactSolver: tighter cost limit never lowers the optimal Cmax") {
    DLSInstance inst = thesisM4();
    auto solveWith = [&](double Gbar) {
        ExactParams ep; ep.maxInstallments = 4; ep.allowRepeats = false; ep.costLimit = Gbar;
        return ExactSolver(ep).solve(inst, SolverConfig{});
    };
    DLSSolution loose = solveWith(1e9);     // unconstrained
    DLSSolution tight = solveWith(24.5);    // constrained
    CHECK(loose.status == SolveStatus::Optimal);
    CHECK(tight.status == SolveStatus::Optimal);
    CHECK(tight.cost <= 24.5 + 1e-3);                  // limit respected
    CHECK(tight.makespan >= loose.makespan - 1e-6);    // cost limit can only hurt makespan
}

TEST_CASE("Reverse bi-criteria: min G subject to Cmax <= C-bar (thesis m=4 Pareto)") {
    DLSInstance inst = thesisM4();
    SimplexScheduleEvaluator ev;
    auto minCostFor = [&](double Cbar) {
        EvaluatorConfig cfg; cfg.objective = EvalObjective::MinCost; cfg.makespanLimit = Cbar;
        return ev.evaluate(inst, {0,1,2,3}, cfg);
    };
    // Same Pareto points as the forward direction, queried the other way:
    DLSSolution a = minCostFor(41.5);
    CHECK(a.status == SolveStatus::Optimal);
    CHECK(a.makespan <= 41.5 + 1e-3);
    CHECK(a.cost == doctest::Approx(24.25).epsilon(1e-3));
    CHECK(minCostFor(60.656).cost == doctest::Approx(24.1334).epsilon(1e-3));
    // Below the minimum achievable Cmax (26.333) it is infeasible.
    CHECK(minCostFor(20.0).status == SolveStatus::Infeasible);
}

TEST_CASE("Cost: G is reported and the cost limit is respected") {
    DLSInstance inst = thesisM4();
    SimplexScheduleEvaluator ev;
    EvaluatorConfig cfg; cfg.costLimit = 24.25;
    DLSSolution s = ev.evaluate(inst, {0,1,2,3}, cfg);
    REQUIRE(s.status == SolveStatus::Optimal);
    CHECK(s.cost <= 24.25 + 1e-3);                 // limit respected
    CHECK(s.cost == doctest::Approx(24.25).epsilon(1e-3));  // binding here
}

//=================== validation: result return (β, model A) ================
// Model A: non-overlapping, FIFO collection after distribution.

TEST_CASE("Result return: single processor closed form (send + compute + return)") {
    // S=0, C=1, A=1, V=10, beta=0.5: send 10 (Cα=10), compute 10 (Aα=10),
    // return Cβα = 5 => Cmax = 10 + 10 + 5 = 25 (vs 20 without return).
    DLSInstance inst({{0.0, 1.0, 1.0, 1e9}}, 10.0);
    SimplexScheduleEvaluator ev; EvaluatorConfig cfg;
    CHECK(ev.evaluate(inst, {0}, cfg).makespan == doctest::Approx(20.0));   // beta=0 default
    inst.setResultFraction(0.5);
    CHECK(ev.evaluate(inst, {0}, cfg).makespan == doctest::Approx(25.0));   // with return
}

TEST_CASE("Result return: larger beta never lowers the makespan; load conserved") {
    DLSInstance inst({{0.0,0.2,0.3,1e9},{0.1,0.15,0.25,1e9},{0.2,0.25,0.4,1e9}}, 5000.0);
    SimplexScheduleEvaluator ev; EvaluatorConfig cfg;
    double m0 = ev.evaluate(inst, {0,1,2}, cfg).makespan;          // no return
    inst.setResultFraction(1.0);
    DLSSolution s = ev.evaluate(inst, {0,1,2}, cfg);
    CHECK(s.status == SolveStatus::Optimal);
    CHECK(s.makespan >= m0 - 1e-6);                                // return only adds time
    CHECK(s.conservesLoad(inst.totalLoad(), 1.0));
}

//=================== validation: memory boundary (B) =======================
TEST_CASE("Boundary B: single-installment infeasible when sum(B) < V, multi feasible") {
    // 2 processors, B=400 each => single-installment capacity 800 < V=1000.
    DLSInstance inst({{0.0, 0.1, 0.2, 400.0}, {0.0, 0.1, 0.2, 400.0}}, 1000.0);

    ExactParams single; single.maxInstallments = 2; single.allowRepeats = false;
    CHECK(ExactSolver(single).solve(inst, SolverConfig{}).status == SolveStatus::Infeasible);

    ExactParams multi; multi.maxInstallments = 5; multi.allowRepeats = true;
    DLSSolution s = ExactSolver(multi).solve(inst, SolverConfig{});
    CHECK(s.feasible());
    CHECK(s.conservesLoad(inst.totalLoad(), 1.0));
    CHECK(fragmentsWithinMemory(s, inst));            // every installment <= B
}

TEST_CASE("Boundary B: exactly at threshold (sum(B) == V) is feasible and tight") {
    // 2 processors, B=500 each => single-installment capacity exactly V=1000.
    DLSInstance inst({{0.0, 0.1, 0.2, 500.0}, {0.0, 0.1, 0.2, 500.0}}, 1000.0);
    ExactParams single; single.maxInstallments = 2; single.allowRepeats = false;
    DLSSolution s = ExactSolver(single).solve(inst, SolverConfig{});
    CHECK(s.feasible());
    CHECK(s.conservesLoad(inst.totalLoad(), 1.0));
    CHECK(fragmentsWithinMemory(s, inst));            // each load forced to ~500
}

//=================== validation: startup-cost boundary (S) =================
TEST_CASE("Boundary S: large startups make the B&B prune (fewer nodes than S=0)") {
    // Tiny comm/compute, huge startup: extending a prefix is pruned immediately.
    DLSInstance s0({{0.0,0.01,0.01,1e9},{0.0,0.01,0.01,1e9},{0.0,0.01,0.01,1e9}}, 100.0);
    DLSInstance sBig({{1000.0,0.01,0.01,1e9},{1000.0,0.01,0.01,1e9},{1000.0,0.01,0.01,1e9}}, 100.0);
    ExactParams ep; ep.maxInstallments = 4; ep.allowRepeats = true;

    DLSSolution a = ExactSolver(ep).solve(s0, SolverConfig{});
    DLSSolution b = ExactSolver(ep).solve(sBig, SolverConfig{});
    CHECK(a.status == SolveStatus::Optimal);
    CHECK(b.status == SolveStatus::Optimal);
    CHECK(b.iterations < a.iterations);               // pruning engaged with S>0
}

TEST_CASE("Warm-started B&B: correct optimum and effective pruning") {
    // Heterogeneous N=5, S>0: the incumbent is seeded from fast heuristics, so
    // far fewer nodes are explored. (Without the warm start this took 368 nodes;
    // here it is well under that — and the optimum is unchanged.)
    DLSInstance inst({{1.0,0.2,0.3,1e9},{0.8,0.15,0.25,1e9},{1.2,0.25,0.4,1e9},
                      {0.5,0.18,0.22,1e9},{0.9,0.13,0.33,1e9}}, 500.0);
    ExactParams ep; ep.maxInstallments = 5; ep.allowRepeats = true;
    DLSSolution s = ExactSolver(ep).solve(inst, SolverConfig{});
    CHECK(s.status == SolveStatus::Optimal);
    CHECK(s.conservesLoad(500.0, 1.0));
    CHECK(s.makespan == doctest::Approx(88.843).epsilon(1e-3));   // optimum unchanged by warm start
    CHECK(s.iterations < 250);                                    // pruning improved (was 368)

    // Warm start must not change the proven optimum vs brute force.
    DLSInstance small({{0.5,0.2,0.3,1e9},{0.7,0.15,0.25,1e9},{0.3,0.25,0.4,1e9}}, 1000.0);
    ExactParams e2; e2.maxInstallments = 3; e2.allowRepeats = false;
    CHECK(ExactSolver(e2).solve(small, SolverConfig{}).status == SolveStatus::Optimal);
}

TEST_CASE("Warm start is a switch: same optimum, the original B&B explores more") {
    DLSInstance inst({{1.0,0.2,0.3,1e9},{0.8,0.15,0.25,1e9},{1.2,0.25,0.4,1e9},
                      {0.5,0.18,0.22,1e9},{0.9,0.13,0.33,1e9}}, 500.0);
    ExactParams warm; warm.maxInstallments = 5; warm.allowRepeats = true; warm.warmStart = true;
    ExactParams orig; orig.maxInstallments = 5; orig.allowRepeats = true; orig.warmStart = false;
    DLSSolution w = ExactSolver(warm).solve(inst, SolverConfig{});
    DLSSolution o = ExactSolver(orig).solve(inst, SolverConfig{});
    REQUIRE(w.status == SolveStatus::Optimal);
    REQUIRE(o.status == SolveStatus::Optimal);
    CHECK(w.makespan == doctest::Approx(o.makespan));     // identical optimum either way
    CHECK(w.iterations <= o.iterations);                  // warm start never explores more
    CHECK(w.iterations < o.iterations);                   // and here it explores strictly fewer
}

//=================== validation: size scaling + cross-checks ===============
TEST_CASE("Scaling: across sizes, exact bounds the GA and load is conserved") {
    std::vector<DLSInstance> insts = {
        DLSInstance({{0.1,0.20,0.30,1e9},{0.2,0.15,0.25,1e9}}, 5000.0),                 // N=2
        DLSInstance({{0.1,0.20,0.30,1e9},{0.2,0.15,0.25,1e9},{0.3,0.25,0.40,1e9}}, 8000.0), // N=3
        DLSInstance({{0.1,0.20,0.30,1e9},{0.2,0.15,0.25,1e9},
                     {0.3,0.25,0.40,1e9},{0.1,0.18,0.22,1e9}}, 12000.0),                // N=4
    };
    const int L = 4;
    for (DLSInstance& inst : insts) {
        ExactParams ep; ep.maxInstallments = L; ep.allowRepeats = true;
        double exactCmax = ExactSolver(ep).solve(inst, SolverConfig{}).makespan;

        GAParams gp; gp.installments = L; gp.populationSize = 12;
        gp.maxGenerations = 120; gp.noImprovementLimit = 50;
        SolverConfig c; c.seed = 11;
        DLSSolution ga = GASolver(gp).solve(inst, c);

        CHECK(ga.feasible());
        CHECK(ga.conservesLoad(inst.totalLoad(), 1.0));
        CHECK(exactCmax <= ga.makespan + 1e-6);       // exact is a lower bound
    }
}

//=================== MLSD (multiple loads) =================================
TEST_CASE("MLSD evaluator reproduces the thesis Example 1 reference makespans") {
    MlsdInstance inst({{32.0,0.0},{2.0,0.0}}, {identicalProc(), identicalProc(), identicalProc()});
    MlsdScheduleEvaluator ev;
    MlsdStructure full; full.taskOrder={0,1}; full.procSeq={{0,1,2},{0,1,2}};
    CHECK(ev.evaluate(inst, full).makespan == doctest::Approx(40.0));
    MlsdStructure red;  red.taskOrder={0,1};  red.procSeq={{0,1,2},{0,1}};
    CHECK(ev.evaluate(inst, red).makespan == doctest::Approx(39.3333).epsilon(1e-4));
}

TEST_CASE("MLSD exact solver: per-task conservation and Example 1 optimum") {
    MlsdInstance inst({{32.0,0.0},{2.0,0.0}}, {identicalProc(), identicalProc(), identicalProc()});
    MlsdSolver solver;
    MlsdSolution s = solver.solveFixedOrder(inst, {0,1});   // thesis fixed-order optimum
    CHECK(s.status == SolveStatus::Optimal);
    CHECK(s.makespan == doctest::Approx(39.3333).epsilon(1e-4));
    // per-task load conservation: sum_k loads[l][k] == V of that task
    for (std::size_t l = 0; l < s.loads.size(); ++l) {
        double sum = 0.0; for (double a : s.loads[l]) sum += a;
        CHECK(sum == doctest::Approx(inst.tasks()[s.taskOrder[l]].size).epsilon(1e-4));
    }
    CHECK(solver.solve(inst).makespan <= 39.3333 + 1e-3);   // global optimum no worse
}

TEST_CASE("MLSD exact searches task orders (global optimum <= any fixed order)") {
    // The exact solver searches all task orders; the global optimum can only be
    // <= the optimum for any fixed order. (Here the larger task first is better.)
    MlsdInstance inst({{2.0,0.0},{32.0,0.0}}, {identicalProc(), identicalProc(), identicalProc()});
    MlsdSolver solver;
    double global = solver.solve(inst).makespan;
    CHECK(global <= solver.solveFixedOrder(inst, {0,1}).makespan + 1e-6);
    CHECK(global <= solver.solveFixedOrder(inst, {1,0}).makespan + 1e-6);
    CHECK(global == doctest::Approx(39.3333).epsilon(1e-4));   // large-task-first optimum
}

TEST_CASE("MLSD result return (model A): single task/proc matches the closed form") {
    Processor p; p.commStartup=0; p.commRate=1; p.computeRate=1; p.memoryLimit=1e18;
    MlsdScheduleEvaluator ev;
    MlsdStructure st; st.taskOrder={0}; st.procSeq={{0}};

    MlsdInstance noRet({{10.0, 0.0}}, {p});            // beta=0
    CHECK(ev.evaluate(noRet, st).makespan == doctest::Approx(20.0));

    MlsdInstance ret({{10.0, 0.5}}, {p});              // beta=0.5
    CHECK(ev.evaluate(ret, st).makespan == doctest::Approx(25.0));   // send 10 + compute 10 + return 5
}

TEST_CASE("MLSD result return: larger beta never lowers the makespan") {
    MlsdInstance base({{20.0,0.0},{8.0,0.0}}, {identicalProc(), identicalProc()});
    MlsdSolver solver;
    double m0 = solver.solve(base).makespan;
    MlsdInstance withRet({{20.0,1.0},{8.0,1.0}}, {identicalProc(), identicalProc()});
    double m1 = solver.solve(withRet).makespan;
    CHECK(m1 >= m0 - 1e-6);
}

TEST_CASE("MLSD GA: reproducible and reaches the exact optimum on a small instance") {
    MlsdInstance inst({{32.0,0.0},{2.0,0.0}}, {identicalProc(), identicalProc(), identicalProc()});
    double exactOpt = MlsdSolver().solve(inst).makespan;   // 39.333

    MlsdGaSolver::Params p; p.populationSize=24; p.maxGenerations=120;
    MlsdGaSolver ga(p);
    MlsdSolution a = ga.solve(inst, 1);
    MlsdSolution b = ga.solve(inst, 1);
    CHECK(a.status == SolveStatus::Feasible);
    CHECK(a.makespan == b.makespan);                       // reproducible (same seed)
    CHECK(a.makespan >= exactOpt - 1e-6);                  // exact is a lower bound
    CHECK(a.makespan <= exactOpt + 1e-3);                  // GA reaches the optimum here
}

#ifdef DLS_WITH_HIGHS
TEST_CASE("MlsdMilpSolver matches the brute-force enumerator (independent exact)") {
    // Thesis Example 1: m=3 identical, V=(32,2) => optimum 39.333.
    MlsdInstance ex1({{32.0,0.0},{2.0,0.0}}, {identicalProc(), identicalProc(), identicalProc()});
    MlsdSolution milp = MlsdMilpSolver().solve(ex1);
    REQUIRE(milp.status == SolveStatus::Optimal);
    CHECK(milp.makespan == doctest::Approx(39.3333).epsilon(1e-3));
    CHECK(milp.makespan == doctest::Approx(MlsdSolver().solve(ex1).makespan).epsilon(1e-3));

    // Heterogeneous instances: MILP optimum == enumerator optimum.
    Processor p1; p1.commStartup=0.1; p1.commRate=0.2; p1.computeRate=0.3; p1.memoryLimit=1e18;
    Processor p2; p2.commStartup=0.2; p2.commRate=0.15; p2.computeRate=0.25; p2.memoryLimit=1e18;
    Processor p3; p3.commStartup=0.0; p3.commRate=0.25; p3.computeRate=0.4; p3.memoryLimit=1e18;
    std::vector<MlsdInstance> insts = {
        MlsdInstance({{20.0,0.0},{8.0,0.0}}, {p1, p2, p3}),
        MlsdInstance({{15.0,0.0},{15.0,0.0},{5.0,0.0}}, {p1, p2}),
    };
    for (const MlsdInstance& inst : insts) {
        MlsdSolution mm = MlsdMilpSolver().solve(inst);
        MlsdSolution bf = MlsdSolver().solve(inst);
        REQUIRE(mm.status == SolveStatus::Optimal);
        CHECK(mm.makespan == doctest::Approx(bf.makespan).epsilon(1e-3));   // independent exact methods agree
        // per-task conservation
        for (std::size_t l = 0; l < mm.loads.size(); ++l) {
            double sum = 0.0; for (double v : mm.loads[l]) sum += v;
            CHECK(sum == doctest::Approx(inst.tasks()[mm.taskOrder[l]].size).epsilon(1e-3));
        }
    }
}
#endif

//=================== OptV: maximize load within a deadline (dual) ==========

TEST_CASE("OptV evaluator: single processor max load = T/(C+A)") {
    // 1 processor S=0, C=A=1, deadline T=20 => Cmax=2α<=20 => max load 10.
    DLSInstance inst({{0.0,1.0,1.0,1e9}}, /*V ignored=*/1.0);
    SimplexScheduleEvaluator ev;
    EvaluatorConfig c; c.objective = EvalObjective::MaxLoad; c.makespanLimit = 20.0;
    DLSSolution s = ev.evaluate(inst, {0}, c);
    REQUIRE(s.status == SolveStatus::Optimal);
    CHECK(s.totalAssignedLoad() == doctest::Approx(10.0));
    CHECK(s.makespan <= 20.0 + 1e-6);
}

TEST_CASE("OptV evaluator: MaxLoad without a deadline is rejected") {
    DLSInstance inst({{0.0,1.0,1.0,1e9}}, 1.0);
    SimplexScheduleEvaluator ev;
    EvaluatorConfig c; c.objective = EvalObjective::MaxLoad;   // no makespanLimit
    CHECK(ev.evaluate(inst, {0}, c).status == SolveStatus::Failure);
}

TEST_CASE("OptVSolver: round-trip duality with OptT (max load in T* recovers V)") {
    // OptT min makespan for V, then OptV with deadline = that makespan over the
    // SAME sequence space must recover exactly V (load monotonicity).
    DLSInstance inst({{0.5,0.20,0.30,1e9},{0.7,0.15,0.25,1e9},{0.3,0.25,0.40,1e9}}, 10000.0);
    ExactParams ep; ep.maxInstallments = 3; ep.allowRepeats = false;
    DLSSolution optT = ExactSolver(ep).solve(inst, SolverConfig{});
    REQUIRE(optT.status == SolveStatus::Optimal);

    OptVParams vp; vp.deadline = optT.makespan; vp.maxInstallments = 3; vp.allowRepeats = false;
    DLSSolution optV = OptVSolver(vp).solve(inst, SolverConfig{});
    REQUIRE(optV.status == SolveStatus::Optimal);
    CHECK(optV.totalAssignedLoad() == doctest::Approx(10000.0).epsilon(1e-4));   // recovers V
    CHECK(optV.makespan <= optT.makespan + 1e-3);
}

TEST_CASE("OptVSolver: more time processes at least as much load") {
    DLSInstance inst({{0.1,0.2,0.3,1e9},{0.2,0.15,0.25,1e9}}, 1.0);
    auto loadFor = [&](double T) {
        OptVParams vp; vp.deadline = T; vp.maxInstallments = 2; vp.allowRepeats = false;
        return OptVSolver(vp).solve(inst, SolverConfig{}).totalAssignedLoad();
    };
    CHECK(loadFor(50.0)  >= loadFor(20.0)  - 1e-6);
    CHECK(loadFor(100.0) >= loadFor(50.0)  - 1e-6);
}

//=================== FPTAS for OptV on DLS{Cᵢ=0} (guarantee) ===============

// Exact max load for DLS{Cᵢ=0}-OptV by brute force over subsets, using the
// Sᵢ·Aᵢ ordering and the closed-form V (eq. 2.1).
static double bruteOptVCi0(const std::vector<std::pair<double,double>>& sa /*{S,A}*/, double T) {
    std::vector<int> ord(sa.size());
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(),
                     [&](int a, int b) { return sa[a].first * sa[a].second < sa[b].first * sa[b].second; });
    const int m = (int)sa.size();
    double best = 0.0;
    for (int mask = 0; mask < (1 << m); ++mask) {
        double V = 0.0;
        for (int i = 0; i < m; ++i) if (mask & (1 << i)) V += T / sa[ord[i]].second;
        for (int i = 0; i < m; ++i) if (mask & (1 << i))
            for (int j = i; j < m; ++j) if (mask & (1 << j))
                V -= sa[ord[i]].first / sa[ord[j]].second;
        best = std::max(best, V);
    }
    return best;
}

static DLSInstance ci0Instance(const std::vector<std::pair<double,double>>& sa) {
    std::vector<Processor> ps;
    for (auto& p : sa) { Processor q; q.commStartup=p.first; q.commRate=0.0; q.computeRate=p.second; q.memoryLimit=1e18; ps.push_back(q); }
    return DLSInstance(ps, 1.0);   // totalLoad ignored by OptV
}

TEST_CASE("FPTAS OptV: single processor gives (T-S)/A exactly") {
    DLSInstance inst = ci0Instance({{2.0, 1.0}});       // S=2, A=1
    inst.setTotalLoad(1000.0);                          // V-cap must not fire here
    FptasOptVParams fp; fp.deadline = 12.0; fp.epsilon = 0.1;
    DLSSolution s = FptasOptVSolver(fp).solve(inst, SolverConfig{});
    REQUIRE(s.status == SolveStatus::Feasible);
    CHECK(s.totalAssignedLoad() == doctest::Approx(10.0));   // (12-2)/1
}

TEST_CASE("FPTAS OptV: within the (1-ε) guarantee of the brute-force optimum") {
    std::vector<std::pair<double,double>> sa = {{1.0,2.0},{3.0,1.0},{2.0,4.0},{5.0,1.0},{1.0,1.0}};
    const double T = 20.0;
    const double opt = bruteOptVCi0(sa, T);
    DLSInstance inst = ci0Instance(sa);
    inst.setTotalLoad(1000.0);                          // V-cap must not fire here
    for (double eps : {0.5, 0.1, 0.01}) {
        FptasOptVParams fp; fp.deadline = T; fp.epsilon = eps;
        double v = FptasOptVSolver(fp).solve(inst, SolverConfig{}).totalAssignedLoad();
        CHECK(v >= (1.0 - eps) * opt - 1e-6);     // provable guarantee
        CHECK(v <= opt + 1e-6);                   // never exceeds the true optimum
    }
}

TEST_CASE("FPTAS OptV: tighter ε is at least as good and reaches the optimum") {
    std::vector<std::pair<double,double>> sa = {{2.0,3.0},{4.0,1.0},{1.0,5.0},{3.0,2.0}};
    const double T = 30.0, opt = bruteOptVCi0(sa, T);
    DLSInstance inst = ci0Instance(sa);
    inst.setTotalLoad(1000.0);                          // V-cap must not fire here
    auto vFor = [&](double eps) { FptasOptVParams fp; fp.deadline=T; fp.epsilon=eps;
                                  return FptasOptVSolver(fp).solve(inst, SolverConfig{}).totalAssignedLoad(); };
    CHECK(vFor(0.01) >= vFor(0.5) - 1e-6);
    CHECK(vFor(0.01) == doctest::Approx(opt).epsilon(1e-2));   // essentially optimal
}

TEST_CASE("FPTAS OptV agrees (within ε) with the general OptVSolver on Cᵢ=0") {
    std::vector<std::pair<double,double>> sa = {{1.0,2.0},{3.0,1.0},{2.0,3.0}};
    const double T = 15.0;
    DLSInstance inst = ci0Instance(sa);
    inst.setTotalLoad(1000.0);                          // V-cap must not fire here
    OptVParams vp; vp.deadline = T; vp.maxInstallments = 3; vp.allowRepeats = false;
    const double exactV = OptVSolver(vp).solve(inst, SolverConfig{}).totalAssignedLoad();

    FptasOptVParams fp; fp.deadline = T; fp.epsilon = 0.05;
    double v = FptasOptVSolver(fp).solve(inst, SolverConfig{}).totalAssignedLoad();
    CHECK(exactV == doctest::Approx(bruteOptVCi0(sa, T)).epsilon(1e-4));   // LP == closed form
    CHECK(v >= (1.0 - 0.05) * exactV - 1e-6);
    CHECK(v <= exactV + 1e-6);
}

TEST_CASE("FPTAS OptV: rejects a non-Cᵢ=0 instance and a missing deadline") {
    DLSInstance withComm({{1.0, 0.5, 2.0, 1e18}}, 1.0);   // commRate != 0
    FptasOptVParams fp; fp.deadline = 10.0;
    CHECK(FptasOptVSolver(fp).solve(withComm, SolverConfig{}).status == SolveStatus::Failure);
    FptasOptVParams noT;   // deadline = inf
    CHECK(FptasOptVSolver(noT).solve(ci0Instance({{1.0,1.0}}), SolverConfig{}).status == SolveStatus::Failure);
}

//=================== FPTAS for OptT on DLS{Cᵢ=0} (guarantee) ===============

TEST_CASE("FPTAS OptT: single processor gives S + V·A exactly") {
    DLSInstance inst = ci0Instance({{2.0, 1.0}});       // S=2, A=1
    inst.setTotalLoad(10.0);                            // V=10
    DLSSolution s = FptasOptTSolver({0.1}).solve(inst, SolverConfig{});
    REQUIRE(s.status == SolveStatus::Feasible);
    CHECK(s.makespan == doctest::Approx(12.0));         // S + V·A
    CHECK(s.conservesLoad(10.0, 1e-4));
}

// Exact min time for DLS{Cᵢ=0}-OptT by brute force: over non-empty subsets, the
// equal-completion time T' = (V + Σ cumᵢ/Aᵢ)/Σ(1/Aᵢ) with Sᵢ·Aᵢ ordering.
static double bruteOptTCi0(const std::vector<std::pair<double,double>>& sa, double V) {
    std::vector<int> ord(sa.size());
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(),
                     [&](int a, int b) { return sa[a].first*sa[a].second < sa[b].first*sa[b].second; });
    const int m = (int)sa.size();
    double best = std::numeric_limits<double>::infinity();
    for (int mask = 1; mask < (1 << m); ++mask) {
        double cum = 0.0, sInvA = 0.0, sCumA = 0.0;
        for (int i = 0; i < m; ++i) if (mask & (1 << i)) {
            cum += sa[ord[i]].first; sInvA += 1.0/sa[ord[i]].second; sCumA += cum/sa[ord[i]].second;
        }
        const double Tp = (V + sCumA) / sInvA;
        // valid only if all loads (T'-cum)/A >= 0 for the used processors
        bool ok = true; double c2 = 0.0;
        for (int i = 0; i < m && ok; ++i) if (mask & (1 << i)) { c2 += sa[ord[i]].first; if (Tp < c2 - 1e-9) ok = false; }
        if (ok) best = std::min(best, Tp);
    }
    return best;
}

TEST_CASE("FPTAS OptT: within (1+ε) of the brute-force optimum, load conserved") {
    std::vector<std::pair<double,double>> sa = {{1.0,2.0},{3.0,1.0},{2.0,4.0},{5.0,1.0},{1.0,1.0}};
    DLSInstance inst = ci0Instance(sa);
    inst.setTotalLoad(8.0);
    const double tOpt = bruteOptTCi0(sa, 8.0);                  // true T_OPT (Cᵢ=0)
    REQUIRE(tOpt > 0.0);
    for (double eps : {0.2, 0.05}) {
        DLSSolution s = FptasOptTSolver({eps}).solve(inst, SolverConfig{});
        REQUIRE(s.status == SolveStatus::Feasible);
        CHECK(s.conservesLoad(8.0, 1e-3));                       // processes exactly V
        CHECK(s.makespan >= tOpt - 1e-4);                        // can't beat the optimum
        CHECK(s.makespan <= (1.0 + eps) * tOpt + 1e-4);          // provable guarantee
    }
}

TEST_CASE("ExactSolver finds the true optimum on a Cᵢ=0 instance (ideal-bound fix)") {
    // Regression: the ideal-processor lower bound must not inject ideal startups
    // into the makespan (which previously pruned the optimum here).
    std::vector<std::pair<double,double>> sa = {{1.0,2.0},{3.0,1.0},{2.0,4.0},{5.0,1.0},{1.0,1.0}};
    DLSInstance inst = ci0Instance(sa);
    inst.setTotalLoad(8.0);
    ExactParams ep; ep.maxInstallments = 5; ep.allowRepeats = false;
    DLSSolution s = ExactSolver(ep).solve(inst, SolverConfig{});
    CHECK(s.status == SolveStatus::Optimal);
    CHECK(s.makespan == doctest::Approx(bruteOptTCi0(sa, 8.0)));  // == 6.0, not 6.286
}

TEST_CASE("FPTAS OptT: the returned time suffices to process V (round-trip with OptV)") {
    DLSInstance inst = ci0Instance({{1.0,2.0},{3.0,1.0},{2.0,3.0}});
    inst.setTotalLoad(6.0);
    DLSSolution s = FptasOptTSolver({0.1}).solve(inst, SolverConfig{});
    REQUIRE(s.status == SolveStatus::Feasible);
    OptVParams vp; vp.deadline = s.makespan; vp.maxInstallments = 3; vp.allowRepeats = false;
    double maxV = OptVSolver(vp).solve(inst, SolverConfig{}).totalAssignedLoad();
    CHECK(maxV >= 6.0 - 1e-4);                                   // deadline is feasible for V
}

TEST_CASE("FPTAS OptT: tighter ε is at least as good, and rejects non-Cᵢ=0") {
    DLSInstance inst = ci0Instance({{2.0,3.0},{4.0,1.0},{1.0,5.0},{3.0,2.0}});
    inst.setTotalLoad(12.0);
    auto tFor = [&](double eps) { return FptasOptTSolver({eps}).solve(inst, SolverConfig{}).makespan; };
    CHECK(tFor(0.02) <= tFor(0.4) + 1e-6);

    DLSInstance withComm({{1.0, 0.5, 2.0, 1e18}}, 5.0);          // commRate != 0
    CHECK(FptasOptTSolver({0.1}).solve(withComm, SolverConfig{}).status == SolveStatus::Failure);
}

//=================== Dual bisection exact (OptV-based, cross-check) =========

TEST_CASE("DualBisectionSolver matches the primal B&B (same optimum, different method)") {
    std::vector<DLSInstance> insts = {
        DLSInstance({{0.5,0.20,0.30,1e9},{0.7,0.15,0.25,1e9},{0.3,0.25,0.40,1e9}}, 10000.0),
        DLSInstance({{0.0,0.23,0.12,1e9},{0.1,0.11,0.23,1e9},{0.2,0.11,0.31,1e9}}, 5000.0),
        DLSInstance({{1.0,0.2,0.3,1e9},{0.8,0.15,0.25,1e9},{1.2,0.25,0.4,1e9}}, 500.0),
    };
    for (DLSInstance& inst : insts) {
        DualBisectionParams dp; dp.maxInstallments = 3; dp.allowRepeats = false;
        DLSSolution dual = DualBisectionSolver(dp).solve(inst, SolverConfig{});
        ExactParams ep; ep.maxInstallments = 3; ep.allowRepeats = false;
        DLSSolution bnb = ExactSolver(ep).solve(inst, SolverConfig{});
        REQUIRE(dual.status == SolveStatus::Optimal);
        CHECK(dual.conservesLoad(inst.totalLoad(), 1.0));
        CHECK(dual.makespan == doctest::Approx(bnb.makespan).epsilon(1e-3));   // dual == primal optimum
    }
}

TEST_CASE("DualBisectionSolver: single-processor closed form and registry access") {
    DLSInstance one({{0.0, 0.11, 0.12, 1e9}}, 1000.0);                 // (C+A)·V = 230
    CHECK(DualBisectionSolver().solve(one, SolverConfig{}).makespan == doctest::Approx(230.0));
    DLSInstance two({{0.0,0.20,0.30,1e9},{0.1,0.15,0.25,1e9}}, 1000.0);
    DLSSolution s = makeSolver("exact-dual")->solve(two, SolverConfig{});
    CHECK(s.feasible());
    CHECK(s.conservesLoad(1000.0, 1.0));
}

//=================== MapReduce (Berlińska ch. 4) ===========================

TEST_CASE("MapReduce: single mapper closed form (both equivalent T forms)") {
    // 1 mapper A=2, V=10, S=1, C=0.5, γ₀=1, r=1, no reducer compute.
    MapReduceInstance inst({2.0}, 10.0);
    inst.setStartup(1.0); inst.setReadRate(0.5); inst.setResultFraction(1.0);
    MapReduceSolution s = MapReduceSolver().solve(inst);
    REQUIRE(s.status == SolveStatus::Optimal);
    // α₁ = V; T = S + V·(A + γ₀C) + t_red = 1 + 10·2.5 + 0 = 26.
    CHECK(s.mapperLoads.size() == 1);
    CHECK(s.mapperLoads[0] == doctest::Approx(10.0));
    CHECK(s.makespan == doctest::Approx(26.0));
}

TEST_CASE("MapReduce: reducer execution time τ(x)=a_red·x·log₂x") {
    // 1 mapper A=1, V=8, S=C=0, γ₀=1, r=1, s_red=2, a_red=1.
    // z = γ₀V/r = 8 => t_red = 2 + 1·8·log₂8 = 2 + 24 = 26; T = V·A + t_red = 34.
    MapReduceInstance inst({1.0}, 8.0);
    inst.setReducerStartup(2.0); inst.setReducerRate(1.0);
    MapReduceSolution s = MapReduceSolver().solve(inst);
    REQUIRE(s.status == SolveStatus::Optimal);
    CHECK(s.reducerTime == doctest::Approx(26.0));
    CHECK(s.makespan    == doctest::Approx(34.0));
}

TEST_CASE("MapReduce: partition is conserved and satisfies the thesis identity") {
    // The solved load partition must obey the dual schedule-length identity
    // mS + αₘ(Aₘ+g) ≡ S + α₁A₁ + gV  (g = γ₀C/r), and conserve the load.
    MapReduceInstance inst({1.0, 2.0, 3.0}, 100.0);
    inst.setStartup(1.0); inst.setReadRate(1.0); inst.setResultFraction(0.5);
    MapReduceSolution s = MapReduceSolver().solve(inst);
    REQUIRE(s.status == SolveStatus::Optimal);
    CHECK(s.totalAssignedLoad() == doctest::Approx(100.0));            // Σαᵢ = V

    const double g = 0.5 * 1.0 / 1.0;                                  // γ₀C/r
    const int m = (int)s.mapperLoads.size();
    const double Am = inst.mapperRates()[s.mapperOrder[m - 1]];
    const double A1 = inst.mapperRates()[s.mapperOrder[0]];
    const double form1 = m * inst.startup() + s.mapperLoads[m - 1] * (Am + g) + s.reducerTime;
    const double form2 = inst.startup() + s.mapperLoads[0] * A1 + g * 100.0 + s.reducerTime;
    CHECK(form1 == doctest::Approx(form2));                            // identity (4.49)
    CHECK(s.makespan == doctest::Approx(form1));
    // Faster mappers (smaller Aᵢ, activated first) receive at least as much load.
    CHECK(s.mapperLoads.front() >= s.mapperLoads.back() - 1e-9);
}

TEST_CASE("MapReduce: over-provisioning drops mappers (αₘ < 0)") {
    // Many processors for a small load with non-trivial startups: using all of
    // them would force a negative load on the slowest, so some are dropped.
    MapReduceInstance inst(std::vector<double>(8, 1.0), 2.0);   // 8 identical mappers
    inst.setStartup(1.0); inst.setReadRate(1.0); inst.setResultFraction(1.0);
    MapReduceSolution s = MapReduceSolver().solve(inst);
    REQUIRE(s.status == SolveStatus::Optimal);
    CHECK((int)s.mapperOrder.size() < 8);                            // not all mappers used
    CHECK(s.totalAssignedLoad() == doctest::Approx(2.0));
    for (double a : s.mapperLoads) CHECK(a >= -1e-9);                // all loads non-negative
}

TEST_CASE("MapReduce: more reducers shorten the read-bound makespan (a_red=0)") {
    // With no reducer compute (a_red=0), increasing r shrinks the γ₀C/r read
    // term, so the schedule gets no longer (monotone non-increasing in r).
    auto makespanFor = [](int r) {
        MapReduceInstance inst({1.0, 2.0}, 50.0);
        inst.setStartup(0.5); inst.setReadRate(1.0); inst.setResultFraction(1.0);
        inst.setNumReducers(r);
        return MapReduceSolver().solve(inst).makespan;
    };
    CHECK(makespanFor(2) <= makespanFor(1) + 1e-9);
    CHECK(makespanFor(5) <= makespanFor(2) + 1e-9);
}

//=================== Linear daisy chain topology ===========================

TEST_CASE("Linear chain: analytic anchors (1 proc, 2 procs, free link)") {
    // 1 processor: it computes everything => T = A0·V.
    LinearChainInstance one({{/*A=*/1.0, /*C=*/0.0}}, 30.0);
    ChainSolution s1 = LinearChainSolver().solve(one);
    REQUIRE(s1.status == SolveStatus::Optimal);
    CHECK(s1.makespan == doctest::Approx(30.0));

    // 2 identical procs, link rate C1=1: balance α0=2α1 => α0=20, α1=10, T=20.
    LinearChainInstance two({{1.0, 0.0}, {1.0, 1.0}}, 30.0);
    ChainSolution s2 = LinearChainSolver().solve(two);
    REQUIRE(s2.status == SolveStatus::Optimal);
    CHECK(s2.makespan == doctest::Approx(20.0));
    CHECK(s2.loads[0] == doctest::Approx(20.0));
    CHECK(s2.loads[1] == doctest::Approx(10.0));

    // Free link (C1=0): the chain behaves like perfect parallelism => T = V/2.
    LinearChainInstance freeLink({{1.0, 0.0}, {1.0, 0.0}}, 30.0);
    CHECK(LinearChainSolver().solve(freeLink).makespan == doctest::Approx(15.0));
}

TEST_CASE("Linear chain: optimum satisfies equal-finish + conservation") {
    LinearChainInstance inst({{0.3, 0.0}, {0.25, 0.2}, {0.4, 0.15}, {0.22, 0.3}}, 1000.0);
    ChainSolution s = LinearChainSolver().solve(inst);
    REQUIRE(s.status == SolveStatus::Optimal);
    CHECK(s.totalAssignedLoad() == doctest::Approx(1000.0));     // conservation

    // Every participating processor finishes at the makespan (optimality principle).
    for (int i = 0; i < (int)inst.numProcessors(); ++i) {
        double f = LinearChainSolver::finishTime(inst, s.loads, i);
        CHECK(f <= s.makespan + 1e-4);
        if (s.loads[i] > 1e-6) CHECK(f == doctest::Approx(s.makespan).epsilon(1e-4));
    }
}

TEST_CASE("Linear chain: a very slow link starves the downstream processor") {
    // P1's feeding link is so slow that forwarding load to it is not worth it.
    LinearChainInstance inst({{1.0, 0.0}, {1.0, 1000.0}}, 100.0);
    ChainSolution s = LinearChainSolver().solve(inst);
    REQUIRE(s.status == SolveStatus::Optimal);
    CHECK(s.loads[0] > 99.0);          // P0 keeps almost all the load
    CHECK(s.loads[1] < 1.0);           // only a sliver is worth forwarding over the slow link
    CHECK(s.makespan < 100.0);         // marginally below A0·V=100 (the tiny offload still helps)
    CHECK(s.makespan > 99.0);
}

//=================== Multi-level tree topology =============================

TEST_CASE("Tree: one node computes everything (T = A0·V)") {
    TreeInstance one({{2.0, 0.0, -1}}, 10.0);
    TreeSolution s = TreeSolver().solve(one);
    REQUIRE(s.status == SolveStatus::Optimal);
    CHECK(s.makespan == doctest::Approx(20.0));
}

TEST_CASE("Tree generalizes the chain: a path tree == LinearChainSolver") {
    // Identical params as a chain: node i feeds node i+1.
    LinearChainInstance chain({{1.0,0.0},{1.0,1.0},{0.5,0.5},{2.0,0.3}}, 100.0);
    TreeInstance path({{1.0,0.0,-1},{1.0,1.0,0},{0.5,0.5,1},{2.0,0.3,2}}, 100.0);
    double cm = LinearChainSolver().solve(chain).makespan;
    double tm = TreeSolver().solve(path).makespan;
    CHECK(tm == doctest::Approx(cm).epsilon(1e-4));
}

TEST_CASE("Tree generalizes the star: a 1-level tree == the single-port star LP") {
    // Root distributes to leaf children; make the root a pure distributor
    // (huge compute rate => keeps ~0) so it matches the S=0 star evaluator.
    TreeInstance star({{1e12, 0.0, -1}, {1.0,0.2,0}, {1.0,0.15,0}, {1.0,0.25,0}}, 1000.0);
    double tm = TreeSolver().solve(star).makespan;

    DLSInstance worker({{0.0,0.20,1.0,1e18},{0.0,0.15,1.0,1e18},{0.0,0.25,1.0,1e18}}, 1000.0);
    double sm = SimplexScheduleEvaluator().evaluate(worker, {0,1,2}, EvaluatorConfig{}).makespan;
    CHECK(tm == doctest::Approx(sm).epsilon(1e-3));   // tree(star) == star single-port LP
}

TEST_CASE("Tree: optimum satisfies equal-finish + conservation on a 2-level tree") {
    // Root 0 -> children 1, 2; child 1 -> grandchildren 3, 4.
    TreeInstance inst({{0.3,0.0,-1}, {0.25,0.2,0}, {0.4,0.15,0},
                       {0.5,0.3,1}, {0.35,0.25,1}}, 1000.0);
    TreeSolution s = TreeSolver().solve(inst);
    REQUIRE(s.status == SolveStatus::Optimal);
    CHECK(s.totalAssignedLoad() == doctest::Approx(1000.0));
    for (int i = 0; i < (int)inst.numNodes(); ++i) {
        double f = TreeSolver::finishTime(inst, s.loads, i);
        CHECK(f <= s.makespan + 1e-4);
        if (s.loads[i] > 1e-6) CHECK(f == doctest::Approx(s.makespan).epsilon(1e-4));
    }
}

//=================== General graph topology (mesh) =========================

TEST_CASE("Graph reduces to the tree when its edges already form a tree") {
    // Same structure/rates as a tree; the only spanning arborescence is the tree.
    TreeInstance tree({{0.3,0.0,-1},{0.25,0.2,0},{0.4,0.15,0},{0.5,0.3,1}}, 1000.0);
    double tm = TreeSolver().solve(tree).makespan;

    GraphInstance g({0.3, 0.25, 0.4, 0.5},
                    {{0,1,0.2},{0,2,0.15},{1,3,0.3}}, 1000.0);
    GraphSolution gs = GraphSolver().solve(g);
    REQUIRE(gs.status == SolveStatus::Optimal);
    CHECK(gs.makespan == doctest::Approx(tm).epsilon(1e-4));
    CHECK(gs.totalAssignedLoad() == doctest::Approx(1000.0));
}

TEST_CASE("Graph routes through the cheaper path (shortcut beats the slow direct link)") {
    // Node 2's direct link from the source is terrible (rate 10), but 0—1—2 is
    // fast (0.1 each). The optimal arborescence is the chain 0→1→2.
    GraphInstance g({1.0, 1.0, 1.0}, {{0,1,0.1},{1,2,0.1},{0,2,10.0}}, 30.0);
    GraphSolution gs = GraphSolver().solve(g);
    REQUIRE(gs.status == SolveStatus::Optimal);

    TreeInstance chain({{1.0,0.0,-1},{1.0,0.1,0},{1.0,0.1,1}}, 30.0);   // 0→1→2
    TreeInstance star ({{1.0,0.0,-1},{1.0,0.1,0},{1.0,10.0,0}}, 30.0);  // 0→1, 0→2 direct
    double mChain = TreeSolver().solve(chain).makespan;
    double mStar  = TreeSolver().solve(star).makespan;
    CHECK(gs.makespan == doctest::Approx(mChain).epsilon(1e-4));   // picked the chain
    CHECK(gs.makespan < mStar - 1e-6);                            // strictly better than the direct link
    CHECK(gs.parent[2] == 1);                                     // node 2 is fed via node 1
}

TEST_CASE("Graph: best arborescence is no worse than any specific one; load conserved") {
    GraphInstance g({0.5, 0.3, 0.4, 0.35},
                    {{0,1,0.2},{0,2,0.25},{1,2,0.1},{1,3,0.3},{2,3,0.15}}, 500.0);
    GraphSolution gs = GraphSolver().solve(g);
    REQUIRE(gs.status == SolveStatus::Optimal);
    CHECK(gs.totalAssignedLoad() == doctest::Approx(500.0));

    // A particular arborescence (a star from the source over its direct edges)
    // is one candidate, so the chosen optimum cannot be worse than it.
    TreeInstance someTree({{0.5,0.0,-1},{0.3,0.2,0},{0.4,0.25,0},{0.35,0.3,1}}, 500.0);
    CHECK(gs.makespan <= TreeSolver().solve(someTree).makespan + 1e-6);
}

//=================== Multilayer applications (Berlińska ch. 5) =============

static double maxPiece(const std::vector<ComputePiece>& ps, double x) {
    double t = 0.0; for (const ComputePiece& p : ps) t = std::max(t, p.intercept + p.slope * x); return t;
}

TEST_CASE("x·log₂x convex pieces: exact at 2^y, convex, and over-estimate between") {
    const double a = 1.0;
    std::vector<ComputePiece> ps = xLogXConvexPieces(a, 16.0);
    auto tau = [&](double x) { return a * x * std::log2(x); };
    for (int y = 0; y <= 4; ++y) {                         // exact at the breakpoints 2^y
        double x = std::ldexp(1.0, y);
        CHECK(maxPiece(ps, x) == doctest::Approx(tau(x)).epsilon(1e-9));
    }
    for (std::size_t i = 0; i + 1 < ps.size(); ++i)        // convex: slopes non-decreasing
        CHECK(ps[i].slope <= ps[i + 1].slope + 1e-12);
    for (double x : {3.0, 5.0, 9.0, 13.0})                 // over-estimate strictly between
        CHECK(maxPiece(ps, x) >= tau(x) - 1e-9);
}

TEST_CASE("Multilayer: single layer / single reducer closed form") {
    MultilayerInstance inst;
    inst.totalLoad = 8.0; inst.numMappers = 1; inst.mapperRate = 1.0;
    inst.startup = 0.0; inst.readRate = 0.0; inst.mapperFraction = 1.0; inst.bisectionWidth = 1;
    inst.layers = {{/*count=*/1, /*startup=*/0.0, /*rate=*/1.0, /*γ=*/1.0}};
    MultilayerSolution s = MultilayerSolver().solve(inst);
    REQUIRE(s.status == SolveStatus::Feasible);
    CHECK(s.mapperTime == doctest::Approx(8.0));           // m·S + A·V/m = 8
    CHECK(s.layerInput[0] == doctest::Approx(8.0));        // L₁ = γ₀·V
    CHECK(s.layerCompute[0] == doctest::Approx(24.0));     // τ = 1·8·log₂8 = 24
    CHECK(s.makespan == doctest::Approx(32.0));            // 8 + 0 + 24
}

TEST_CASE("Multilayer: γ-chain sizes the layers, more reducers shorten compute") {
    MultilayerInstance inst;
    inst.totalLoad = 1000.0; inst.numMappers = 4; inst.mapperRate = 0.01;
    inst.startup = 0.0; inst.readRate = 0.0; inst.mapperFraction = 0.5; inst.bisectionWidth = 8;
    inst.layers = {{2, 0.0, 1e-3, 0.5}, {2, 0.0, 1e-3, 1.0}};
    MultilayerSolution s = MultilayerSolver().solve(inst);
    REQUIRE(s.status == SolveStatus::Feasible);
    CHECK(s.layerInput[0] == doctest::Approx(500.0));      // γ₀·V = 0.5·1000
    CHECK(s.layerInput[1] == doctest::Approx(250.0));      // ·γ₁ = ·0.5

    // More reducers in layer 1 => smaller per-reducer x·log₂x compute.
    auto computeFor = [](int r1) {
        MultilayerInstance i; i.totalLoad=1000; i.numMappers=4; i.mapperRate=0.01;
        i.mapperFraction=1.0; i.bisectionWidth=16; i.layers={{r1,0.0,1e-3,1.0}};
        return MultilayerSolver().solve(i).layerCompute[0];
    };
    CHECK(computeFor(4) < computeFor(1));
    CHECK(computeFor(8) < computeFor(4));
}

TEST_CASE("Multilayer: more layers never shorten the schedule; wider bisection helps reads") {
    auto makespanLayers = [](int R) {
        MultilayerInstance i; i.totalLoad=1024; i.numMappers=4; i.mapperRate=0.01;
        i.readRate=0.001; i.mapperFraction=1.0; i.bisectionWidth=4;
        for (int p = 0; p < R; ++p) i.layers.push_back({2, 0.1, 1e-4, 1.0});
        return MultilayerSolver().solve(i).makespan;
    };
    CHECK(makespanLayers(2) >= makespanLayers(1) - 1e-9);
    CHECK(makespanLayers(3) >= makespanLayers(2) - 1e-9);

    auto makespanBisect = [](int l) {
        MultilayerInstance i; i.totalLoad=1024; i.numMappers=4; i.mapperRate=0.01;
        i.readRate=0.01; i.mapperFraction=1.0; i.bisectionWidth=l;
        i.layers={{8, 0.0, 1e-4, 1.0}};
        return MultilayerSolver().solve(i).makespan;
    };
    CHECK(makespanBisect(8) <= makespanBisect(2) + 1e-9);   // more channels => faster reads
}

//=================== file I/O (continued) ==================================
TEST_CASE("Processor reader rejects bad header counts and missing file") {
    ProgrammeParameters pp; pp.maxProcNum = 0; pp.numProcParams = 4;  // bad maxProcNum
    ProcessorsParameters proc;
    std::string err;
    CHECK_FALSE(proc.ReadProcessorsParam(&pp, "/tmp/whatever.txt", &err));

    pp.maxProcNum = 3;
    CHECK_FALSE(proc.ReadProcessorsParam(&pp, "/no/such/dls_proc.txt", &err)); // missing file
}

//=================== energy model (Marszałkowski 2020) =====================
// Validates the energy criterion: #1 piecewise running energy ε(α) and #2 the
// four-state (idle/startup/network/running) + originator total-energy accounting,
// plus that the LP MinCost objective truly minimizes the closed-form energy.

// A worker with the classic affine time and a single energy piece {l, k}.
static Processor energyProc(double S, double C, double A, double k,
                            double pIdle, double pStartup, double pNetwork) {
    Processor p;
    p.commStartup = S; p.commRate = C; p.computeRate = A; p.memoryLimit = 1e18;
    p.powerIdle = pIdle; p.powerStartup = pStartup; p.powerNetwork = pNetwork;
    p.energyPieces = {{0.0, k}};
    return p;
}

TEST_CASE("Energy: legacy instances carry no energy model and report energy 0") {
    DLSInstance inst = canonical5();
    CHECK_FALSE(inst.usesEnergyModel());
    DLSSolution s = SimplexScheduleEvaluator().evaluate(inst, {0,1,2,3,4}, EvaluatorConfig{});
    CHECK(s.status == SolveStatus::Optimal);
    CHECK(s.energy == doctest::Approx(0.0));            // no regression on the makespan path
    CHECK(scheduleEnergy(inst, s) == doctest::Approx(0.0));
}

TEST_CASE("Energy: piecewise ε(α) beats the proportional model out of core (thesis Tab 3.4)") {
    // In-core piece {0, k1}, steeper out-of-core piece {l2<0, k2}, meeting at ρ.
    const double k1 = 0.4, k2 = 22.0, rho = 7500.0;
    const double l2 = rho * (k1 - k2);                  // continuity: k1·ρ = k2·ρ + l2
    Processor p; p.commStartup = 0; p.commRate = 0.002; p.computeRate = 0.006;
    p.memoryLimit = 1e18; p.energyPieces = {{0.0, k1}, {l2, k2}};

    CHECK(runningEnergy(p, rho)      == doctest::Approx(k1 * rho));        // meet at ρ
    CHECK(runningEnergy(p, 0.5*rho)  == doctest::Approx(k1 * 0.5*rho));   // in core: cheap piece
    const double big = 2.0 * rho;
    CHECK(runningEnergy(p, big)      == doctest::Approx(k2*big + l2));    // out of core: steep piece
    // The proportional model l·α (here k1·α) badly underestimates out of core.
    CHECK(runningEnergy(p, big) > k1 * big * 5.0);
}

TEST_CASE("Energy: scheduleEnergy matches a hand-computed single-installment schedule") {
    // One worker, V=100, single installment. S=1,C=0.1,A=0.2; P^S=5,P^N=10,k=2,P^I=3;
    // master P^N0=8, P^I0=4. T = S + C·V + A·V = 1 + 10 + 20 = 31.
    //   startup  P^S·S          = 5
    //   network  P^N·C·V        = 100
    //   running  k·V            = 200
    //   worker idle (busy=T)    = 0
    //   master   P^N0·(S+C·V) + P^I0·(T−(S+C·V)) = 8·11 + 4·20 = 168
    //   total                   = 473
    DLSInstance inst({energyProc(1.0, 0.1, 0.2, 2.0, /*Pi*/3.0, /*Ps*/5.0, /*Pn*/10.0)}, 100.0);
    inst.setOriginatorPowerNetwork(8.0);
    inst.setOriginatorPowerIdle(4.0);
    CHECK(inst.usesEnergyModel());

    DLSSolution s = SimplexScheduleEvaluator().evaluate(inst, {0}, EvaluatorConfig{});
    CHECK(s.status == SolveStatus::Optimal);
    CHECK(s.makespan == doctest::Approx(31.0));
    CHECK(s.energy   == doctest::Approx(473.0));
    CHECK(scheduleEnergy(inst, s) == doctest::Approx(473.0));   // closed form agrees
}

TEST_CASE("Energy: MinCost minimizes the closed-form energy and beats MinMakespan's split") {
    // M0 computes slower but its energy/load is cheap; M1 is fast but energy-costly.
    // The time-optimal split favours M1; the energy-optimal split favours M0, so the
    // two criteria genuinely diverge and min-energy <= the makespan optimum's energy.
    DLSInstance inst({energyProc(1.0, 0.1, 0.2, 1.0, 2.0, 4.0, 6.0),    // cheap energy
                      energyProc(1.0, 0.1, 0.1, 5.0, 2.0, 4.0, 6.0)},   // expensive energy
                     100.0);
    inst.setOriginatorPowerNetwork(6.0);
    inst.setOriginatorPowerIdle(2.0);
    const std::vector<int> seq = {0, 1};

    EvaluatorConfig tcfg; tcfg.objective = EvalObjective::MinMakespan;
    EvaluatorConfig ecfg; ecfg.objective = EvalObjective::MinCost;

    for (const char* backend : {"simplex", "highs"}) {
        std::unique_ptr<ScheduleEvaluator> ev = makeScheduleEvaluator(backend);
        if (!ev) continue;                              // highs absent in the no-HiGHS build
        DLSSolution st = ev->evaluate(inst, seq, tcfg);
        DLSSolution se = ev->evaluate(inst, seq, ecfg);
        REQUIRE(st.status == SolveStatus::Optimal);
        REQUIRE(se.status == SolveStatus::Optimal);
        CHECK(se.conservesLoad(100.0, 1e-3));
        // The LP-minimized energy equals the independent closed-form value...
        CHECK(se.energy == doctest::Approx(scheduleEnergy(inst, se)));
        // ...and is no worse than the energy of the makespan-optimal schedule.
        CHECK(se.energy <= st.energy + 1e-6);
    }
}

TEST_CASE("Energy: time-energy trade-off front is monotone (tighter makespan costs energy)") {
    DLSInstance inst({energyProc(1.0, 0.1, 0.2, 1.0, 2.0, 4.0, 6.0),
                      energyProc(1.0, 0.1, 0.1, 5.0, 2.0, 4.0, 6.0)},
                     100.0);
    inst.setOriginatorPowerNetwork(6.0);
    inst.setOriginatorPowerIdle(2.0);
    const std::vector<int> seq = {0, 1};
    SimplexScheduleEvaluator ev;

    // Minimum achievable makespan on this sequence.
    EvaluatorConfig tcfg; tcfg.objective = EvalObjective::MinMakespan;
    const double Tmin = ev.evaluate(inst, seq, tcfg).makespan;

    // Unconstrained min energy (it runs at some makespan >= Tmin).
    EvaluatorConfig loose; loose.objective = EvalObjective::MinCost;
    const double Eloose = ev.evaluate(inst, seq, loose).energy;

    // Forcing the schedule down to the minimum makespan can only raise energy.
    EvaluatorConfig tight; tight.objective = EvalObjective::MinCost;
    tight.makespanLimit = Tmin + 1e-6;
    DLSSolution stight = ev.evaluate(inst, seq, tight);
    REQUIRE(stight.status == SolveStatus::Optimal);
    CHECK(stight.makespan <= Tmin + 1e-3);
    CHECK(stight.energy >= Eloose - 1e-6);                 // Pareto monotonicity
}

TEST_CASE("Energy: instance I/O round-trips power rates and energy pieces") {
    DLSInstance inst({energyProc(1.0, 0.1, 0.2, 2.0, 3.0, 5.0, 10.0)}, 100.0);
    inst.processors()[0].energyPieces = {{0.0, 0.4}, {-7500.0*21.6, 22.0}};   // in/out of core
    inst.setOriginatorPowerNetwork(8.0);
    inst.setOriginatorPowerIdle(4.0);

    std::ostringstream out;
    writeInstance(out, inst);

    DLSInstance back;
    std::string err;
    std::istringstream in(out.str());
    REQUIRE(readInstance(in, back, err));
    REQUIRE(back.usesEnergyModel());
    const Processor& p = back.processors()[0];
    CHECK(p.powerIdle    == doctest::Approx(3.0));
    CHECK(p.powerStartup == doctest::Approx(5.0));
    CHECK(p.powerNetwork == doctest::Approx(10.0));
    REQUIRE(p.energyPieces.size() == 2);
    CHECK(p.energyPieces[1].slope == doctest::Approx(22.0));
    CHECK(back.originatorPowerNetwork() == doctest::Approx(8.0));
    CHECK(back.originatorPowerIdle()    == doctest::Approx(4.0));
    // Energy of the same schedule is identical after a round-trip.
    DLSSolution a = SimplexScheduleEvaluator().evaluate(inst, {0}, EvaluatorConfig{});
    DLSSolution b = SimplexScheduleEvaluator().evaluate(back, {0}, EvaluatorConfig{});
    CHECK(a.energy == doctest::Approx(b.energy));
}

//=================== energy-objective MILP (#3, Marszałkowski) ==============
// Cross-validates the energy MILP (sequence + split chosen by the MIP) against
// brute force over all single-installment sequences scored with the per-sequence
// evaluator's MinCost (= energy) objective.

#ifdef DLS_WITH_HIGHS
// All non-empty ordered selections of distinct processors from {0..N-1}.
static std::vector<std::vector<int>> allDistinctSequences(int N) {
    std::vector<std::vector<int>> out;
    std::vector<int> cur;
    std::vector<bool> used(N, false);
    std::function<void()> dfs = [&]() {
        if (!cur.empty()) out.push_back(cur);
        for (int i = 0; i < N; ++i) if (!used[i]) {
            used[i] = true; cur.push_back(i); dfs(); cur.pop_back(); used[i] = false;
        }
    };
    dfs();
    return out;
}

TEST_CASE("Energy MILP: minimum energy equals brute force over sequences") {
    DLSInstance inst({energyProc(1.0, 0.10, 0.20, 1.0, 2.0, 4.0, 6.0),
                      energyProc(0.5, 0.12, 0.15, 3.0, 2.0, 4.0, 6.0),
                      energyProc(2.0, 0.08, 0.10, 5.0, 2.0, 4.0, 6.0)}, 100.0);
    inst.setOriginatorPowerNetwork(6.0);
    inst.setOriginatorPowerIdle(2.0);

    SimplexScheduleEvaluator ev;
    EvaluatorConfig ec; ec.objective = EvalObjective::MinCost;     // = minimize energy
    double best = std::numeric_limits<double>::infinity();
    for (const std::vector<int>& seq : allDistinctSequences(3)) {
        DLSSolution s = ev.evaluate(inst, seq, ec);
        if (s.status == SolveStatus::Optimal && s.conservesLoad(100.0, 1e-3))
            best = std::min(best, s.energy);
    }

    MilpParams mp; mp.maxInstallments = 3; mp.minimizeCost = true; // energy mode => minimize E
    DLSSolution m = MilpSolver(mp).solve(inst, SolverConfig{});
    REQUIRE(m.status == SolveStatus::Optimal);
    CHECK(m.conservesLoad(100.0, 1e-3));
    CHECK(m.energy == doctest::Approx(best).epsilon(1e-3));        // MIP == brute force
    CHECK(m.energy == doctest::Approx(scheduleEnergy(inst, m)));   // reported == closed form
}

TEST_CASE("Energy MILP: bi-criteria (min E s.t. Cmax <= C-bar) matches brute force") {
    DLSInstance inst({energyProc(1.0, 0.10, 0.20, 1.0, 2.0, 4.0, 6.0),
                      energyProc(0.5, 0.12, 0.15, 3.0, 2.0, 4.0, 6.0),
                      energyProc(2.0, 0.08, 0.10, 5.0, 2.0, 4.0, 6.0)}, 100.0);
    inst.setOriginatorPowerNetwork(6.0);
    inst.setOriginatorPowerIdle(2.0);

    // Minimum makespan (energy ignored) sets the tightest feasible time budget.
    MilpParams tp; tp.maxInstallments = 3;
    const double Tmin = MilpSolver(tp).solve(inst, SolverConfig{}).makespan;
    const double Cbar = Tmin * 1.25;                               // a binding-but-feasible budget

    SimplexScheduleEvaluator ev;
    EvaluatorConfig ec; ec.objective = EvalObjective::MinCost; ec.makespanLimit = Cbar;
    double best = std::numeric_limits<double>::infinity();
    for (const std::vector<int>& seq : allDistinctSequences(3)) {
        DLSSolution s = ev.evaluate(inst, seq, ec);
        if (s.status == SolveStatus::Optimal && s.conservesLoad(100.0, 1e-3))
            best = std::min(best, s.energy);
    }

    MilpParams mp; mp.maxInstallments = 3; mp.minimizeCost = true; mp.makespanLimit = Cbar;
    DLSSolution m = MilpSolver(mp).solve(inst, SolverConfig{});
    REQUIRE(m.status == SolveStatus::Optimal);
    CHECK(m.makespan <= Cbar + 1e-3);
    CHECK(m.energy == doctest::Approx(best).epsilon(1e-3));
}

TEST_CASE("Energy MILP: reverse bi-criteria (min Cmax s.t. E <= E-bar) respects the limit") {
    DLSInstance inst({energyProc(1.0, 0.10, 0.20, 1.0, 2.0, 4.0, 6.0),
                      energyProc(0.5, 0.12, 0.15, 3.0, 2.0, 4.0, 6.0)}, 100.0);
    inst.setOriginatorPowerNetwork(6.0);
    inst.setOriginatorPowerIdle(2.0);

    // Unconstrained minimum energy, then allow a little slack as the budget.
    MilpParams ep; ep.maxInstallments = 2; ep.minimizeCost = true;
    const double Emin = MilpSolver(ep).solve(inst, SolverConfig{}).energy;

    MilpParams mp; mp.maxInstallments = 2; mp.costLimit = Emin * 1.10;   // E <= 1.1·Emin
    DLSSolution m = MilpSolver(mp).solve(inst, SolverConfig{});          // minimize Cmax
    REQUIRE(m.status == SolveStatus::Optimal);
    CHECK(m.energy <= Emin * 1.10 + 1e-3);                               // honoured the energy cap
}

TEST_CASE("Energy MILP: optimizing energy with piecewise TIME is rejected") {
    Processor p = energyProc(1.0, 0.1, 0.2, 2.0, 3.0, 5.0, 10.0);
    p.computePieces = {{0.0, 0.2}, {-10.0, 0.7}};      // multi-piece TIME (unsupported here)
    DLSInstance inst({p}, 100.0);
    MilpParams mp; mp.minimizeCost = true;
    DLSSolution s = MilpSolver(mp).solve(inst, SolverConfig{});
    CHECK(s.status == SolveStatus::Failure);
}
#endif  // DLS_WITH_HIGHS

//=================== online PSR + GSS/SSC heuristics (#4) ===================
// Validates the LP-free online simulation heuristics (Marszałkowski 2020 §6.2):
// schedule consistency, the memory invariant, lower/upper bounds, the super
// meta-rule, and hand-computed single-processor schedules.

// Asserts a fragment list is a valid single-port schedule and conserves load.
static void checkOnlineSchedule(const DLSInstance& inst, const DLSSolution& s) {
    REQUIRE(s.status == SolveStatus::Feasible);
    CHECK(s.conservesLoad(inst.totalLoad(), 1e-4));
    double prevCommFinish = 0.0, maxFinish = 0.0;
    for (const LoadFragment& f : s.fragments) {
        CHECK(f.loadSize > 0.0);
        CHECK(f.commStart >= prevCommFinish - 1e-6);         // single port: transfers serialize
        CHECK(f.computeStart >= f.commFinish - 1e-6);        // compute after receive
        CHECK(f.computeFinish >= f.computeStart - 1e-6);
        prevCommFinish = f.commFinish;
        maxFinish = std::max(maxFinish, f.computeFinish);
    }
    CHECK(s.makespan == doctest::Approx(maxFinish));         // makespan = last compute finish
}

TEST_CASE("Online: produces a consistent, load-conserving single-port schedule") {
    DLSInstance inst({{0.1, 0.11, 0.52, 4000.0}, {0.2, 0.21, 0.22, 5000.0},
                      {0.3, 0.31, 0.32, 1500.0}}, 100000.0);
    for (ChunkRule rule : {ChunkRule::SSC, ChunkRule::GSS}) {
        OnlineParams p; p.chunk = rule; p.psr = Psr::All;
        DLSSolution s = OnlineSolver(p).solve(inst, SolverConfig{});
        checkOnlineSchedule(inst, s);
        CHECK(s.makespan >= divisibleLoadLowerBoundTight(inst) - 1e-6);   // valid lower bound
    }
}

TEST_CASE("Online: SSC never exceeds a worker's memory (stays in core)") {
    DLSInstance inst({{0.1, 0.11, 0.52, 4000.0}, {0.2, 0.21, 0.22, 5000.0},
                      {0.3, 0.31, 0.32, 1500.0}}, 100000.0);
    OnlineParams p; p.chunk = ChunkRule::SSC; p.psr = Psr::ComputeRate;
    DLSSolution s = OnlineSolver(p).solve(inst, SolverConfig{});
    checkOnlineSchedule(inst, s);
    for (const LoadFragment& f : s.fragments)
        CHECK(f.loadSize <= inst.processors()[f.processorId].memoryLimit + 1e-6);
}

TEST_CASE("Online: LB <= LP-reoptimum(sequence) <= heuristic makespan") {
    DLSInstance inst({{0.1, 0.11, 0.52, 60000.0}, {0.2, 0.21, 0.22, 60000.0},
                      {0.3, 0.31, 0.32, 60000.0}}, 100000.0);
    OnlineParams p; p.chunk = ChunkRule::GSS; p.psr = Psr::All;
    DLSSolution s = OnlineSolver(p).solve(inst, SolverConfig{});
    checkOnlineSchedule(inst, s);

    // Re-optimizing the heuristic's OWN activation order with the LP can only
    // help (it optimises the split): LB <= lp.makespan <= heuristic makespan.
    // This certifies the heuristic sits above the true optimum without having to
    // enumerate the unbounded number of installments GSS may use.
    DLSSolution lp = SimplexScheduleEvaluator().evaluate(inst, s.sequence, EvaluatorConfig{});
    REQUIRE(lp.status == SolveStatus::Optimal);
    CHECK(lp.makespan <= s.makespan + 1e-4);
    CHECK(lp.makespan >= divisibleLoadLowerBoundTight(inst) - 1e-6);
    CHECK(s.makespan  >= divisibleLoadLowerBoundTight(inst) - 1e-6);
}

TEST_CASE("Online: super (best of all PSRs) is no worse than any single rule") {
    DLSInstance inst({{0.05, 0.11, 0.52, 3000.0}, {0.20, 0.05, 0.22, 5000.0},
                      {0.30, 0.31, 0.08, 1500.0}}, 80000.0);
    OnlineParams sp; sp.psr = Psr::All; sp.chunk = ChunkRule::GSS;
    const double superMk = OnlineSolver(sp).solve(inst, SolverConfig{}).makespan;
    for (Psr r : {Psr::ComputeRate, Psr::CommRate, Psr::CommStartup, Psr::MemoryDesc, Psr::EnergySlope}) {
        OnlineParams p; p.psr = r; p.chunk = ChunkRule::GSS;
        DLSSolution s = OnlineSolver(p).solve(inst, SolverConfig{});
        checkOnlineSchedule(inst, s);
        CHECK(superMk <= s.makespan + 1e-6);
    }
}

TEST_CASE("Online: single-processor schedule matches the closed form") {
    // 1 worker, V=100, ample memory => one chunk. makespan = S + C·V + A·V.
    DLSInstance one({{1.0, 0.1, 0.2, 1e18}}, 100.0);
    for (ChunkRule rule : {ChunkRule::SSC, ChunkRule::GSS}) {
        OnlineParams p; p.chunk = rule; p.psr = Psr::ComputeRate;
        DLSSolution s = OnlineSolver(p).solve(one, SolverConfig{});
        checkOnlineSchedule(one, s);
        CHECK(s.fragments.size() == 1);
        CHECK(s.makespan == doctest::Approx(1.0 + 0.1 * 100 + 0.2 * 100));   // 31
    }
}

TEST_CASE("Online: memory-limited single processor serializes into hand-computed chunks") {
    // 1 worker, V=100, RAM=40, S=1,C=0.1,A=0.2 => SSC chunks 40,40,20, fully
    // serialized (one worker): finishes 13, 26, 33. makespan = 33.
    DLSInstance one({{1.0, 0.1, 0.2, 40.0}}, 100.0);
    OnlineParams p; p.chunk = ChunkRule::SSC; p.psr = Psr::ComputeRate;
    DLSSolution s = OnlineSolver(p).solve(one, SolverConfig{});
    checkOnlineSchedule(one, s);
    REQUIRE(s.fragments.size() == 3);
    CHECK(s.fragments[0].loadSize == doctest::Approx(40.0));
    CHECK(s.fragments[1].loadSize == doctest::Approx(40.0));
    CHECK(s.fragments[2].loadSize == doctest::Approx(20.0));
    CHECK(s.makespan == doctest::Approx(33.0));
}

//=================== isoefficiency / isoenergy maps (#5) ====================
// Validates the 2-D performance-map generator (Marszałkowski 2020 §4 & §6.3):
// grid shape, axis ticks, the analytic isoenergy E = k·V, monotone isoefficiency,
// a single-point closed form, and the CSV writer.

TEST_CASE("Map: grid shape and axis ticks") {
    using namespace dls::bench;
    MapConfig c;
    c.xAxis = MapAxis::Processors; c.xMin = 1; c.xMax = 7; c.xSteps = 4;
    c.yAxis = MapAxis::CommRate;   c.yMin = 0.1; c.yMax = 0.5; c.ySteps = 3;
    c.solver = "single-round";
    PerformanceMap m = computePerformanceMap(c);

    REQUIRE(m.ys.size() == 3);
    REQUIRE(m.xs.size() == 4);
    REQUIRE(m.grid.size() == 3);
    REQUIRE(m.grid[0].size() == 4);
    CHECK(m.xs.front() == doctest::Approx(1.0));
    CHECK(m.xs.back()  == doctest::Approx(7.0));
    CHECK(m.xs[1]      == doctest::Approx(3.0));          // 1 + (7-1)*1/3
    CHECK(m.ys.front() == doctest::Approx(0.1));
    CHECK(m.ys.back()  == doctest::Approx(0.5));
}

TEST_CASE("Map: isoenergy is the analytic k·V and is flat along compute rate") {
    using namespace dls::bench;
    MapConfig c;
    c.metric = MapMetric::Energy; c.solver = "single-round";
    c.baseLoad = 100.0; c.baseCommStartup = 0.0;
    c.xAxis = MapAxis::ComputeRate; c.xMin = 0.1; c.xMax = 0.4; c.xSteps = 4;
    c.yAxis = MapAxis::EnergySlope; c.yMin = 1.0; c.yMax = 4.0; c.ySteps = 4;
    PerformanceMap m = computePerformanceMap(c);

    for (std::size_t yi = 0; yi < m.ys.size(); ++yi) {
        const double k = m.ys[yi];
        for (std::size_t xi = 0; xi < m.xs.size(); ++xi)
            // running energy ε = Σ k·αᵢ = k·V, independent of the compute rate (x).
            CHECK(m.grid[yi][xi] == doctest::Approx(k * c.baseLoad));
    }
}

TEST_CASE("Map: isoefficiency is monotone (more processors help, more startup hurts)") {
    using namespace dls::bench;
    MapConfig c;
    c.solver = "single-round"; c.baseLoad = 100.0; c.baseCommRate = 0.1; c.baseComputeRate = 0.2;
    c.xAxis = MapAxis::Processors;  c.xMin = 1; c.xMax = 5; c.xSteps = 5;
    c.yAxis = MapAxis::CommStartup; c.yMin = 0.0; c.yMax = 1.0; c.ySteps = 3;
    PerformanceMap m = computePerformanceMap(c);

    for (std::size_t yi = 0; yi < m.ys.size(); ++yi)
        for (std::size_t xi = 1; xi < m.xs.size(); ++xi)
            CHECK(m.grid[yi][xi] <= m.grid[yi][xi - 1] + 1e-6);   // makespan ↓ with N
    for (std::size_t xi = 0; xi < m.xs.size(); ++xi)
        for (std::size_t yi = 1; yi < m.ys.size(); ++yi)
            CHECK(m.grid[yi][xi] >= m.grid[yi - 1][xi] - 1e-6);   // makespan ↑ with startup
}

TEST_CASE("Map: single grid point matches the closed-form makespan") {
    using namespace dls::bench;
    MapConfig c;
    c.solver = "single-round"; c.baseProcessors = 1; c.baseLoad = 100.0;
    c.baseCommStartup = 1.0; c.baseCommRate = 0.1; c.baseComputeRate = 0.2;
    c.xAxis = MapAxis::CommRate;  c.xMin = 0.1; c.xMax = 0.1; c.xSteps = 1;
    c.yAxis = MapAxis::ComputeRate; c.yMin = 0.2; c.yMax = 0.2; c.ySteps = 1;
    PerformanceMap m = computePerformanceMap(c);
    REQUIRE(m.grid.size() == 1);
    REQUIRE(m.grid[0].size() == 1);
    CHECK(m.grid[0][0] == doctest::Approx(1.0 + 0.1 * 100 + 0.2 * 100));   // S + C·V + A·V = 31
}

TEST_CASE("Map: CSV writer emits header + one row per y tick") {
    using namespace dls::bench;
    MapConfig c;
    c.solver = "single-round";
    c.xAxis = MapAxis::Processors; c.xMin = 1; c.xMax = 3; c.xSteps = 3;
    c.yAxis = MapAxis::CommRate;   c.yMin = 0.1; c.yMax = 0.2; c.ySteps = 2;
    PerformanceMap m = computePerformanceMap(c);
    std::ostringstream out;
    writePerformanceMap(out, m);
    const std::string s = out.str();
    CHECK(s.find("isoline map") != std::string::npos);
    CHECK(s.find("y\\x,1,2,3") != std::string::npos);     // header row with x ticks
    // Two comment lines + header + 2 data rows = 5 newlines.
    CHECK(std::count(s.begin(), s.end(), '\n') == 5);
}

//=================== front-end contract: expandSchedule + JSON =============
// Validates the GUI-enabling layer: full per-fragment timing reconstruction
// (for a Gantt chart) and the JSON emitters that serialize the contract.

TEST_CASE("expandSchedule: reconstructed makespan equals the solver's makespan") {
    // Heterogeneous, ample memory, single-installment per processor.
    DLSInstance inst({{0.1, 0.11, 0.52, 1e9}, {0.2, 0.21, 0.22, 1e9},
                      {0.3, 0.31, 0.32, 1e9}}, 1000.0);
    DLSSolution s = SimplexScheduleEvaluator().evaluate(inst, {0, 1, 2}, EvaluatorConfig{});
    REQUIRE(s.status == SolveStatus::Optimal);

    const double rebuilt = expandSchedule(inst, s);
    CHECK(rebuilt == doctest::Approx(s.makespan).epsilon(1e-4));   // matches the LP makespan
    for (const LoadFragment& f : s.fragments) {                    // timing is now consistent
        CHECK(f.commFinish    >= f.commStart    - 1e-9);
        CHECK(f.computeStart  >= f.commFinish   - 1e-9);
        CHECK(f.computeFinish >= f.computeStart - 1e-9);
        CHECK(f.computeFinish <= s.makespan + 1e-4);
    }
}

TEST_CASE("expandSchedule: honours release time on the compute start") {
    DLSInstance inst({{0.0, 0.1, 0.2, 1e9}}, 100.0);
    inst.processors()[0].releaseTime = 50.0;                       // can't compute before t=50
    DLSSolution s = SimplexScheduleEvaluator().evaluate(inst, {0}, EvaluatorConfig{});
    REQUIRE(s.status == SolveStatus::Optimal);
    const double rebuilt = expandSchedule(inst, s);
    CHECK(s.fragments[0].computeStart >= 50.0 - 1e-9);
    CHECK(rebuilt == doctest::Approx(s.makespan).epsilon(1e-4));
}

TEST_CASE("JSON: instance emitter carries the full model") {
    DLSInstance inst({energyProc(1.0, 0.1, 0.2, 2.0, 3.0, 5.0, 10.0)}, 100.0);
    inst.setOriginatorPowerNetwork(8.0);
    inst.setOriginatorPowerIdle(4.0);
    std::ostringstream out;
    writeInstanceJson(out, inst);
    const std::string j = out.str();
    CHECK(j.find("\"class\":\"dls\"") != std::string::npos);
    CHECK(j.find("\"totalLoad\":100") != std::string::npos);
    CHECK(j.find("\"originator\":{\"powerNetwork\":8") != std::string::npos);
    CHECK(j.find("\"power\":{\"idle\":3") != std::string::npos);
    CHECK(j.find("\"energyPieces\":[[0,2]]") != std::string::npos);
}

TEST_CASE("JSON: solution emitter exposes status, KPIs and Gantt timing") {
    DLSInstance inst({energyProc(1.0, 0.1, 0.2, 2.0, 3.0, 5.0, 10.0)}, 100.0);
    inst.setOriginatorPowerNetwork(8.0); inst.setOriginatorPowerIdle(4.0);
    DLSSolution s = SimplexScheduleEvaluator().evaluate(inst, {0}, EvaluatorConfig{});
    expandSchedule(inst, s);
    std::ostringstream out;
    writeSolutionJson(out, s);
    const std::string j = out.str();
    CHECK(j.find("\"status\":\"Optimal\"") != std::string::npos);
    CHECK(j.find("\"feasible\":true") != std::string::npos);
    CHECK(j.find("\"makespan\":31") != std::string::npos);
    CHECK(j.find("\"energy\":473") != std::string::npos);
    CHECK(j.find("\"computeFinish\":31") != std::string::npos);
}

TEST_CASE("JSON: string escaping handles quotes, backslashes and control chars") {
    CHECK(dls::json::str("a\"b") == "\"a\\\"b\"");
    CHECK(dls::json::str("a\\b") == "\"a\\\\b\"");
    CHECK(dls::json::str("a\nb") == "\"a\\nb\"");
}

//=================== time–energy Pareto front (#Phase2) ====================
TEST_CASE("timeEnergyFront: monotone front from fast/high-energy to slow/low-energy") {
    // M0 slow compute but cheap energy; M1 fast but energy-costly (criteria diverge).
    DLSInstance inst({energyProc(1.0, 0.1, 0.2, 1.0, 2.0, 4.0, 6.0),
                      energyProc(1.0, 0.1, 0.1, 5.0, 2.0, 4.0, 6.0)}, 100.0);
    inst.setOriginatorPowerNetwork(6.0);
    inst.setOriginatorPowerIdle(2.0);
    SimplexScheduleEvaluator ev;
    const std::vector<int> seq = {0, 1};

    std::vector<ParetoPoint> f = timeEnergyFront(inst, ev, seq, 6);
    REQUIRE(f.size() >= 2);                                   // a real trade-off exists

    // Sorted by makespan ascending; energy strictly non-increasing along it.
    for (std::size_t i = 1; i < f.size(); ++i) {
        CHECK(f[i].makespan >= f[i - 1].makespan - 1e-6);
        CHECK(f[i].energy   <= f[i - 1].energy   + 1e-6);
    }
    CHECK(f.front().energy > f.back().energy);                // faster end costs more energy

    // Endpoints equal the direct single-criterion optima for this sequence.
    EvaluatorConfig tc; tc.objective = EvalObjective::MinMakespan;
    EvaluatorConfig ec; ec.objective = EvalObjective::MinCost;
    CHECK(f.front().makespan == doctest::Approx(ev.evaluate(inst, seq, tc).makespan));
    CHECK(f.back().energy    == doctest::Approx(ev.evaluate(inst, seq, ec).energy));
}

TEST_CASE("timeEnergyFront: collapses to one point when there is no trade-off") {
    // Single processor: one feasible split, so min-makespan == min-energy schedule.
    DLSInstance inst({energyProc(1.0, 0.1, 0.2, 2.0, 3.0, 5.0, 10.0)}, 100.0);
    SimplexScheduleEvaluator ev;
    std::vector<ParetoPoint> f = timeEnergyFront(inst, ev, {0}, 5);
    REQUIRE(f.size() == 1);
    CHECK(f[0].makespan == doctest::Approx(31.0));
}

//=================== §Validation: three-tier framework ========================
// Structural invariants (Tier 1), analytic anchors (Tier 2), stress sweeps
// (Tier 3), and regression guards.  Added to catch algorithmic bugs such as
// the FPTAS OptV load-overflow when T >> V (first caught: 2025-07).

// ---- shared helpers --------------------------------------------------------

// Studio-default 3-processor DLS instance (V=1000).
static DLSInstance studioInst() {
    return DLSInstance({
        {0.1, 0.11, 0.52, 4000.0},   // P0
        {0.2, 0.21, 0.22, 5000.0},   // P1
        {0.3, 0.31, 0.32, 1500.0}    // P2
    }, 1000.0);
}

// Same geometry as studioInst() but Cᵢ=0 and V=1000.
// FPTAS OptV requires commRate == 0 for all processors.
static DLSInstance ci0StudioInst() {
    DLSInstance inst = ci0Instance({{0.1, 0.52}, {0.2, 0.22}, {0.3, 0.32}});
    inst.setTotalLoad(1000.0);
    return inst;
}

// Names of makespan-minimising solvers that accept a plain DLSInstance.
static std::vector<std::string> makespanSolvers() {
    return {"ga", "best-rate", "online", "single-round", "exact", "exact-dual"};
}

// Check structural invariants of a feasible DLS solution.
// isOptV=true  : solver maximises load within a deadline  (Σαᵢ ≤ V is fine).
// isOptV=false : solver minimises makespan                (Σαᵢ must equal V).
//
// Timing-ordering (commFinish >= commStart, etc.) and comm-overlap checks are
// only applied to fragments where computeFinish > 0. LP-based solvers (exact,
// best-rate, exact-dual) leave computeFinish = 0 (not populated); those checks
// are skipped for them automatically.
static void checkInvariants(const std::string& /*label*/,
                             const DLSSolution& sol,
                             const DLSInstance& inst,
                             bool isOptV = false) {
    const double V   = inst.totalLoad();
    const double tol = 1e-6;
    REQUIRE(sol.feasible());

    const double loaded = sol.totalAssignedLoad();
    CHECK(loaded >= -tol);
    CHECK(loaded <= V + tol);
    if (!isOptV)
        CHECK(loaded == doctest::Approx(V).epsilon(1e-4));

    std::vector<std::pair<double,double>> cv;   // comm intervals for overlap check
    for (const LoadFragment& f : sol.fragments) {
        CHECK(f.loadSize  >= -tol);
        CHECK(f.commStart >= -tol);
        if (f.computeFinish > 0.0) {   // timing is populated (not LP-partial)
            CHECK(f.commFinish    >= f.commStart    - tol);
            CHECK(f.computeStart  >= f.commFinish   - tol);
            CHECK(f.computeFinish >= f.computeStart - tol);
            CHECK(f.computeFinish <= sol.makespan   + tol);
            if (f.commFinish > f.commStart + tol)
                cv.push_back({f.commStart, f.commFinish});
        }
    }

    if (!isOptV && !sol.fragments.empty())
        CHECK(sol.makespan >= divisibleLoadLowerBoundTight(inst) - tol);

    if (cv.size() > 1) {
        std::sort(cv.begin(), cv.end());
        for (std::size_t i = 1; i < cv.size(); ++i)
            CHECK(cv[i].first >= cv[i-1].second - tol);
    }
}

// ---- Tier 1: structural invariants -----------------------------------------

TEST_CASE("Validation T1: studio instance — all makespan solvers pass structural invariants") {
    const DLSInstance inst = studioInst();
    SolverOptions opt; opt.maxInstallments = 3; opt.allowRepeats = true;
    for (const std::string& name : makespanSolvers()) {
        auto s = makeSolver(name, opt);
        REQUIRE(s != nullptr);
        DLSSolution sol = s->solve(inst, SolverConfig{42});
        checkInvariants(name, sol, inst);
    }
}

TEST_CASE("Validation T1: single-processor instance — all makespan solvers pass structural invariants") {
    // P0: S=0, C=0.2, A=0.5, B=∞. One processor computes everything; T=(C+A)·V.
    DLSInstance inst({{0.0, 0.2, 0.5, 1e9}}, 100.0);
    SolverOptions opt; opt.maxInstallments = 3;
    for (const std::string& name : makespanSolvers()) {
        auto s = makeSolver(name, opt);
        REQUIRE(s != nullptr);
        DLSSolution sol = s->solve(inst, SolverConfig{1});
        checkInvariants(name, sol, inst);
    }
}

TEST_CASE("Validation T1: five-processor instance — all makespan solvers pass structural invariants") {
    DLSInstance inst({
        {0.0, 0.23, 0.12, 1e9},
        {0.1, 0.11, 0.23, 1e9},
        {0.0, 0.11, 0.11, 1e9},
        {0.2, 0.14, 0.31, 1e9},
        {0.0, 0.11, 0.11, 1e9}
    }, 5000.0);
    SolverOptions opt; opt.maxInstallments = 5; opt.allowRepeats = false;
    for (const std::string& name : makespanSolvers()) {
        auto s = makeSolver(name, opt);
        REQUIRE(s != nullptr);
        DLSSolution sol = s->solve(inst, SolverConfig{99});
        checkInvariants(name, sol, inst);
    }
}

// ---- Tier 2: analytic anchors -----------------------------------------------

TEST_CASE("Validation T2: single-processor closed form T = (C+A)·V when S=0") {
    // With one processor and S=0, the origin distributes V to itself; T=(C+A)·V.
    DLSInstance inst({{0.0, 0.2, 0.5, 1e9}}, 100.0);
    const double expected = (0.2 + 0.5) * 100.0;                          // 70.0
    for (const char* name : {"single-round", "exact", "exact-dual"}) {
        auto s = makeSolver(name);
        REQUIRE(s != nullptr);
        DLSSolution sol = s->solve(inst, SolverConfig{});
        REQUIRE(sol.feasible());
        CHECK(sol.makespan == doctest::Approx(expected));
        CHECK(sol.conservesLoad(100.0, 1e-4));
    }
}

TEST_CASE("Validation T2: homogeneous 2-processor closed form T = V·A/2 when C=S=0") {
    // Two identical processors, no startup, no comm: each gets V/2; T = V·A/2.
    const double A = 1.0, V = 100.0;
    DLSInstance inst({{0.0, 0.0, A, 1e9}, {0.0, 0.0, A, 1e9}}, V);
    const double expected = homogeneousSingleRound(0.0, 0.0, A, V, 2).makespan;   // 50.0
    DLSSolution sol = makeSolver("single-round")->solve(inst, SolverConfig{});
    REQUIRE(sol.feasible());
    CHECK(sol.makespan == doctest::Approx(expected));
    CHECK(sol.conservesLoad(V, 1e-4));
}

TEST_CASE("Validation T2: FPTAS OptV with large T saturates at V (V-cap correctness)") {
    // T=5000 >> V=1000: uncapped Σαᵢ ≈ 47 000. After the fix, assigned load = V.
    DLSInstance inst = ci0StudioInst();
    FptasOptVParams fp; fp.deadline = 5000.0; fp.epsilon = 0.1;
    DLSSolution sol = FptasOptVSolver(fp).solve(inst, SolverConfig{});
    REQUIRE(sol.status == SolveStatus::Feasible);
    CHECK(sol.totalAssignedLoad() == doctest::Approx(1000.0).epsilon(1e-4));
    CHECK(sol.makespan <= 5000.0 + 1e-6);
    checkInvariants("fptas-optv T=5000", sol, inst, /*isOptV=*/true);
}

TEST_CASE("Validation T2: FPTAS OptV with T < min(Sᵢ) yields an empty schedule") {
    // All processors have Sᵢ ≥ 0.1; T=0.05 < 0.1 so none can be activated.
    DLSInstance inst = ci0StudioInst();
    FptasOptVParams fp; fp.deadline = 0.05; fp.epsilon = 0.1;
    DLSSolution sol = FptasOptVSolver(fp).solve(inst, SolverConfig{});
    REQUIRE(sol.feasible());
    CHECK(sol.fragments.empty());
    CHECK(sol.totalAssignedLoad() == doctest::Approx(0.0));
    CHECK(sol.makespan == doctest::Approx(0.0));
}

TEST_CASE("Validation T2: FPTAS OptT single-processor closed form T = S + V·A") {
    DLSInstance inst = ci0Instance({{2.0, 0.5}});   // S=2, A=0.5
    inst.setTotalLoad(20.0);
    DLSSolution sol = FptasOptTSolver({0.1}).solve(inst, SolverConfig{});
    REQUIRE(sol.feasible());
    CHECK(sol.makespan == doctest::Approx(2.0 + 20.0 * 0.5));   // 12.0
    CHECK(sol.conservesLoad(20.0, 1e-4));
}

TEST_CASE("Validation T2: MapReduce with C=0 splits load equally between identical mappers") {
    // 2 identical mappers A=1, V=10, all defaults (S=C=0, γ₀=1, r=1, t_red=0).
    // Equal-finish → each mapper gets 5; T = 5.
    MapReduceInstance inst({1.0, 1.0}, 10.0);
    MapReduceSolution s = MapReduceSolver().solve(inst);
    REQUIRE(s.status == SolveStatus::Optimal);
    CHECK(s.totalAssignedLoad() == doctest::Approx(10.0));
    CHECK(s.makespan            == doctest::Approx(5.0));
    for (double a : s.mapperLoads) CHECK(a == doctest::Approx(5.0).epsilon(1e-4));
}

// ---- Tier 3: stress sweeps --------------------------------------------------

TEST_CASE("Validation T3: FPTAS OptV assigned load is non-decreasing in T") {
    // Increasing the deadline can never reduce the amount of load processable.
    DLSInstance inst = ci0StudioInst();   // V=1000
    double prev = -1.0;
    for (double T : {0.05, 0.15, 0.3, 0.5, 1.0, 5.0, 20.0, 100.0, 521.0, 5000.0}) {
        FptasOptVParams fp; fp.deadline = T; fp.epsilon = 0.1;
        DLSSolution sol = FptasOptVSolver(fp).solve(inst, SolverConfig{});
        REQUIRE(sol.feasible());
        double load = sol.totalAssignedLoad();
        CHECK(load >= prev - 1e-6);      // non-decreasing
        CHECK(load <= 1000.0 + 1e-6);   // never exceeds V
        prev = load;
    }
}

TEST_CASE("Validation T3: FPTAS OptV assigned load never exceeds V across V range") {
    // Fix T=100; sweep V from tiny (overflow-prone) to large. Always: loaded <= V.
    for (double V : {0.5, 1.0, 10.0, 100.0, 500.0, 1000.0, 5000.0}) {
        DLSInstance inst = ci0Instance({{0.1, 0.52}, {0.2, 0.22}, {0.3, 0.32}});
        inst.setTotalLoad(V);
        FptasOptVParams fp; fp.deadline = 100.0; fp.epsilon = 0.1;
        DLSSolution sol = FptasOptVSolver(fp).solve(inst, SolverConfig{});
        REQUIRE(sol.feasible());
        CHECK(sol.totalAssignedLoad() <= V + 1e-6);
    }
}

TEST_CASE("Validation T3: makespan is non-increasing as identical processors are added (C=S=0)") {
    // S=C=0, A=1: each additional processor halves the per-processor load.
    // T = V/m strictly decreasing; single-round reproduces the closed form.
    const double A = 1.0, V = 100.0;
    double prev = std::numeric_limits<double>::infinity();
    for (int m = 1; m <= 6; ++m) {
        std::vector<Processor> ps(static_cast<std::size_t>(m));
        for (Processor& p : ps) { p.computeRate = A; p.memoryLimit = 1e9; }
        DLSInstance inst(ps, V);
        DLSSolution sol = makeSolver("single-round")->solve(inst, SolverConfig{});
        REQUIRE(sol.feasible());
        CHECK(sol.makespan <= prev + 1e-6);
        CHECK(sol.makespan == doctest::Approx(V * A / m).epsilon(1e-4));
        prev = sol.makespan;
    }
}

// ---- Regression guards ------------------------------------------------------

TEST_CASE("[regression] FPTAS OptV: assigned load never exceeds V when T >> V") {
    // Bug: large T caused uncapped Σαᵢ ≈ 47 000 on a V=1000 instance.
    // Fixed by scaling all fragments down when totalAssigned > V.
    const DLSInstance inst = ci0StudioInst();
    for (double T : {1000.0, 2000.0, 5000.0, 10000.0, 1e6}) {
        FptasOptVParams fp; fp.deadline = T; fp.epsilon = 0.1;
        DLSSolution sol = FptasOptVSolver(fp).solve(inst, SolverConfig{});
        REQUIRE(sol.feasible());
        CHECK(sol.totalAssignedLoad() <= 1000.0 + 1e-6);
    }
}

TEST_CASE("[regression] makespan gap is never negative for standard makespan solvers") {
    // Bug: gap was displayed as (makespan - LB_for_full_V) / LB_for_full_V for OptV
    // solvers, which gave -52% when T < time needed to process all V.
    // Here we verify makespan >= LB for all makespan-minimising solvers.
    const DLSInstance inst = studioInst();
    const double lb = divisibleLoadLowerBoundTight(inst);
    SolverOptions opt; opt.maxInstallments = 3;
    for (const std::string& name : makespanSolvers()) {
        auto s = makeSolver(name, opt);
        REQUIRE(s != nullptr);
        DLSSolution sol = s->solve(inst, SolverConfig{5});
        REQUIRE(sol.feasible());
        CHECK(sol.makespan >= lb - 1e-6);
    }
}

// Temporary debug: trace what FptasOptT produces for the failing OptT case
