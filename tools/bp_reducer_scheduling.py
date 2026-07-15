#!/usr/bin/env python3
"""
tools/bp_reducer_scheduling.py

RESEARCH PROTOTYPE — not yet wired into the C++ library, CLI, API, or Studio.
This is milestone 1 (correct math, validated) of the branch-and-price plan;
production integration is a separate, later milestone.

Problem: heterogeneous multi-channel MapReduce reducer read scheduling.

The existing MapReduce model (mapreduce_solver.hpp, Berlińska thesis §4.4)
handles "many reducers" two ways: (1) treat them as one r-times-faster
aggregate resource (no real per-reducer decision), or (2) a bisection-width
(channel-capacity) constraint with a FIXED, formula-derived round-robin read
order — the assignment of who-reads-what-when is never actually optimized.
This prototype fills that gap for the case the thesis doesn't solve:
heterogeneous reducers (different read rates) with an OPTIMIZED read schedule
under the same channel-capacity constraint, optionally with per-reducer
affinity (only some mapper outputs matter to some reducers).

Model (discrete time ticks, integer durations — exact arithmetic, no rounding
error to confound correctness validation):
  - m mapper outputs. Mapper i's output becomes readable at ready[i] (an input
    here — computed upstream by the existing closed-form mapper-side solver in
    a real integration; this prototype takes it as given so it can focus on
    the genuinely unsolved part, the read schedule).
  - r reducers, each with an integer read rate rate[j] (ticks per work unit).
    duration(i, j) = size[i] * rate[j] (heterogeneous: reducers read the same
    output at different speeds).
  - capacity l: at most l reads may be "in progress" (started, not yet
    finished) across ALL reducers at any single tick — the shared channel /
    bisection-width limit.
  - affinity[j]: the set of mapper indices reducer j must read. Defaults to
    every mapper for every reducer (the thesis's assumption); the "advanced"
    switch is simply a smaller per-reducer subset — the algorithm doesn't
    change shape, only which ticks/columns are feasible.
  - objective, switchable:
      "makespan" — minimize the latest reducer finish time (classic Cmax).
      "balance"  — minimize the SUM of reducer finish times (classic ΣCj /
                   total completion time): a reducer sitting idle while
                   others hog the channel counts against this objective even
                   if the overall Cmax is unchanged. This is the standard,
                   well-established alternative scheduling criterion (Cmax vs.
                   ΣCj) — not "total busy time," which is a fixed input under
                   fixed affinity and so wouldn't be an interesting objective.

Branch-and-price:
  - A COLUMN = one reducer's complete, feasible read plan (an order and exact
    start tick for every mapper in its affinity set), found in isolation
    (ignoring other reducers — the channel-sharing coupling lives in the
    master, not the column).
  - MASTER LP: pick a convex combination of generated columns per reducer
    (one column per reducer in the integer solution), respecting the shared
    channel-capacity constraint at every tick, minimizing the chosen
    objective. Solved with scipy's HiGHS-backed linprog.
  - PRICING: for one reducer, given the master's dual prices, find the single
    most-improving new column — a shortest-path DP over (subset-read, tick)
    states. Exact for the modest sizes this prototype targets.
  - BRANCHING: simple binary branching on "does reducer j use column p"
    (forbid / force), the most straightforward valid branch-and-price
    branching scheme — adequate for a correctness-focused prototype; a real
    integration would want a stronger branching rule (see the honest
    limitations note at the bottom of this file).

Validation: every instance this module can solve at "tiny" scale is also
solved by exhaustive brute force (validate_random, brute_force_optimal), and
run() asserts the two agree — the standing rule for any new solver in this
codebase: ship validation, don't just ship an algorithm.
"""
from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

INF = float("inf")


# ---- instance ---------------------------------------------------------------

@dataclass
class Instance:
    """Goal: describe one reducer-read-scheduling instance.
    ready/size are per mapper; rate is per reducer; capacity is the shared
    channel limit; affinity[j] is the sorted list of mapper indices reducer j
    must read (default: all of them); objective is 'makespan' or 'balance'."""
    ready: List[int]
    size: List[int]
    rate: List[int]
    capacity: int
    affinity: List[List[int]] = field(default_factory=list)
    objective: str = "makespan"

    def __post_init__(self):
        m, r = len(self.ready), len(self.rate)
        if not self.affinity:
            self.affinity = [list(range(m)) for _ in range(r)]
        assert self.objective in ("makespan", "balance")
        assert len(self.size) == m
        assert len(self.affinity) == r

    @property
    def m(self) -> int:
        return len(self.ready)

    @property
    def r(self) -> int:
        return len(self.rate)

    def duration(self, i: int, j: int) -> int:
        return self.size[i] * self.rate[j]

    def horizon(self) -> int:
        """Goal: a safe upper bound on any reducer's finish tick, accounting
        for worst-case channel contention (not just a reducer's own reads):
        with capacity as low as 1, every read across every reducer could end
        up fully serialized one after another.
        Output: max ready time + the sum of every read duration across every
        reducer's affinity set + 1."""
        total = sum(self.duration(i, j) for j in range(self.r) for i in self.affinity[j])
        return max(self.ready, default=0) + total + 1


@dataclass
class Column:
    """Goal: one reducer's complete feasible read plan.
    order/starts describe when each mapper in the reducer's affinity set is
    read; finish is the tick the last read completes; occupied is the sorted
    set of ticks during which some read of this plan is in progress (used to
    build the master's shared capacity constraint)."""
    reducer: int
    starts: Dict[int, int]
    finish: int
    occupied: Tuple[int, ...]

    def key(self) -> tuple:
        """Goal: a hashable identity so the column pool can dedupe.
        Output: a tuple usable as a dict/set key."""
        return (self.reducer, tuple(sorted(self.starts.items())))


# ---- pricing: per-reducer shortest-path DP over (subset, tick) --------------

def price_reducer(inst: Instance, j: int, mu: Dict[int, float], finish_weight: float,
                   horizon: int, forced_tick: Dict[int, int] = None,
                   banned_ticks: Dict[int, set] = None) -> List[Column]:
    """Goal: find reducer j's improving read plans given the master's current
    dual prices (the pricing subproblem of column generation) and this
    branch-and-price node's per-mapper timing constraints.
    Inputs: inst; reducer index j; mu = dual price per tick of the shared
    capacity constraint; finish_weight = the coefficient the master's
    objective places on this reducer's finish tick (the epigraph dual in
    makespan mode, or the constant 1 in balance mode, unified so both
    objectives share one DP); horizon = ticks to search up to; forced_tick[i]
    = the only tick mapper i may start at (branching "start(i) = t" side);
    banned_ticks[i] = ticks mapper i may NOT start at (branching
    "start(i) != t" side). Both keyed by mapper index, this reducer only —
    the natural, DP-enforceable branching variable (unlike forbidding a
    whole column by identity, which the DP has no way to search around).
    Output: ALL distinct-finish-tick plans reconstructable from the DP (one
    per achievable finish tick, not just the single global-best one) — a
    single best-reduced-cost column per call is only GUARANTEED to find some
    improving column when one exists, not every column an eventual integer
    solution needs; at a degenerate LP vertex several dual vectors can be
    simultaneously valid, and one of them can genuinely never reveal a
    column that only helps in combination with a DIFFERENT reducer's choice.
    Returning every distinct-finish candidate (cheap: the DP already computes
    dp[full][t] for every t) gives column generation many more chances to
    escape that stall. The caller filters by actual reduced cost."""
    forced_tick = forced_tick or {}
    banned_ticks = banned_ticks or {}
    req = inst.affinity[j]
    n = len(req)
    if n == 0:
        return None
    full = (1 << n) - 1
    # dp[mask][t] = best (min) accumulated (-sum mu) value reachable having
    # read exactly `mask` and become free at tick t; parent[...] reconstructs
    # the plan. Both indexed [mask][tick].
    dp = [[None] * (horizon + 1) for _ in range(1 << n)]
    parent: List[List[Optional[tuple]]] = [[None] * (horizon + 1) for _ in range(1 << n)]
    dp[0][0] = 0.0
    mu_prefix = [0.0] * (horizon + 2)
    for t in range(horizon + 1):
        mu_prefix[t + 1] = mu_prefix[t] + mu.get(t, 0.0)

    def occ_cost(a: int, b: int) -> float:
        return mu_prefix[b] - mu_prefix[a]   # sum of mu[t] for t in [a, b)

    for t in range(horizon):
        for mask in range(1 << n):
            val = dp[mask][t]
            if val is None:
                continue
            # idle one tick (lets the DP wait out an expensive window)
            if dp[mask][t + 1] is None or dp[mask][t + 1] > val:
                dp[mask][t + 1] = val
                parent[mask][t + 1] = (mask, t, None)
            # start reading one unread mapper now
            for bit in range(n):
                if mask & (1 << bit):
                    continue
                i = req[bit]
                if t < inst.ready[i]:
                    continue
                if i in forced_tick and t != forced_tick[i]:
                    continue
                if t in banned_ticks.get(i, ()):
                    continue
                dur = inst.duration(i, j)
                finish_t = t + dur
                if finish_t > horizon:
                    continue
                nval = val - occ_cost(t, finish_t)
                nmask = mask | (1 << bit)
                if dp[nmask][finish_t] is None or dp[nmask][finish_t] > nval:
                    dp[nmask][finish_t] = nval
                    parent[nmask][finish_t] = (mask, t, i)

    def reconstruct(t: int) -> Column:
        starts: Dict[int, int] = {}
        mask, tt = full, t
        while mask != 0 or tt != 0:
            pm, pt, i = parent[mask][tt]
            if i is not None:
                starts[i] = pt
            mask, tt = pm, pt
        occupied = set()
        for i, s in starts.items():
            occupied.update(range(s, s + inst.duration(i, j)))
        return Column(reducer=j, starts=starts, finish=t, occupied=tuple(sorted(occupied)))

    return [reconstruct(t) for t in range(horizon + 1) if dp[full][t] is not None]


def plan_reduced_cost(inst: Instance, col: Column, pi: float, mu: Dict[int, float],
                      finish_weight: float) -> float:
    """Goal: the reduced cost of a specific column (used both to decide
    whether pricing found an improving column, and, in branch-and-price, to
    re-check forced/forbidden columns). Output: finish_weight*finish - pi -
    sum(mu over occupied ticks)."""
    return finish_weight * col.finish - pi - sum(mu.get(t, 0.0) for t in col.occupied)


# ---- master LP + column generation ------------------------------------------

@dataclass
class ColGenResult:
    """Goal: the outcome of solving one node's LP relaxation by column
    generation. lam[(reducer, column_index)] = fractional weight; obj = the
    LP's objective value (a valid lower bound at this branch-and-price node);
    columns[j] = the pool of generated columns for reducer j."""
    lam: Dict[Tuple[int, int], float]
    obj: float
    columns: List[List[Column]]
    feasible: bool


def _greedy_column_for_order(inst: Instance, j: int, order: Sequence[int],
                             ft: Dict[int, int], bt: Dict[int, set]) -> Optional[Column]:
    """Goal: the earliest-finish column for reducer j that reads its mappers
    in exactly this order, respecting readiness and this node's per-mapper
    timing constraints (always feasible against OTHER reducers in isolation
    — capacity sharing is the master's concern, not this construction's).
    Output: the Column, or None if `order`/ft/bt can't be reconciled (e.g. a
    forced tick earlier than a still-in-progress prior read)."""
    t = 0
    starts: Dict[int, int] = {}
    for i in order:
        s = max(t, inst.ready[i])
        if i in ft:
            if s > ft[i]:
                return None
            s = ft[i]
        while s in bt.get(i, ()):
            s += 1
            if i in ft and s > ft[i]:
                return None
        starts[i] = s
        t = s + inst.duration(i, j)
    occupied = set()
    for i, s in starts.items():
        occupied.update(range(s, s + inst.duration(i, j)))
    return Column(reducer=j, starts=starts, finish=t if order else 0,
                 occupied=tuple(sorted(occupied)))


def _seed_columns(inst: Instance, forced_tick: Dict[int, Dict[int, int]] = None,
                  banned_ticks: Dict[int, Dict[int, set]] = None) -> List[List[Column]]:
    """Goal: seed each reducer's column pool richly enough that column
    generation reliably escapes degenerate-dual stalls (a single best-
    reduced-cost column per pricing call is only GUARANTEED to find some
    improving column, not every column the eventual integer optimum needs —
    at a degenerate LP vertex, several dual vectors can all be locally
    valid, and one of them can simply never reveal a column that only helps
    in combination with another reducer's choice; the standard, cheap
    mitigation for a prototype at this scale is to seed a wide variety of
    "natural" columns up front, rather than lean entirely on pricing to
    discover everything from a single greedy start).
    Inputs: inst; forced_tick[j][i] / banned_ticks[j][i] — this node's
    branching constraints, reducer j, mapper i.
    Output: columns[j] = one Column per distinct ordering of reducer j's
    required mapper set (deduplicated by key). Only tractable because
    affinity sets are small at this prototype's validation scale — a real
    integration would need a less exhaustive seed (see the file-level notes
    on this limitation)."""
    forced_tick = forced_tick or {}
    banned_ticks = banned_ticks or {}
    cols: List[List[Column]] = []
    for j in range(inst.r):
        ft = forced_tick.get(j, {})
        bt = banned_ticks.get(j, {})
        seen = set()
        pool: List[Column] = []
        for order in itertools.permutations(inst.affinity[j]):
            col = _greedy_column_for_order(inst, j, order, ft, bt)
            if col is not None and col.key() not in seen:
                seen.add(col.key())
                pool.append(col)
        cols.append(pool)
    return cols


def solve_node(inst: Instance, horizon: int,
              forced_tick: Dict[int, Dict[int, int]] = None,
              banned_ticks: Dict[int, Dict[int, set]] = None,
              max_iters: int = 200) -> ColGenResult:
    """Goal: solve one branch-and-price node's LP relaxation via column
    generation.
    Inputs: inst; horizon (tick search bound); forced_tick[j][i] /
    banned_ticks[j][i] = this node's branching constraints on when reducer j
    may start reading mapper i (see price_reducer); max_iters caps the
    generation loop (safety net against float-tolerance stalls).
    Output: ColGenResult. feasible=False means this node's branching
    constraints admit no valid schedule at all (a real, expected outcome of
    branching — e.g. forcing two reducers into a genuine capacity deadlock —
    not a bug; the search prunes it like any other infeasible node)."""
    forced_tick = forced_tick or {}
    banned_ticks = banned_ticks or {}
    columns: List[List[Column]] = _seed_columns(inst, forced_tick, banned_ticks)

    lam, obj, duals, solved, still_infeasible = _solve_master(inst, columns, horizon)
    for _ in range(max_iters):
        if not solved:
            return ColGenResult(lam={}, obj=INF, columns=columns, feasible=False)
        pi, mu, rho = duals
        improved = False
        for j in range(inst.r):
            fw = _finish_weight(inst, j, duals)
            candidates = price_reducer(inst, j, mu, fw, horizon,
                                       forced_tick.get(j), banned_ticks.get(j))
            existing = {c.key() for c in columns[j]}
            # Add every improving candidate this call found, not just the
            # single best — see price_reducer's docstring on why relying on
            # only the global-best column can stall at a degenerate vertex.
            for col in candidates:
                if col.key() in existing:
                    continue
                rc = plan_reduced_cost(inst, col, pi[j], mu, fw)
                if rc < -1e-7:
                    columns[j].append(col)
                    existing.add(col.key())
                    improved = True
        if not improved:
            break
        lam, obj, duals, solved, still_infeasible = _solve_master(inst, columns, horizon)
    if not solved or still_infeasible:
        return ColGenResult(lam={}, obj=INF, columns=columns, feasible=False)
    return ColGenResult(lam=lam, obj=obj, columns=columns, feasible=True)


def _finish_weight(inst: Instance, j: int, duals) -> float:
    """Goal: the coefficient the master's objective places on reducer j's
    finish tick, unifying both objective modes for the pricing DP.
    Output: the epigraph dual rho_j in makespan mode; the constant 1.0 in
    balance mode (finish time enters the objective directly there)."""
    if inst.objective == "balance":
        return 1.0
    _, _, rho = duals
    return rho[j]


def _solve_master(inst: Instance, columns: List[List[Column]], horizon: int):
    """Goal: solve the restricted master LP for the current column pools.
    Inputs: inst; columns (every column already respects the current node's
    forced_tick/banned_ticks constraints — enforced inside price_reducer's
    DP, not here); horizon.
    Output: (lam dict, objective value, (pi, mu, rho) duals, solved bool,
    still_infeasible bool). pi[j] = convexity dual; mu[t] = capacity dual;
    rho[j] = epigraph dual (makespan mode only, else zeros).

    Two Big-M feasibility relaxations keep this master solvable (and its
    duals meaningful) at every column-generation iteration, even before
    enough columns exist to satisfy the real constraints:
      - a penalized slack per capacity row (absorbs a temporary capacity
        violation — needed the moment two reducers' only known columns
        happen to conflict, before pricing has found a way around it);
      - a penalized artificial variable per convexity row (absorbs "no
        constraint-respecting column known yet for reducer j" — needed the
        moment branching constraints leave a reducer with an empty seed,
        before pricing has found a first real one).
    Both are driven to zero by column generation as soon as real columns can
    replace them; either still being positive at final convergence means the
    node is genuinely infeasible (only expected under branching restrictions
    that truly admit no schedule, not from the base instance itself — a
    fully serial one-read-at-a-time fallback always exists there)."""
    idx: List[Tuple[int, int]] = []          # (reducer, column position) per LP variable
    for j in range(inst.r):
        for p in range(len(columns[j])):
            idx.append((j, p))
    ncols = len(idx)
    has_T = inst.objective == "makespan"
    used_ticks = [t for t in range(horizon + 1)
                  if any(t in columns[j][p].occupied
                        for j in range(inst.r) for p in range(len(columns[j])))]
    n_cap_slack = len(used_ticks)
    n_art = inst.r
    M = 10.0 * (inst.r * (horizon + 1) + 10)   # dominates any real objective value
    nvars = ncols + (1 if has_T else 0) + n_cap_slack + n_art
    T_idx = ncols if has_T else None
    slack_base = ncols + (1 if has_T else 0)
    art_base = slack_base + n_cap_slack

    c = np.zeros(nvars)
    if has_T:
        c[T_idx] = 1.0
    else:
        for k, (j, p) in enumerate(idx):
            c[k] = columns[j][p].finish
    c[slack_base:slack_base + n_cap_slack] = M
    c[art_base:art_base + n_art] = M

    # equality: convexity, one row per reducer (plus its artificial column)
    A_eq = np.zeros((inst.r, nvars))
    b_eq = np.ones(inst.r)
    for k, (j, p) in enumerate(idx):
        A_eq[j, k] = 1.0
    for j in range(inst.r):
        A_eq[j, art_base + j] = 1.0

    rows_ub = []
    rhs_ub = []
    if has_T:
        for j in range(inst.r):
            row = np.zeros(nvars)
            for k, (jj, p) in enumerate(idx):
                if jj == j:
                    row[k] = columns[j][p].finish
            row[T_idx] = -1.0
            rows_ub.append(row); rhs_ub.append(0.0)

    for si, t in enumerate(used_ticks):
        row = np.zeros(nvars)
        for k, (j, p) in enumerate(idx):
            if t in columns[j][p].occupied:
                row[k] = 1.0
        row[slack_base + si] = -1.0     # Σ(reads at t) - slack <= capacity
        rows_ub.append(row); rhs_ub.append(float(inst.capacity))

    A_ub = np.array(rows_ub) if rows_ub else None
    b_ub = np.array(rhs_ub) if rhs_ub else None
    bounds = [(0, None)] * nvars

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return {}, INF, ({}, {}, {}), False, True

    slack_vals = res.x[slack_base:slack_base + n_cap_slack]
    art_vals = res.x[art_base:art_base + n_art]
    relax_total = float(slack_vals.sum()) + float(art_vals.sum())
    still_infeasible = relax_total > 1e-6

    lam = {idx[k]: res.x[k] for k in range(ncols) if res.x[k] > 1e-9}
    pi = {j: res.eqlin.marginals[j] for j in range(inst.r)}
    mu: Dict[int, float] = {}
    rho = {j: 0.0 for j in range(inst.r)}
    row_i = 0
    if has_T:
        for j in range(inst.r):
            rho[j] = res.ineqlin.marginals[row_i]
            row_i += 1
    for t in used_ticks:
        mu[t] = res.ineqlin.marginals[row_i]
        row_i += 1
    obj = INF if still_infeasible else (res.fun - M * relax_total)
    return lam, obj, (pi, mu, rho), True, still_infeasible


# ---- branch-and-price --------------------------------------------------------

def _pick_branch_var(res: ColGenResult, j: int) -> Tuple[int, int]:
    """Goal: pick the (mapper, tick) branching variable for a fractional
    reducer j — a mapper where its positive-weight columns disagree on the
    start tick (guaranteed to exist: distinct columns differ in `starts` by
    construction of Column.key(), so if reducer j has more than one
    positive-weight column, at least one mapper's tick must differ).
    Inputs: the node's ColGenResult; the fractional reducer index j.
    Output: (mapper index, tick) — branch into force(mapper)=tick vs
    ban(mapper)=tick. Ties broken by the highest-weight column's choice."""
    picks = sorted(((p, w) for (jj, p), w in res.lam.items() if jj == j),
                   key=lambda pw: -pw[1])
    top_starts = res.columns[j][picks[0][0]].starts
    for p, _ in picks[1:]:
        other = res.columns[j][p].starts
        for i, t in top_starts.items():
            if other.get(i) != t:
                return i, t
    raise AssertionError("fractional reducer has no disagreeing mapper — Column.key() invariant broken")


def branch_and_price(inst: Instance, horizon: Optional[int] = None,
                     max_nodes: int = 500) -> Tuple[float, Dict[int, Column]]:
    """Goal: solve the instance to proven optimality via branch-and-price.
    Inputs: inst; horizon (default: inst.horizon()); max_nodes safety cap.
    Output: (optimal objective value, {reducer: chosen Column}).

    Branches on "does reducer j start reading mapper i at tick t" — the
    natural DP-enforceable decision (see price_reducer) — rather than on
    whole-column identity, which the pricing subproblem has no way to search
    around once its single best candidate happens to be excluded."""
    horizon = horizon if horizon is not None else inst.horizon()
    best_obj = INF
    best_sol: Dict[int, Column] = {}
    # node = (forced_tick: {j: {mapper: tick}}, banned_ticks: {j: {mapper: {tick,...}}})
    stack = [({}, {})]
    nodes = 0
    while stack and nodes < max_nodes:
        nodes += 1
        forced_tick, banned_ticks = stack.pop()
        res = solve_node(inst, horizon, forced_tick, banned_ticks)
        if not res.feasible or res.obj >= best_obj - 1e-9:
            continue   # infeasible or can't beat the incumbent — prune
        # find a fractional reducer (one whose lambda isn't a clean 0/1 pick)
        frac_j = None
        chosen: Dict[int, Column] = {}
        for j in range(inst.r):
            picks = [(p, w) for (jj, p), w in res.lam.items() if jj == j]
            if len(picks) == 1 and picks[0][1] > 1 - 1e-6:
                chosen[j] = res.columns[j][picks[0][0]]
            elif frac_j is None:
                frac_j = j
        if frac_j is None:
            if res.obj < best_obj - 1e-9:
                best_obj, best_sol = res.obj, chosen
            continue
        i, t = _pick_branch_var(res, frac_j)
        force2 = {k: dict(v) for k, v in forced_tick.items()}
        force2.setdefault(frac_j, {})[i] = t
        stack.append((force2, banned_ticks))
        ban2 = {k: {m: set(s) for m, s in v.items()} for k, v in banned_ticks.items()}
        ban2.setdefault(frac_j, {}).setdefault(i, set()).add(t)
        stack.append((forced_tick, ban2))
    return best_obj, best_sol


# ---- brute-force ground truth (validation only, tiny instances) -------------

def _feasible_schedule_cost(inst: Instance, orders: List[Tuple[int, ...]], horizon: int):
    """Goal: given a FIXED per-reducer read order, find the true minimum-cost
    schedule respecting readiness, per-reducer sequencing, and the shared
    capacity limit — a memoized DP over (tick, per-reducer position, per-
    reducer free-at tick) states. Voluntary idling (a reducer waits even
    though capacity and a ready item are both available) is kept as a real
    option: with a SHARED capacity limit across reducers, sometimes waiting
    for a better near-term combination beats greedily filling every slot
    (unlike ordinary single-machine list scheduling, where it never helps) —
    so this cannot be simplified to "always start whatever's ready."
    Inputs: inst; orders[j] = the fixed sequence of mapper indices reducer j
    reads (from inst.affinity[j]); horizon.
    Output: (objective value, {reducer: finish tick}) for the best schedule
    achievable under this fixed ordering, or (INF, {}) if none fits."""
    r = inst.r
    memo: Dict[tuple, float] = {}
    choice: Dict[tuple, tuple] = {}   # state -> (subset started, next free_at per reducer in subset)

    def solve(t: int, pos: Tuple[int, ...], free_at: Tuple[int, ...]) -> float:
        if all(pos[j] == len(orders[j]) for j in range(r)):
            return max(free_at) if inst.objective == "makespan" else sum(free_at)
        if t > horizon:
            return INF
        key = (t, pos, free_at)
        if key in memo:
            return memo[key]
        candidates = [j for j in range(r)
                     if pos[j] < len(orders[j]) and free_at[j] <= t
                     and inst.ready[orders[j][pos[j]]] <= t]
        in_progress = sum(1 for f in free_at if f > t)
        room = inst.capacity - in_progress
        best_val, best_choice = INF, None
        # voluntary idle (skip this tick without starting anyone) is always
        # a candidate move, in addition to every feasible subset of starts.
        options = [frozenset()]
        for size in range(1, min(len(candidates), room) + 1):
            options.extend(frozenset(s) for s in itertools.combinations(candidates, size))
        for subset in options:
            npos, nfree = list(pos), list(free_at)
            for j in subset:
                i = orders[j][pos[j]]
                npos[j] += 1
                nfree[j] = t + inst.duration(i, j)
            val = solve(t + 1, tuple(npos), tuple(nfree))
            if val < best_val:
                best_val, best_choice = val, subset
        memo[key] = best_val
        choice[key] = best_choice
        return best_val

    obj = solve(0, tuple([0] * r), tuple([0] * r))
    if obj == INF:
        return INF, {}
    # reconstruct the finish ticks by replaying the recorded best choices
    t, pos, free_at = 0, tuple([0] * r), tuple([0] * r)
    while not all(pos[j] == len(orders[j]) for j in range(r)):
        subset = choice[(t, pos, free_at)]
        npos, nfree = list(pos), list(free_at)
        for j in subset:
            i = orders[j][pos[j]]
            npos[j] += 1
            nfree[j] = t + inst.duration(i, j)
        t, pos, free_at = t + 1, tuple(npos), tuple(nfree)
    return obj, {j: free_at[j] for j in range(r)}


def brute_force_optimal(inst: Instance, horizon: Optional[int] = None) -> float:
    """Goal: ground truth via full exhaustive search — every combination of
    per-reducer read orders, exact-fit over start times. Only tractable for
    TINY instances (this is the point: it's the cross-check, not a solver).
    Output: the true optimal objective value."""
    horizon = horizon if horizon is not None else inst.horizon()
    best = INF
    order_choices = [list(itertools.permutations(inst.affinity[j])) for j in range(inst.r)]
    for combo in itertools.product(*order_choices):
        obj, _ = _feasible_schedule_cost(inst, list(combo), horizon)
        best = min(best, obj)
    return best


# ---- validation harness -------------------------------------------------------

def random_instance(rng: random.Random, m_max=3, r_max=2, cap_max=2) -> Instance:
    """Goal: a random TINY instance (small enough for brute force).
    Output: an Instance with random ready times, sizes, rates, capacity, and
    (sometimes) partial affinity."""
    m = rng.randint(2, m_max)
    r = rng.randint(1, r_max)
    ready = [rng.randint(0, 3) for _ in range(m)]
    size = [rng.randint(1, 3) for _ in range(m)]
    rate = [rng.randint(1, 2) for _ in range(r)]
    capacity = rng.randint(1, cap_max)
    if rng.random() < 0.5:
        affinity = [sorted(rng.sample(range(m), rng.randint(1, m))) for _ in range(r)]
    else:
        affinity = [list(range(m)) for _ in range(r)]
    objective = rng.choice(["makespan", "balance"])
    return Instance(ready=ready, size=size, rate=rate, capacity=capacity,
                    affinity=affinity, objective=objective)


def validate_random(n: int = 40, seed: int = 0) -> None:
    """Goal: cross-check branch_and_price against brute_force_optimal on n
    random tiny instances, covering both objectives and both affinity modes.
    Output: none; raises AssertionError with the offending instance on the
    first mismatch, else prints a pass summary."""
    rng = random.Random(seed)
    checked = {"makespan": 0, "balance": 0, "full": 0, "partial": 0}
    for trial in range(n):
        inst = random_instance(rng)
        bf = brute_force_optimal(inst)
        bp_obj, _ = branch_and_price(inst)
        full = all(inst.affinity[j] == list(range(inst.m)) for j in range(inst.r))
        assert abs(bf - bp_obj) < 1e-6, (
            f"MISMATCH trial={trial} objective={inst.objective} "
            f"brute_force={bf} branch_and_price={bp_obj} inst={inst}"
        )
        checked[inst.objective] += 1
        checked["full" if full else "partial"] += 1
    print(f"validate_random: {n}/{n} instances matched brute force "
         f"(makespan={checked['makespan']}, balance={checked['balance']}, "
         f"full-affinity={checked['full']}, partial-affinity={checked['partial']})")


if __name__ == "__main__":
    validate_random()
