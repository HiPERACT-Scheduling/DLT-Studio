# TODO

- [x] Solver Studio (Solve & Gantt tab): for OptV-family solvers (`optv`,
      `fptas-optv`), the UI already replaces the "Lower bound"/gap card with
      "Deadline T" / "Load / V (%)" — correct, since those solvers answer the
      dual question (maximize load within a deadline T) and a gap-to-lower-
      bound comparison against the full-V bound is meaningless for them.
      Done: added a `<Hint>` ("?") on the "Deadline T" card explaining why no
      gap-to-lower-bound appears, so it doesn't just silently disappear.

- [x] Bug found while verifying the above: `optv`'s solver category is
      `"loadmax"`, which wasn't in the Studio's `TIER_ORDER` list, so the
      exact OptV solver was never bucketed into any tier button and was
      unreachable from the algorithm picker under "Maximize load by a
      deadline" — only its FPTAS approximation showed up. Confirmed
      `loadmax` is a deliberate, distinct backend category (its own
      metadata entry, tag "a different question" — not a misclassification),
      so the fix was adding `'loadmax'` to `TIER_ORDER` client-side (not
      recategorizing `optv` server-side). It never collides with `'exact'`:
      the load-objective pool only ever contains `loadmax`/`approx` solvers,
      the time-objective pool only `exact`/etc., so both tiers can share the
      list without ever showing together. Verified live: the tier button now
      reads "Load maximisation" (its own metadata name), `optv` solves
      correctly without needing the FPTAS-only "Infinite bandwidth" tick.
