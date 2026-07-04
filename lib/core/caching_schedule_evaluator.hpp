//---------------------------------------------------------------------------
// core/caching_schedule_evaluator.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// A ScheduleEvaluator decorator that memoizes results by activation sequence.
// Because the underlying LP is deterministic, evaluating the same sequence
// twice yields the same makespan + load split; a GA re-visits the same
// sequences heavily (elitism, selection duplicates, recurring genotypes), so
// caching removes the large majority of LP solves. Returning a cached result
// is identical to re-solving, so behaviour is preserved exactly.
//
// Ownership/threading: hold one instance per solve() run (as GASolver does).
// The cache is per-instance, so concurrent runs never share it.
//
// The cache is unbounded: it grows with the number of distinct sequences seen
// (at most procCount^sequenceLength). That is fine for typical instance sizes;
// for very long runs over large instances a size cap could be added later.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_CACHING_SCHEDULE_EVALUATOR_HPP
#define DLS_CORE_CACHING_SCHEDULE_EVALUATOR_HPP

#include <cstddef>
#include <unordered_map>
#include <vector>

#include "dls_instance.hpp"
#include "dls_solution.hpp"
#include "schedule_evaluator.hpp"

namespace dls {

class CachingScheduleEvaluator : public ScheduleEvaluator {
public:
    // Goal:   wrap an evaluator and memoize its results by sequence.
    // Input:  inner - the evaluator that actually solves (e.g. simplex). It
    //         must outlive this decorator (stored by reference).
    // Output: a caching evaluator.
    explicit CachingScheduleEvaluator(const ScheduleEvaluator& inner)
        : inner_(inner) {}

    // Goal:   identify this evaluator (wraps the inner backend's name).
    // Output: "caching(<inner>)".
    std::string name() const override { return "caching(" + inner_.name() + ")"; }

    // Goal:   return the schedule for `sequence`, solving via the inner
    //         evaluator only on the first occurrence of that sequence.
    // Input:  instance, sequence, config - forwarded to the inner evaluator on
    //         a miss. (Caching assumes instance/config are fixed for the life
    //         of this object, which holds within one solve() run.)
    // Output: the (possibly cached) DLSSolution for the sequence.
    DLSSolution evaluate(const DLSInstance&      instance,
                         const std::vector<int>& sequence,
                         const EvaluatorConfig&  config) const override {
        auto it = cache_.find(sequence);
        if (it != cache_.end()) return it->second;
        DLSSolution sol = inner_.evaluate(instance, sequence, config);
        cache_.emplace(sequence, sol);
        return sol;
    }

    // Goal:   report cache usage (for diagnostics / tests).
    // Output: number of distinct sequences solved and cached.
    std::size_t cachedCount() const { return cache_.size(); }

private:
    // Hash for a sequence (vector<int>) so it can key an unordered_map.
    struct SequenceHash {
        std::size_t operator()(const std::vector<int>& v) const {
            std::size_t h = 1469598103934665603ULL;        // FNV-1a basis
            for (int x : v) {
                h ^= static_cast<std::size_t>(static_cast<unsigned>(x));
                h *= 1099511628211ULL;                     // FNV prime
            }
            return h;
        }
    };

    const ScheduleEvaluator& inner_;   // the real solver being memoized
    mutable std::unordered_map<std::vector<int>, DLSSolution, SequenceHash> cache_;  // sequence -> result
};

}  // namespace dls

#endif  // DLS_CORE_CACHING_SCHEDULE_EVALUATOR_HPP
