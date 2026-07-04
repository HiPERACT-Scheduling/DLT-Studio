//---------------------------------------------------------------------------
// util/random.hpp
//
// Injectable random-number generator shared across the library. Wrapping
// std::mt19937_64 (instead of the global rand()/srand) makes every stochastic
// algorithm reproducible from an explicit seed and thread-safe: each solver
// run owns its own Rng, so concurrent runs never share generator state.
//---------------------------------------------------------------------------

#ifndef DLS_UTIL_RANDOM_HPP
#define DLS_UTIL_RANDOM_HPP

#include <cstdint>
#include <random>

namespace util {

class Rng {
public:
    // Goal:   construct a generator seeded deterministically.
    // Input:  seed - the seed; the same seed reproduces the same sequence.
    // Output: a ready-to-use generator.
    explicit Rng(std::uint64_t seed) : gen_(seed) {}

    // Goal:   draw a uniform integer in [0, n).
    // Input:  n - exclusive upper bound.
    // Output: an int in [0, n), or 0 if n <= 0.
    int index(int n) {
        if (n <= 0) return 0;
        std::uniform_int_distribution<int> dist(0, n - 1);
        return dist(gen_);
    }

    // Goal:   draw a uniform real in [0, 1).
    // Output: a double in [0, 1).
    double uniform01() {
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        return dist(gen_);
    }

    // Goal:   Bernoulli trial with probability expectedHits/trials (the
    //         semantics of the GA's former IsProbabilityMet).
    // Input:  expectedHits - expected number of hits over `trials` draws;
    //         trials       - the number of trials the rate is expressed over.
    // Output: true with probability clamp(expectedHits/trials, 0, 1).
    bool bernoulli(double expectedHits, long trials) {
        if (trials <= 0) return false;
        double p = expectedHits / static_cast<double>(trials);
        if (p <= 0.0) return false;
        if (p >= 1.0) return true;
        return uniform01() < p;
    }

private:
    std::mt19937_64 gen_;   // underlying 64-bit Mersenne Twister
};

}  // namespace util

#endif  // DLS_UTIL_RANDOM_HPP
