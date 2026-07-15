//---------------------------------------------------------------------------
// mapreduce/mapreduce_instance.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Problem definition for scheduling MapReduce computations as a divisible load
// (Berlińska thesis, chapter 4). This is the third problem class in the
// library, after single-load DLS (core/) and MLSD (mlsd/). It is the first DLS
// model with *precedence*: a map phase produces intermediate data that a
// reduce phase consumes.
//
// Model (single-port star, macroscopic view):
//   - A divisible input load V is partitioned into mapper chunks α₁..α_m.
//   - Mapper i runs on processor Pi and processes its chunk at macroscopic rate
//     Aᵢ (time per byte; folds in reading, computing, storing). A one-off code
//     startup S elapses per processor, sequentially (the "mS" term).
//   - Each mapper emits γ₀·αᵢ bytes of intermediate results (γ₀ = result
//     multiplicity fraction).
//   - r reducers each read γ₀·V/r bytes — a chunk γ₀·αᵢ/r from every mapper —
//     at read rate C, then run in time t_red = s_red + a_red·(z·log₂z),
//     z = γ₀·V/r (the τ(x)=a_red·x·log₂x sorting complexity).
//   - Objective: minimize the total schedule length T.
//
// Only the mapper computing rates Aᵢ differ between processors; S, C, γ₀, the
// reducer parameters and r are global. See MODEL.md for the symbol glossary.
//
// bisectionWidth (l) additionally caps concurrent mapper-to-reducer read
// channels; the plain closed-form MapReduceSolver ignores it (it assumes an
// unconstrained network). Only mapreduce_bwidth_solver.hpp reads it.
//---------------------------------------------------------------------------

#ifndef DLS_MAPREDUCE_INSTANCE_HPP
#define DLS_MAPREDUCE_INSTANCE_HPP

#include <string>
#include <vector>

namespace dls {

class MapReduceInstance {
public:
    MapReduceInstance() = default;

    // Goal:   construct a ready-to-solve MapReduce instance.
    // Input:  mapperRates - Aᵢ (time/byte) for each mapper processor;
    //         totalLoad   - V, the input load to partition.
    MapReduceInstance(std::vector<double> mapperRates, double totalLoad)
        : mapperRates_(std::move(mapperRates)), totalLoad_(totalLoad) {}

    // ---- problem data (mappers) ------------------------------------------
    std::vector<double>&       mapperRates()       { return mapperRates_; }
    const std::vector<double>& mapperRates() const { return mapperRates_; }
    std::size_t numMappers() const { return mapperRates_.size(); }

    double totalLoad() const      { return totalLoad_; }
    void   setTotalLoad(double v) { totalLoad_ = v; }

    // ---- global parameters (with fluent setters) -------------------------
    double startup() const         { return startup_; }        // S: per-processor code load
    double readRate() const        { return readRate_; }       // C: reducer read rate (time/byte)
    double resultFraction() const  { return resultFraction_; } // γ₀: output/input ratio
    int    numReducers() const     { return numReducers_; }    // r
    double reducerStartup() const  { return reducerStartup_; } // s_red
    double reducerRate() const     { return reducerRate_; }    // a_red: reducer compute rate
    int    bisectionWidth() const  { return bisectionWidth_; } // l: concurrent read channels (mapreduce_bwidth_solver.hpp only)

    void setStartup(double s)         { startup_ = s; }
    void setReadRate(double c)        { readRate_ = c; }
    void setResultFraction(double g)  { resultFraction_ = g; }
    void setNumReducers(int r)        { numReducers_ = r; }
    void setReducerStartup(double s)  { reducerStartup_ = s; }
    void setReducerRate(double a)     { reducerRate_ = a; }
    void setBisectionWidth(int l)     { bisectionWidth_ = l; }

    const std::string& name() const { return name_; }
    void setName(std::string n)     { name_ = std::move(n); }

    // ---- validation ------------------------------------------------------
    // Goal:   check the instance is well-formed before solving.
    // Input:  error - optional out-param; set to a reason on failure.
    // Output: true if valid, false otherwise.
    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& m) { if (error) *error = m; return false; };
        if (mapperRates_.empty()) return fail("instance has no mappers");
        if (totalLoad_ <= 0.0)    return fail("totalLoad (V) must be positive");
        if (numReducers_ < 1)     return fail("numReducers (r) must be >= 1");
        for (std::size_t i = 0; i < mapperRates_.size(); ++i)
            if (mapperRates_[i] <= 0.0)
                return fail("mapper " + std::to_string(i) + " has a non-positive rate Aᵢ");
        if (startup_ < 0.0 || readRate_ < 0.0 || resultFraction_ < 0.0 ||
            reducerStartup_ < 0.0 || reducerRate_ < 0.0)
            return fail("a global parameter is negative");
        if (bisectionWidth_ < 1) return fail("bisectionWidth (l) must be >= 1");
        if (error) error->clear();
        return true;
    }

private:
    std::vector<double> mapperRates_;        // Aᵢ per mapper
    double              totalLoad_      = 0.0;  // V
    double              startup_        = 0.0;  // S
    double              readRate_       = 0.0;  // C
    double              resultFraction_ = 1.0;  // γ₀
    int                 numReducers_    = 1;    // r
    double              reducerStartup_ = 0.0;  // s_red
    double              reducerRate_    = 0.0;  // a_red
    int                 bisectionWidth_ = 1;    // l (ignored by the plain closed-form MapReduceSolver)
    std::string         name_;
};

}  // namespace dls

#endif  // DLS_MAPREDUCE_INSTANCE_HPP
