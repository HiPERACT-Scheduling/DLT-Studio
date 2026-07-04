//---------------------------------------------------------------------------
// core/dls_instance.hpp
//
// Problem definition for Divisible Load Scheduling (DLS).
//
// This is the *problem* half of the parameter split: it describes the
// instance to be solved and nothing about how to solve it. Every solver
// (exact or heuristic) consumes the same DLSInstance, which is what gives
// the library its uniform interface.
//
// Model (single master / star network, single-port, multi-installment):
//   - A divisible load of total size `totalLoad` (V) must be partitioned
//     and distributed to worker processors.
//   - The load is dispatched as a sequence of installments. Communication
//     is sequential (single-port): the master sends one fragment at a time.
//   - A processor may receive several installments (it can appear multiple
//     times in the activation sequence).
//   - Objective: minimize the makespan Cmax (time until the last processor
//     finishes computing its assigned load).
//
// Per-processor cost coefficients (affine communication + computation),
// matching the four columns of the legacy proc_par table:
//   commStartup  S : fixed latency to start a transfer to this processor
//   commRate     C : transfer time per unit of load
//   computeRate  A : computation time per unit of load
//   memoryLimit  B : maximum load a single installment may carry (buffer)
//
// NOTE: the legacy LP reuses S as a startup for both the communication and
// the computation term. We keep a single startup field here to stay faithful
// to that model; if a separate computation startup is ever needed, add a
// `computeStartup` field rather than overloading S.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_DLS_INSTANCE_HPP
#define DLS_CORE_DLS_INSTANCE_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace dls {

// Communication / computation model assumed by the schedule evaluator.
// Only SinglePort is implemented today; the enum is here so the contract
// does not have to change when other models are added.
enum class CommunicationModel {
    SinglePort   // master communicates with one worker at a time (sequential)
};

// One linear piece of a convex piecewise-linear processing-time function:
// processing time of load α on this piece is intercept + slope·α.
struct ComputePiece {
    double intercept = 0.0;   // a : fixed time component of the piece
    double slope     = 0.0;   // b : time per unit of load on the piece
};

// One linear piece of a convex piecewise-linear *running-energy* function:
// energy to process load α on this piece is intercept + slope·α. This is the
// memory-hierarchy energy model of Marszałkowski (2020), §3.3: an in-core piece
// {l=0, k=k₁} and a steeper out-of-core piece {l=l₂<0, k=k₂>k₁}, the two
// crossing at the available-RAM size ρ. Symbols mirror the thesis:
//   slope     k : running energy per unit of load on the piece   (e.g. J/MB)
//   intercept l : fixed energy offset of the piece               (e.g. J)
// Empty (default) => the implicit single piece {0, linearCost}, i.e. the classic
// proportional energy l·α that the thesis shows is inaccurate out of core.
struct EnergyPiece {
    double intercept = 0.0;   // l : fixed energy component of the piece
    double slope     = 0.0;   // k : energy per unit of load on the piece
};

// Cost coefficients of one worker processor.
struct Processor {
    double commStartup    = 0.0;   // S : fixed latency per transfer
    double commRate       = 0.0;   // C : communication time per unit load
    double computeRate    = 0.0;   // A : computation time per unit load
    double memoryLimit    = 0.0;   // B : max load per installment (<=0 => unbounded)
    double computeStartup = 0.0;   // p : computation startup time (per installment)
    double releaseTime    = 0.0;   // r : processor unavailable before this time
    double deadline       = 0.0;   // d : completion deadline (<=0 => none/infinity)
    double fixedCost      = 0.0;   // f : fixed cost of using this processor
    double linearCost     = 0.0;   // l : cost per unit of load on this processor
    // Optional convex piecewise-linear processing time: the processing time of
    // load α is max over pieces of (intercept + slope·α), modeling a memory
    // hierarchy (a fast core-RAM piece + a steeper virtual/disk piece). Empty
    // (default) => the single implicit piece {computeStartup, computeRate},
    // i.e. the classic affine time p + A·α.
    std::vector<ComputePiece> computePieces{};

    // ---- energy model (Marszałkowski 2020) — all optional, default off -----
    // Power rates of the worker's non-computing states (e.g. Watts). The
    // computing state's energy is captured directly by energyPieces below
    // (which fold the running power × time into energy-per-load), matching the
    // thesis ε(α) formulation. A machine idle for time tᴵ spends P^I·tᴵ, etc.
    double powerIdle      = 0.0;   // P^I : power while powered-on but not busy
    double powerStartup   = 0.0;   // P^S : power while waking up (over startup S)
    double powerNetwork   = 0.0;   // P^N : power while sending/receiving
    // Convex piecewise-linear running energy ε(α) = max over pieces of
    // (intercept + slope·α). Empty => the implicit single piece {0, linearCost},
    // reproducing the classic proportional energy l·α exactly.
    std::vector<EnergyPiece> energyPieces{};

    std::string name{};            // optional, for reporting

    // Goal:   report whether this processor has any per-unit cost.
    // Input:  none.
    // Output: true if it has a positive communication or computation rate.
    bool hasCost() const {
        if (commRate > 0.0 || computeRate > 0.0) return true;
        for (const ComputePiece& pc : computePieces)
            if (pc.slope > 0.0) return true;
        return false;
    }

    // Goal:   resolve the effective processing-time pieces.
    // Input:  none.
    // Output: computePieces if set, else the single implicit piece
    //         {computeStartup, computeRate} (today's affine model).
    std::vector<ComputePiece> effectivePieces() const {
        if (!computePieces.empty()) return computePieces;
        return {ComputePiece{computeStartup, computeRate}};
    }

    // Goal:   resolve the effective running-energy pieces.
    // Input:  none.
    // Output: energyPieces if set, else the single implicit piece
    //         {0, linearCost} (the classic proportional energy l·α).
    std::vector<EnergyPiece> effectiveEnergyPieces() const {
        if (!energyPieces.empty()) return energyPieces;
        return {EnergyPiece{0.0, linearCost}};
    }

    // Goal:   report whether this processor carries any explicit energy data.
    // Input:  none.
    // Output: true if energy pieces or any state power rate are set (i.e. this
    //         processor participates in the four-state energy accounting).
    bool hasEnergyModel() const {
        return !energyPieces.empty() ||
               powerIdle > 0.0 || powerStartup > 0.0 || powerNetwork > 0.0;
    }
};

class DLSInstance {
public:
    // Goal:  construct an empty instance (to be populated via setters).
    DLSInstance() = default;

    // Goal:   construct a ready-to-solve instance.
    // Input:  processors - worker cost coefficients; totalLoad - V, the
    //         total divisible load to distribute.
    // Output: an instance owning a copy of the processor list.
    DLSInstance(std::vector<Processor> processors, double totalLoad)
        : processors_(std::move(processors)), totalLoad_(totalLoad) {}

    // ---- problem data ----------------------------------------------------
    // Goal:   access the worker processor list (mutable / read-only).
    // Output: reference to the processor vector.
    std::vector<Processor>&       processors()       { return processors_; }
    const std::vector<Processor>& processors() const { return processors_; }

    // Goal:   number of worker processors in the instance.
    // Output: the processor count.
    std::size_t numProcessors() const { return processors_.size(); }

    // Goal:   get / set the total divisible load V.
    // Input:  v - the new total load (setter only).
    // Output: the current total load (getter only).
    double totalLoad() const         { return totalLoad_; }
    void   setTotalLoad(double v)    { totalLoad_ = v; }

    // Goal:   get / set the result-return fraction β (results of size β·αᵢ are
    //         returned to the master after computation; 0 = no return, default).
    // Input:  b - the fraction (setter only).
    // Output: the current fraction (getter only).
    double resultFraction() const     { return resultFraction_; }
    void   setResultFraction(double b) { resultFraction_ = b; }

    // Goal:   get / set the originator (master M₀) power rates for the energy
    //         model. The originator draws P^N while starting workers and
    //         distributing load, and P^I once all load is dispatched (it does
    //         not suspend mid-schedule). Default 0 => the master contributes no
    //         energy (workers-only accounting), keeping legacy instances inert.
    // Input:  p - the power rate (setter only).
    // Output: the current rate (getter only).
    double originatorPowerNetwork() const   { return originatorPowerNetwork_; }
    void   setOriginatorPowerNetwork(double p) { originatorPowerNetwork_ = p; }
    double originatorPowerIdle() const       { return originatorPowerIdle_; }
    void   setOriginatorPowerIdle(double p)  { originatorPowerIdle_ = p; }

    // Goal:   report whether this instance uses the energy model at all.
    // Input:  none.
    // Output: true if any processor has energy data or the originator has a
    //         power rate; false => pure makespan/linear-cost legacy instance.
    bool usesEnergyModel() const {
        if (originatorPowerNetwork_ > 0.0 || originatorPowerIdle_ > 0.0) return true;
        for (const Processor& p : processors_)
            if (p.hasEnergyModel()) return true;
        return false;
    }

    // Goal:   get / set the communication model.
    // Input:  m - the model to use (setter only).
    // Output: the current model (getter only).
    CommunicationModel model() const     { return model_; }
    void setModel(CommunicationModel m)  { model_ = m; }

    // Goal:   get / set the optional instance name (for reporting).
    // Input:  n - the name to assign (setter only).
    // Output: the current name (getter only).
    const std::string& name() const   { return name_; }
    void setName(std::string n)        { name_ = std::move(n); }

    // ---- validation ------------------------------------------------------
    // Goal:   check that the instance is well-formed before solving.
    // Input:  error - optional out-param; if non-null it receives a
    //         human-readable reason when validation fails (cleared on success).
    // Output: true if the instance is valid, false otherwise.
    [[nodiscard]] bool validate(std::string* error = nullptr) const {
        auto fail = [&](const std::string& msg) {  // set *error and return false
            if (error) *error = msg;
            return false;
        };
        if (processors_.empty())
            return fail("instance has no processors");
        if (totalLoad_ <= 0.0)
            return fail("totalLoad must be positive");
        for (std::size_t i = 0; i < processors_.size(); ++i) {
            const Processor& p = processors_[i];
            if (p.commRate < 0.0 || p.computeRate < 0.0 || p.commStartup < 0.0 ||
                p.computeStartup < 0.0 || p.releaseTime < 0.0)
                return fail("processor " + std::to_string(i) +
                            " has a negative cost/timing coefficient");
            if (!p.hasCost())
                return fail("processor " + std::to_string(i) +
                            " has zero communication and computation rate");
            // Processing time must be non-decreasing in load: every piece slope >= 0.
            // (Intercepts may be negative — e.g. the disk piece of a memory
            // hierarchy crosses below the core piece at the swap point.)
            for (const ComputePiece& pc : p.computePieces)
                if (pc.slope < 0.0)
                    return fail("processor " + std::to_string(i) +
                                " has a compute piece with negative slope");
            // Energy must be non-decreasing in load: every piece slope >= 0.
            // (Intercepts may be negative — the out-of-core piece l₂ < 0 so the
            // two pieces meet at the RAM size, exactly as for time pieces.)
            for (const EnergyPiece& ep : p.energyPieces)
                if (ep.slope < 0.0)
                    return fail("processor " + std::to_string(i) +
                                " has an energy piece with negative slope");
            if (p.powerIdle < 0.0 || p.powerStartup < 0.0 || p.powerNetwork < 0.0)
                return fail("processor " + std::to_string(i) +
                            " has a negative state power rate");
        }
        if (originatorPowerNetwork_ < 0.0 || originatorPowerIdle_ < 0.0)
            return fail("originator power rate must be >= 0");
        if (resultFraction_ < 0.0) return fail("resultFraction (β) must be >= 0");
        if (resultFraction_ > 1.0) return fail("resultFraction (β) must be <= 1");
        if (error) error->clear();
        return true;
    }

private:
    std::vector<Processor> processors_;
    double                 totalLoad_ = 0.0;
    double                 resultFraction_ = 0.0;   // β: result-return fraction (0 = none)
    double                 originatorPowerNetwork_ = 0.0;  // P^N of master M₀ (0 = ignore master)
    double                 originatorPowerIdle_    = 0.0;  // P^I of master M₀
    CommunicationModel     model_     = CommunicationModel::SinglePort;
    std::string            name_;
};

}  // namespace dls

#endif  // DLS_CORE_DLS_INSTANCE_HPP
