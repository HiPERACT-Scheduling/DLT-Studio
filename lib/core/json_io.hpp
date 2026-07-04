//---------------------------------------------------------------------------
// core/json_io.hpp
//
// Minimal, dependency-free JSON *emitters* for the DLS contract — the stable
// serialization a front-end (web app, notebook, desktop GUI) consumes. Only the
// write side lives here: a GUI driving the library through typed bindings builds
// instances directly, while display / HTTP / CLI paths need structured output.
//
// No third-party JSON library: emitting JSON is just careful number/string
// formatting, so this stays header-only and builds in every configuration. The
// shapes are documented inline and mirror DLSInstance / DLSSolution one-to-one.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_JSON_IO_HPP
#define DLS_CORE_JSON_IO_HPP

#include <cstdio>
#include <ostream>
#include <sstream>
#include <string>

#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"

namespace dls {
namespace json {

// Goal:   format a double with full round-trip precision for JSON.
inline std::string num(double v) {
    std::ostringstream ss;
    ss.precision(17);
    ss << v;
    return ss.str();
}

// Goal:   escape a string as a JSON string literal (including the quotes).
inline std::string str(const std::string& s) {
    std::string o = "\"";
    for (char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {  // other control chars
                    char buf[8];
                    std::snprintf(buf, sizeof buf, "\\u%04x", c);
                    o += buf;
                } else {
                    o += c;
                }
        }
    }
    o += "\"";
    return o;
}

// Goal:   stable JSON name of a solve status.
inline const char* statusName(SolveStatus s) {
    switch (s) {
        case SolveStatus::NotSolved:  return "NotSolved";
        case SolveStatus::Optimal:    return "Optimal";
        case SolveStatus::Feasible:   return "Feasible";
        case SolveStatus::Infeasible: return "Infeasible";
        case SolveStatus::Unbounded:  return "Unbounded";
        case SolveStatus::Failure:    return "Failure";
    }
    return "Unknown";
}

}  // namespace json

// Goal:   emit a DLSInstance as a JSON object (the GUI's editable problem shape).
// Input:  out - the stream; inst - the instance.
// Output: none (written to out). Optional sub-objects (originator power, per-proc
//         power, energy/compute pieces) appear only when populated.
inline void writeInstanceJson(std::ostream& out, const DLSInstance& inst) {
    using namespace json;
    out << "{\"class\":\"dls\""
        << ",\"totalLoad\":" << num(inst.totalLoad())
        << ",\"beta\":"      << num(inst.resultFraction());
    if (inst.originatorPowerNetwork() != 0.0 || inst.originatorPowerIdle() != 0.0)
        out << ",\"originator\":{\"powerNetwork\":" << num(inst.originatorPowerNetwork())
            << ",\"powerIdle\":" << num(inst.originatorPowerIdle()) << "}";
    if (!inst.name().empty()) out << ",\"name\":" << str(inst.name());
    out << ",\"processors\":[";
    const auto& ps = inst.processors();
    for (std::size_t i = 0; i < ps.size(); ++i) {
        const Processor& p = ps[i];
        if (i) out << ",";
        out << "{\"S\":" << num(p.commStartup)
            << ",\"C\":"  << num(p.commRate)
            << ",\"A\":"  << num(p.computeRate)
            << ",\"B\":"  << num(p.memoryLimit)
            << ",\"p\":"  << num(p.computeStartup)
            << ",\"r\":"  << num(p.releaseTime)
            << ",\"d\":"  << num(p.deadline)
            << ",\"f\":"  << num(p.fixedCost)
            << ",\"l\":"  << num(p.linearCost);
        if (p.powerIdle != 0.0 || p.powerStartup != 0.0 || p.powerNetwork != 0.0)
            out << ",\"power\":{\"idle\":" << num(p.powerIdle)
                << ",\"startup\":" << num(p.powerStartup)
                << ",\"network\":" << num(p.powerNetwork) << "}";
        if (!p.energyPieces.empty()) {
            out << ",\"energyPieces\":[";
            for (std::size_t k = 0; k < p.energyPieces.size(); ++k)
                out << (k ? "," : "") << "[" << num(p.energyPieces[k].intercept)
                    << "," << num(p.energyPieces[k].slope) << "]";
            out << "]";
        }
        if (!p.computePieces.empty()) {
            out << ",\"computePieces\":[";
            for (std::size_t k = 0; k < p.computePieces.size(); ++k)
                out << (k ? "," : "") << "[" << num(p.computePieces[k].intercept)
                    << "," << num(p.computePieces[k].slope) << "]";
            out << "]";
        }
        if (!p.name.empty()) out << ",\"name\":" << str(p.name);
        out << "}";
    }
    out << "]}";
}

// Goal:   emit a DLSSolution as a JSON object (drives KPI cards + the Gantt).
// Input:  out - the stream; sol - the solution (run expandSchedule() first to
//         fill the per-fragment timing the Gantt needs).
// Output: none (written to out).
inline void writeSolutionJson(std::ostream& out, const DLSSolution& sol) {
    using namespace json;
    out << "{\"status\":" << str(statusName(sol.status))
        << ",\"feasible\":" << (sol.feasible() ? "true" : "false")
        << ",\"makespan\":" << num(sol.makespan)
        << ",\"cost\":"     << num(sol.cost)
        << ",\"energy\":"   << num(sol.energy)
        << ",\"iterations\":" << sol.iterations
        << ",\"wallTimeSec\":" << num(sol.wallTimeSec)
        << ",\"usedSeed\":" << sol.usedSeed
        << ",\"sequence\":[";
    for (std::size_t i = 0; i < sol.sequence.size(); ++i)
        out << (i ? "," : "") << sol.sequence[i];
    out << "],\"fragments\":[";
    for (std::size_t i = 0; i < sol.fragments.size(); ++i) {
        const LoadFragment& f = sol.fragments[i];
        if (i) out << ",";
        out << "{\"proc\":" << f.processorId
            << ",\"load\":" << num(f.loadSize)
            << ",\"carried\":" << num(f.carriedLoad)
            << ",\"commStart\":" << num(f.commStart)
            << ",\"commFinish\":" << num(f.commFinish)
            << ",\"computeStart\":" << num(f.computeStart)
            << ",\"computeFinish\":" << num(f.computeFinish) << "}";
    }
    out << "]}";
}

}  // namespace dls

#endif  // DLS_CORE_JSON_IO_HPP
