//---------------------------------------------------------------------------
// core/instance_io.hpp
// https://github.com/HiPERACT-Scheduling/DLT-Studio
//
// Text reader/writer for a DLSInstance, exposing the *full* model (not just the
// legacy S/C/A/B columns) in a backward-compatible, line-oriented format.
//
// Format (one directive per line; '#' starts a comment):
//   V <load>                       total divisible load V                (semantic)
//   beta <fraction>                result-return fraction β (default 0)
//   <S> <C> <A> <B> [p r d f l]    one processor: 4 required scalars plus up to
//                                  5 optional trailing — computeStartup p,
//                                  releaseTime r, deadline d, fixedCost f,
//                                  linearCost l (each defaults to 0)
//   pieces <a,b> <a,b> ...         convex compute pieces {intercept,slope} for
//                                  the most recently defined processor
//   power <P^I> <P^S> <P^N>        state power rates (idle/startup/network) of the
//                                  most recently defined processor (energy model)
//   energy <l,k> <l,k> ...         convex running-energy pieces {intercept,slope}
//                                  for the most recently defined processor
//   originator <P^N> <P^I>         master M₀ power (networking / idle) for energy
//
// A plain "S C A B" row (the legacy format) still parses unchanged. The reader
// reports structural errors only; semantic checks (V > 0, ...) are left to
// DLSInstance::validate so a caller can still override fields (e.g. --load).
//---------------------------------------------------------------------------

#ifndef DLS_CORE_INSTANCE_IO_HPP
#define DLS_CORE_INSTANCE_IO_HPP

#include <fstream>
#include <ios>
#include <istream>
#include <ostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/dls_instance.hpp"

namespace dls {

// Goal:   parse a DLS instance from a text stream.
// Input:  in - the stream; inst - filled on success; error - set on failure.
// Output: true on a well-formed parse (instance may still be semantically
//         invalid, e.g. V unset — call inst.validate() to check that).
inline bool readInstance(std::istream& in, DLSInstance& inst, std::string& error) {
    std::vector<Processor> procs;
    double V = 0.0, beta = 0.0;
    double origPowerNet = 0.0, origPowerIdle = 0.0;     // master M₀ power (energy model)
    std::string line;
    int lineNo = 0;

    auto fail = [&](const std::string& m) { error = "line " + std::to_string(lineNo) + ": " + m; return false; };

    while (std::getline(in, line)) {
        ++lineNo;
        if (auto h = line.find('#'); h != std::string::npos) line.erase(h);  // strip comment
        std::istringstream ls(line);
        std::string key;
        if (!(ls >> key)) continue;                          // blank line

        if (key == "V" || key == "v") {
            if (!(ls >> V)) return fail("expected a value after 'V'");
        } else if (key == "beta") {
            if (!(ls >> beta)) return fail("expected a value after 'beta'");
        } else if (key == "pieces") {
            if (procs.empty()) return fail("'pieces' before any processor");
            std::string tok;
            while (ls >> tok) {
                const auto comma = tok.find(',');
                if (comma == std::string::npos) return fail("piece '" + tok + "' must be intercept,slope");
                try {
                    ComputePiece pc;
                    pc.intercept = std::stod(tok.substr(0, comma));
                    pc.slope     = std::stod(tok.substr(comma + 1));
                    procs.back().computePieces.push_back(pc);
                } catch (...) { return fail("malformed piece '" + tok + "'"); }
            }
        } else if (key == "energy") {
            if (procs.empty()) return fail("'energy' before any processor");
            std::string tok;
            while (ls >> tok) {
                const auto comma = tok.find(',');
                if (comma == std::string::npos) return fail("energy piece '" + tok + "' must be intercept,slope");
                try {
                    EnergyPiece ep;
                    ep.intercept = std::stod(tok.substr(0, comma));
                    ep.slope     = std::stod(tok.substr(comma + 1));
                    procs.back().energyPieces.push_back(ep);
                } catch (...) { return fail("malformed energy piece '" + tok + "'"); }
            }
        } else if (key == "power") {
            if (procs.empty()) return fail("'power' before any processor");
            double pi, ps, pn;
            if (!(ls >> pi >> ps >> pn)) return fail("'power' needs three values: P^I P^S P^N");
            procs.back().powerIdle = pi; procs.back().powerStartup = ps; procs.back().powerNetwork = pn;
        } else if (key == "originator") {
            if (!(ls >> origPowerNet >> origPowerIdle)) return fail("'originator' needs two values: P^N P^I");
        } else {
            // A processor row: re-read the whole line as numbers.
            std::istringstream nums(line);
            std::vector<double> v;
            double x;
            while (nums >> x) v.push_back(x);
            if (v.size() < 4) return fail("processor row needs at least 'S C A B'");
            if (v.size() > 9) return fail("processor row has too many fields (max S C A B p r d f l)");
            Processor p;
            p.commStartup = v[0]; p.commRate = v[1]; p.computeRate = v[2]; p.memoryLimit = v[3];
            if (v.size() > 4) p.computeStartup = v[4];
            if (v.size() > 5) p.releaseTime    = v[5];
            if (v.size() > 6) p.deadline       = v[6];
            if (v.size() > 7) p.fixedCost      = v[7];
            if (v.size() > 8) p.linearCost     = v[8];
            procs.push_back(p);
        }
    }
    if (procs.empty()) { error = "no processors found"; return false; }

    inst = DLSInstance(std::move(procs), V);
    inst.setResultFraction(beta);
    inst.setOriginatorPowerNetwork(origPowerNet);
    inst.setOriginatorPowerIdle(origPowerIdle);
    error.clear();
    return true;
}

// Goal:   parse a DLS instance from a file path.
inline bool readInstanceFile(const std::string& path, DLSInstance& inst, std::string& error) {
    std::ifstream in(path);
    if (!in) { error = "cannot open instance file: " + path; return false; }
    return readInstance(in, inst, error);
}

// Goal:   serialize an instance back to the same format (round-trippable).
// Input:  out - the stream; inst - the instance.
// Output: none (written to `out`); trailing scalar fields and pieces are emitted
//         only when non-default, so simple instances stay compact.
inline void writeInstance(std::ostream& out, const DLSInstance& inst) {
    const auto oldPrec = out.precision(17);                   // round-trip fidelity
    out << "V " << inst.totalLoad() << "\n";
    if (inst.resultFraction() > 0.0) out << "beta " << inst.resultFraction() << "\n";
    if (inst.originatorPowerNetwork() != 0.0 || inst.originatorPowerIdle() != 0.0)
        out << "originator " << inst.originatorPowerNetwork() << " " << inst.originatorPowerIdle() << "\n";
    for (const Processor& p : inst.processors()) {
        const double trailing[5] = { p.computeStartup, p.releaseTime, p.deadline, p.fixedCost, p.linearCost };
        int n = 0;                                            // how many trailing fields to emit
        for (int i = 0; i < 5; ++i) if (trailing[i] != 0.0) n = i + 1;
        out << p.commStartup << " " << p.commRate << " " << p.computeRate << " " << p.memoryLimit;
        for (int i = 0; i < n; ++i) out << " " << trailing[i];
        out << "\n";
        if (!p.computePieces.empty()) {
            out << "pieces";
            for (const ComputePiece& pc : p.computePieces) out << " " << pc.intercept << "," << pc.slope;
            out << "\n";
        }
        if (p.powerIdle != 0.0 || p.powerStartup != 0.0 || p.powerNetwork != 0.0)
            out << "power " << p.powerIdle << " " << p.powerStartup << " " << p.powerNetwork << "\n";
        if (!p.energyPieces.empty()) {
            out << "energy";
            for (const EnergyPiece& ep : p.energyPieces) out << " " << ep.intercept << "," << ep.slope;
            out << "\n";
        }
    }
    out.precision(oldPrec);
}

}  // namespace dls

#endif  // DLS_CORE_INSTANCE_IO_HPP
