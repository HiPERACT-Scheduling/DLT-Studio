//---------------------------------------------------------------------------
// heuristics/ga/ga.cpp
//
// Thin command-line driver for the genetic-algorithm DLS solver. It parses
// the three legacy input files, converts them into the uniform core types
// (DLSInstance + GAParams), and runs the algorithm through the standard
// dls::DLSSolver::solve() contract. The evolutionary loop itself lives in
// GASolver; this file only does I/O and translation.
//
// Usage: ga prog_par.txt proc_par.txt ga_par.txt
//---------------------------------------------------------------------------

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include "util/time.hpp"
#include "ga_solver.hpp"

using namespace dls;

// Goal:   parse the legacy input files and solve the DLS instance via GASolver.
// Input:  three positional files (programme, processor, GA params) in any
//         order relative to an optional --evaluator=<backend> flag.
// Output: process exit code; prints the best makespan and the wall time.
int main(int argc, char* argv[])
{
    // ---- parse args: 3 positional files + optional --evaluator=<backend> --
    std::string evaluatorBackend = "simplex";   // LP backend ("simplex" | "highs")
    std::vector<std::string> positional;         // the three input-file paths
    const std::string evalFlag = "--evaluator=";
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a.rfind(evalFlag, 0) == 0) evaluatorBackend = a.substr(evalFlag.size());
        else positional.push_back(a);
    }
    if (positional.size() != 3) {
        std::cerr << std::endl << "Wrong number of input parameters!" << std::endl;
        std::cerr << std::endl << "Proper usage:" << std::endl;
        std::cerr << argv[0] << " [--evaluator=simplex|highs]"
                  << " prog_par.txt proc_par.txt ga_par.txt" << std::endl << std::endl;
        return 0;
    }

    // Fail early and clearly if the requested backend isn't available.
    if (!makeScheduleEvaluator(evaluatorBackend)) {
        std::cerr << "Error: unknown or unavailable evaluator backend: '"
                  << evaluatorBackend << "'. Available: simplex"
#ifdef DLS_WITH_HIGHS
                  << ", highs"
#endif
                  << "." << std::endl;
        return 1;
    }

    // ---- parse the legacy input files; bail out with a message on error --
    std::string err;                        // failure reason from a reader

    ProgrammeParameters progPar;            // programme/runtime knobs
    if (!progPar.ReadProgrammeParam(positional[0].c_str(), &err)) {
        std::cerr << "Error: " << err << std::endl;
        return 1;
    }
    progPar.PrintProgrammeParam();

    ProcessorsParameters procData;          // parsed processor cost table
    if (!procData.ReadProcessorsParam(&progPar, positional[1].c_str(), &err)) {
        std::cerr << "Error: " << err << std::endl;
        return 1;
    }
    procData.PrintProcessorsParam(&progPar);

    GeneticAlgorithmParameters gaPar;       // parsed GA control knobs
    if (!gaPar.ReadGAParam(positional[2].c_str(), &err)) {
        std::cerr << "Error: " << err << std::endl;
        return 1;
    }
    gaPar.PrintGAParam();

    // ---- convert the parsed processor table into a uniform DLSInstance ---
    DLSInstance instance;                   // the problem handed to the solver
    instance.setTotalLoad(static_cast<double>(procData.volume));
    for (int k = 1; k <= procData.procNum; ++k)
    {
        Processor p;                        // one worker's cost coefficients
        p.commStartup = procData.procParams[k][PROC_COMM_STARTUP];
        p.commRate    = procData.procParams[k][PROC_COMM_RATE];
        p.computeRate = procData.procParams[k][PROC_COMPUTE_RATE];
        p.memoryLimit = procData.procParams[k][PROC_MEMORY_LIMIT];
        instance.processors().push_back(p);
    }

    // ---- convert GA + programme params into algorithm params -------------
    GAParams gp;                            // GASolver configuration
    gp.populationSize     = gaPar.populationSize;
    gp.maxGenerations     = gaPar.maxIterNum;
    gp.noImprovementLimit = gaPar.improvementNum;
    gp.installments       = gaPar.chromosomeMaxSize;
    gp.crossoverProb      = gaPar.crossoverProb;
    gp.mutationProb       = gaPar.mutationProb;
    gp.lpMaxIterations    = progPar.solverMaxIter;
    gp.evaluatorBackend   = evaluatorBackend;   // CSimplex (default) or HiGHS

    // ---- solve through the uniform contract, timing the run --------------
    Time timex;                             // wall-clock timer
    timex.SetStartTime();

    GASolver solver(gp);
    SolverConfig cfg;                       // seed unset -> GASolver seeds from clock
    DLSSolution sol = solver.solve(instance, cfg);

    timex.SetStopTime();
    double durationMicros = timex.CalcWallTime();   // elapsed time, microseconds

    if (!sol.feasible()) {
        std::cerr << "Error: no feasible schedule found (status "
                  << static_cast<int>(sol.status) << ")" << std::endl;
        return 1;
    }

    // ---- report: best makespan (as the legacy build did), then timing ---
    std::cout << sol.makespan << std::endl;
    std::cout << std::endl << timex.CurrentDateTime() << " | "
              << "Processing time: "
              << std::to_string(durationMicros / 1000000) << " sec." << std::endl;

    return 0;
}
//---------------------------------------------------------------------------
