//---------------------------------------------------------------------------

#ifndef __GAH__
#define __GAH__

//---------------------------------------------------------------------------

#include <stdio.h>
#include <math.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <exception>

using namespace std;

#include <algorithm>
#include <string>
#include <vector>
#include "core/dls_instance.hpp"
#include "core/dls_solution.hpp"
#include "core/schedule_evaluator.hpp"
#include "util/random.hpp"

#define __CHROLENGTH            5
#define __POPULSIZE             10
#define __MAXLENFILENAME        10
#define __TTABLEPAR             2
#define __InputMaxLineLength    200
#define __PRECISION             10000 //0.00001
#define __COMMENT               '#'

// Column meanings of ProcessorsParameters::procParams[proc][col].
enum ProcParamColumn {
    PROC_COMM_STARTUP = 0,  // S: fixed latency to start a transfer
    PROC_COMM_RATE    = 1,  // C: communication time per unit load
    PROC_COMPUTE_RATE = 2,  // A: computation time per unit load
    PROC_MEMORY_LIMIT = 3   // B: max load per installment (buffer)
};



//-----------------------Programme parameters--------------------------------
class ProgrammeParameters
{
   public:
      ProgrammeParameters();
      ~ProgrammeParameters();

      bool ReadProgrammeParam(const char*, std::string* error = nullptr);
      int PrintProgrammeParam();

      int maxProcNum;
      int numProcParams;
      int chromosomeLength;
      int popSize;
      int solverMaxIter;
      std::string fileLP;        // LP task file (legacy; std::string avoids overflow)
      std::string fileOut;       // output file (legacy)

      int logging;               //defice logging level

};

ProgrammeParameters::ProgrammeParameters()
{

      maxProcNum = 0;
      numProcParams    = 0;
      chromosomeLength = 0;
      popSize  = 0;
      solverMaxIter = 0;
      logging     = 0;
};


ProgrammeParameters::~ProgrammeParameters()
{
};


// Goal:   read the programme-parameter file.
// Input:  progParFile - path; error - optional out-param for a failure reason.
// Output: true on success; false (with *error set) if the file cannot be
//         opened or is malformed/short.
bool ProgrammeParameters::ReadProgrammeParam(const char* progParFile, std::string* error)
{
    ifstream inputfile;
    inputfile.open(progParFile, ios::in);
    if (!inputfile.is_open()) {
        if (error) *error = std::string("cannot open programme file: ") + progParFile;
        return false;
    }

    inputfile >> maxProcNum >> numProcParams >> chromosomeLength >> popSize
              >> solverMaxIter >> fileLP >> fileOut >> logging;
    if (inputfile.fail()) {
        if (error) *error = std::string("malformed programme file: ") + progParFile;
        return false;
    }

    inputfile.close();
    return true;
}

int ProgrammeParameters::PrintProgrammeParam()
{

      cout << endl;
      cout << "Programme parameters:" << endl;
      cout << "=====================" << endl;
      cout << "Max number of processors: " << maxProcNum << endl;
      cout << "Number of processor parameters: " << numProcParams << endl;
      cout << "Max number of chromosome length: " << chromosomeLength << endl;
      cout << "Max size of population: " << popSize << endl;
      cout << "Max number of solver iteration: " << solverMaxIter << endl;
      cout << "LP file name: " << fileLP << endl;
      cout << "Out file name: " << fileOut << endl;
      cout << "logging level: " << logging << endl;
      cout << endl;

   return 1;
};


//-----------------------Processors parameters--------------------------------
class ProcessorsParameters
{
   public:
      ProcessorsParameters();
      ~ProcessorsParameters();
      bool ReadProcessorsParam(ProgrammeParameters*, const char*, std::string* error = nullptr);
      int PrintProcessorsParam(ProgrammeParameters*);

      long     volume;
      int      procNum;
      int      allocRows;       // number of rows allocated in procParams
      double** procParams;


};

ProcessorsParameters::ProcessorsParameters()
{
      volume = 0;
      procNum = 0;
      allocRows = 0;
      procParams = NULL;
}

ProcessorsParameters::~ProcessorsParameters()
{
        if (procParams)
        {
            for (int i = 0; i < allocRows; i++)
                delete[] procParams[i];
            delete[] procParams;
        }
}


// Goal:   read the processor-parameter file (volume line + one row of
//         S C A B per processor; lines starting with '#' are comments).
// Input:  progPar - supplies maxProcNum / numProcParams; procParFile - path;
//         error - optional out-param for a failure reason.
// Output: true on success; false (with *error set) if the file cannot be
//         opened, the header counts are invalid, or a row is malformed.
bool ProcessorsParameters::ReadProcessorsParam(ProgrammeParameters* progPar, const char* procParFile, std::string* error)
{
    if (progPar->maxProcNum <= 0 || progPar->numProcParams < 4) {
        if (error) *error = "invalid header counts (need maxProcNum>0, numProcParams>=4)";
        return false;
    }

    ifstream inputfile;
    inputfile.open(procParFile, ios::in);
    if (!inputfile.is_open()) {
        if (error) *error = std::string("cannot open processor file: ") + procParFile;
        return false;
    }

    std::string line;                              // current input line
    std::getline(inputfile, line);
    if (!line.empty() && line[0] != __COMMENT) {
        std::istringstream(line) >> volume;        // first non-comment line: total load V
    }

    allocRows = progPar->maxProcNum + 1;           // row 0 unused (1-based)
    procParams = new double* [allocRows];
    for (int r = 0; r < allocRows; ++r)
        procParams[r] = new double [progPar->numProcParams];

    int i = 1;                                     // next processor row (1-based)
    while (std::getline(inputfile, line) && i <= progPar->maxProcNum) {
        if (line.empty() || line[0] == __COMMENT) continue;
        std::istringstream ls(line);               // safe tokenised parse (no fixed buffers)
        double s, c, a, b;
        if (!(ls >> s >> c >> a >> b)) {
            if (error) *error = std::string("malformed processor row: ") + line;
            inputfile.close();
            return false;
        }
        procParams[i][PROC_COMM_STARTUP] = s;
        procParams[i][PROC_COMM_RATE]    = c;
        procParams[i][PROC_COMPUTE_RATE] = a;
        procParams[i][PROC_MEMORY_LIMIT] = b;
        ++i;
    }

    procNum = i - 1;
    inputfile.close();

    if (procNum == 0) {
        if (error) *error = "processor file contained no processor rows";
        return false;
    }
    return true;
}

int ProcessorsParameters::PrintProcessorsParam(ProgrammeParameters* progPar)
{

    int i,j;

        cout << endl;
        cout << "Processor parameters:" << endl;
        cout << "=====================" << endl;
        cout << "volume: " << volume << endl;
        cout << "Number of processor: " << procNum << endl;

        for(i=1; i < procNum+1; i++)
        {
           for(j=0; j < progPar->numProcParams; j++)
           {
              cout << procParams[i][j] << "\t";
           }
           cout << endl;
        };


   return 1;
};


//---------------------------------------------------------------------------
class GeneticAlgorithmParameters
{
  public:
   GeneticAlgorithmParameters();
   ~GeneticAlgorithmParameters();
   bool ReadGAParam(const char*, std::string* error = nullptr);
   int PrintGAParam();

   int  maxIterNum;             //Max number of iterations
   int  improvementNum;         //Number of iterations without improvements
   int  populationSize;         //Size of population
   int  chromosomeGeneration;   //Way of chromosome generation; 0- constant length | 1- various length
   int  chromosomeMinSize;      //Minimum size of chromosome
   int  chromosomeMaxSize;      //Maximum size of chromosome
   int  selectionMethod;        //Method of selection  0 - roulette; 1 - tournament
   int  tourGroupSize;          //Size of tournament group
   int  crossoverPoint;         //Crossover method; 0 - constant; 1 - various
   double  crossoverProb;       //Probability of cross over
   int  mutationMethod;         //Method of mutation; 0 - random; 1 - every n gen
   double  mutationProb;        //Mutation probability

};

GeneticAlgorithmParameters::GeneticAlgorithmParameters()
{
//set up parameters
   maxIterNum           = 0;
   improvementNum       = 0;
   populationSize       = 0;
   chromosomeGeneration = 0;
   chromosomeMinSize    = 0;
   chromosomeMaxSize    = 0;
   selectionMethod      = 0;
   tourGroupSize        = 0;
   crossoverPoint       = 0;
   crossoverProb        = 0;
   mutationMethod       = 0;
   mutationProb         = 0;

};

GeneticAlgorithmParameters::~GeneticAlgorithmParameters()
{

};


// Goal:   read the 12-value GA-parameter file.
// Input:  GaAlgFile - path; error - optional out-param for a failure reason.
// Output: true on success; false (with *error set) on open/parse failure.
bool GeneticAlgorithmParameters::ReadGAParam(const char* GaAlgFile, std::string* error)
{
   ifstream inputfile;
   inputfile.open(GaAlgFile, ios::in);
   if (!inputfile.is_open()) {
      if (error) *error = std::string("cannot open GA parameter file: ") + GaAlgFile;
      return false;
   }



   inputfile >> maxIterNum >> improvementNum >> populationSize
             >> chromosomeGeneration >> chromosomeMinSize >> chromosomeMaxSize
             >> selectionMethod >> tourGroupSize >> crossoverPoint
             >> crossoverProb >> mutationMethod >> mutationProb;
   if (inputfile.fail()) {
      if (error) *error = std::string("malformed GA parameter file: ") + GaAlgFile;
      return false;
   }



/*
   while (!ImportEnd)
       {
          if(inputfile.eof())
          {
            ImportEnd = 1;
            continue;
          };




   maxIterNum           = 1000;
   improvementNum       = 100;
   populationSize       = 5;
   chromosomeGeneration = 0;
   chromosomeMinSize    = 1;
   chromosomeMaxSize    = 5;
   selectionMethod      = 0;
   tourGroupSize        = 0;
   crossoverPoint       = 0;
   crossoverProb        = 3;
   mutationMethod       = 0;
   mutationProb         = 1;
*/

   inputfile.close();

   return true;
};

int GeneticAlgorithmParameters::PrintGAParam()
{
   cout << endl;
   cout << "Genetic algorithm parameters:" << endl;
   cout << "=============================" << endl;
   cout << "Max number of iterations: " << maxIterNum           << endl;
   cout << "Max number of iterations without improvement: " << improvementNum       << endl;
   cout << "population size: " << populationSize       << endl;
   cout << "Way of chromosome generation: " << chromosomeGeneration << endl;
   cout << "Min size of chromosome: " << chromosomeMinSize    << endl;
   cout << "Max size of chromosome: " << chromosomeMaxSize    << endl;
   cout << "Selection method: " << selectionMethod      << endl;
   cout << "Size of tournament group: " << tourGroupSize        << endl;
   cout << "Crossover point: " << crossoverPoint       << endl;
   cout << "Crossover probability: " << crossoverProb        << endl;
   cout << "Mutation method: " << mutationMethod       << endl;
   cout << "Mutation probability: " << mutationProb         << endl;
   cout << endl;

  return 1;
}



//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
//---------------------------------------------------------------------------
// chromosome - code string
// A candidate solution is encoded as a sequence of genes (installments).
// The former hand-rolled singly-linked lists (Chromosome/GenotypesList) were
// replaced with std::vector value types: copying is '=', destruction is
// automatic (RAII), and the old manual copy/delete helpers and duplicated
// save logic are gone.

struct Gene {                  // one installment of the schedule
    int    processorNum = 0;   // 1-based processor id (GA convention)
    double dataSize     = 0.0; // load assigned to it (filled by the evaluator)
};

struct Genotype {              // one candidate solution ("chromosome")
    double            cmax = 0.0; // makespan; 0 means unscored / infeasible
    std::vector<Gene> genes;      // the activation sequence
};

//---------------------------------------------------------------------------
// A generation-based population of genotypes plus the best solution found.
class Population
{
  public:
    int GenerateNewPopulation(ProcessorsParameters*, GeneticAlgorithmParameters*);
    int EvaluatePopulation(ProgrammeParameters*, ProcessorsParameters*, GeneticAlgorithmParameters*);
    int NewToOldCopy();
    int CheckImprovement();
    int Select(ProcessorsParameters*, GeneticAlgorithmParameters*);
    int RouletteSelection(ProcessorsParameters*, GeneticAlgorithmParameters*);
    int TournamentSelection(ProcessorsParameters*, GeneticAlgorithmParameters*);
    int CrossOver(ProcessorsParameters*, GeneticAlgorithmParameters*);
    int Mutation(ProcessorsParameters*, GeneticAlgorithmParameters*);

    std::vector<Genotype> oldPop;   // previous generation (selection source)
    std::vector<Genotype> newPop;   // current generation being built/scored
    Genotype best;                  // best genotype found so far
    bool      haveBest = false;     // whether 'best' holds a real solution
    double    bestCmax = 0.0;       // makespan of 'best'

    // Fitness backend, set by GASolver before the loop (contract unchanged).
    const dls::DLSInstance*       dlsInstance = nullptr;       // problem being solved
    const dls::ScheduleEvaluator* scheduleEvaluator = nullptr; // LP-backed evaluator
    dls::EvaluatorConfig          evaluatorConfig;             // LP runtime knobs

    // Injectable RNG, set by GASolver (replaces global rand()): reproducible
    // from a seed and thread-safe (each solve owns its own generator).
    util::Rng* rng = nullptr;

  private:
    // Goal:   order genotypes best-first: a feasible (positive cmax) genotype
    //         beats an infeasible one (cmax == 0); among feasible, smaller is
    //         better (we minimise the makespan).
    // Input:  a, b - the genotypes to compare.
    // Output: true if a is strictly better than b.
    static bool betterThan(const Genotype& a, const Genotype& b) {
        if (a.cmax <= 0.0) return false;   // a infeasible -> never "better"
        if (b.cmax <= 0.0) return true;    // b infeasible, a feasible
        return a.cmax < b.cmax;
    }
};

//---------------------------------------------------------------------------
// Goal:   create the initial random population in newPop.
// Input:  procPar - processor count; gaAlgPar - population/chromosome sizes.
// Output: 1; newPop holds populationSize random genotypes.
int Population::GenerateNewPopulation(ProcessorsParameters* procPar, GeneticAlgorithmParameters* gaAlgPar)
{
    const int n = gaAlgPar->populationSize;      // genotypes to create
    const int L = gaAlgPar->chromosomeMaxSize;   // genes per genotype
    newPop.assign(n, Genotype{});
    for (int g = 0; g < n; ++g) {
        newPop[g].cmax = 0.0;
        newPop[g].genes.resize(L);
        for (int j = 0; j < L; ++j)
            newPop[g].genes[j].processorNum = rng->index(procPar->procNum) + 1;
    }
    return 1;
}

//---------------------------------------------------------------------------
// Goal:   score every genotype in newPop via the core ScheduleEvaluator.
// Input:  legacy params unused (kept for signature compatibility); the
//         instance + evaluator come from the members set by GASolver.
// Output: 1 on success, 0 if no evaluator/instance is wired in. Each genotype
//         gets cmax set and its gene dataSizes filled.
int Population::EvaluatePopulation(ProgrammeParameters* progPar, ProcessorsParameters* procParam, GeneticAlgorithmParameters* gaAlgPar)
{
    (void)progPar; (void)procParam; (void)gaAlgPar;
    if (!scheduleEvaluator || !dlsInstance) return 0;

    for (Genotype& g : newPop) {
        // Skip genotypes already scored: cmax != 0 means unchanged since its
        // last evaluation (crossover/mutation reset cmax to 0 on change), so
        // its makespan and gene dataSizes are still valid.
        if (g.cmax != 0.0) continue;

        // Build the activation sequence (GA is 1-based; the core is 0-based).
        std::vector<int> sequence;               // installment order
        sequence.reserve(g.genes.size());
        for (const Gene& gene : g.genes)
            sequence.push_back(gene.processorNum - 1);

        dls::DLSSolution sol = scheduleEvaluator->evaluate(*dlsInstance, sequence, evaluatorConfig);

        if (sol.status == dls::SolveStatus::Optimal) {
            g.cmax = sol.makespan;
            for (size_t i = 0; i < g.genes.size() && i < sol.fragments.size(); ++i)
                g.genes[i].dataSize = sol.fragments[i].loadSize;
        } else {
            g.cmax = 0.0;                        // infeasible/failure -> 0 (legacy)
        }
    }
    return 1;
}

//---------------------------------------------------------------------------
// Goal:   hand the current generation to oldPop so selection can rebuild newPop.
// Input:  none.
// Output: 1; oldPop = previous newPop, newPop emptied.
int Population::NewToOldCopy()
{
    oldPop = std::move(newPop);
    newPop.clear();
    return 1;
}

//---------------------------------------------------------------------------
// Goal:   record the best solution and report whether this generation improved.
// Input:  none (reads newPop).
// Output: 1 if newPop's best feasible makespan strictly beats the stored best
//         (at __PRECISION), else 0. Updates best/bestCmax/haveBest.
int Population::CheckImprovement()
{
    int    bestIdx = -1;   // index of the best feasible genotype this generation
    double minPos  = 0.0;  // its makespan
    for (int i = 0; i < (int)newPop.size(); ++i) {
        double c = newPop[i].cmax;
        if (c > 0.0 && (bestIdx < 0 || c < minPos)) { minPos = c; bestIdx = i; }
    }
    if (bestIdx < 0) return 0;          // no feasible genotype this generation

    if (!haveBest) {                    // first feasible solution: record, not an "improvement"
        best     = newPop[bestIdx];
        bestCmax = minPos;
        haveBest = true;
        return 0;
    }
    if (floor(minPos * __PRECISION) < floor(bestCmax * __PRECISION)) {
        best     = newPop[bestIdx];
        bestCmax = minPos;
        return 1;
    }
    return 0;
}

//---------------------------------------------------------------------------
// Goal:   build newPop from oldPop using the configured selection strategy.
// Input:  procPar - forwarded; gaAlgPar - selectionMethod (0 roulette, 1 tournament).
// Output: 1; newPop filled with populationSize selected genotypes.
int Population::Select(ProcessorsParameters* procPar, GeneticAlgorithmParameters* gaAlgPar)
{
    if (gaAlgPar->selectionMethod == 1)
        return TournamentSelection(procPar, gaAlgPar);
    return RouletteSelection(procPar, gaAlgPar);
}

//---------------------------------------------------------------------------
// Goal:   fitness-proportionate selection with elitism. The best half of
//         oldPop is carried over; the rest are drawn on a roulette wheel where
//         a smaller makespan gets a larger share.
// Input:  procPar - unused; gaAlgPar - populationSize.
// Output: 1; newPop filled.
int Population::RouletteSelection(ProcessorsParameters* procPar, GeneticAlgorithmParameters* gaAlgPar)
{
    (void)procPar; (void)gaAlgPar;
    const int n = (int)oldPop.size();
    if (n == 0) return 1;

    // Order best-first (smaller positive cmax better; infeasible last).
    std::sort(oldPop.begin(), oldPop.end(), betterThan);

    newPop.clear();
    newPop.reserve(n);

    const int elite = n / 2;                 // elitist half carried over
    for (int i = 0; i < elite; ++i)
        newPop.push_back(oldPop[i]);

    // Raw proportions: smaller cmax -> larger share (minimisation).
    double cmaxSum = 0.0;                     // sum of all makespans
    for (const Genotype& g : oldPop) cmaxSum += g.cmax;

    std::vector<double> proportion(n, 0.0);   // per-genotype share
    for (int i = 0; i < n; ++i)
        proportion[i] = (oldPop[i].cmax != 0.0) ? cmaxSum / oldPop[i].cmax : 0.0;

    double propSum = 0.0;                     // normaliser
    for (int i = 0; i < n; ++i) propSum += proportion[i];
    for (int i = 0; i < n; ++i)
        proportion[i] = (propSum != 0.0) ? proportion[i] / propSum : 0.0;

    std::vector<double> wheel(n, 0.0);        // cumulative proportions in [0, total]
    double total = 0.0;
    for (int i = 0; i < n; ++i) { total += proportion[i]; wheel[i] = total; }

    for (int s = 0; s < n - elite; ++s) {     // fill the remaining slots
        double u = (total > 0.0) ? rng->uniform01() * total : rng->uniform01() * n;  // wheel draw
        int k = n - 1;                        // selected index (default last bin)
        if (total > 0.0) {
            for (int j = 0; j < n; ++j) if (wheel[j] > u) { k = j; break; }
        } else {
            k = (int)u; if (k >= n) k = n - 1;
        }
        newPop.push_back(oldPop[k]);
    }
    return 1;
}

//---------------------------------------------------------------------------
// Goal:   tournament selection — each of populationSize slots is the best of
//         tourGroupSize random contenders from oldPop.
// Input:  procPar - unused; gaAlgPar - populationSize, tourGroupSize.
// Output: 1; newPop filled.
int Population::TournamentSelection(ProcessorsParameters* procPar, GeneticAlgorithmParameters* gaAlgPar)
{
    (void)procPar;
    const int n = (int)oldPop.size();
    if (n == 0) return 1;
    int groupSize = gaAlgPar->tourGroupSize;  // contenders per tournament
    if (groupSize < 1) groupSize = 2;
    if (groupSize > n) groupSize = n;

    newPop.clear();
    newPop.reserve(gaAlgPar->populationSize);
    for (int s = 0; s < gaAlgPar->populationSize; ++s) {
        int bestIdx = rng->index(n);             // first contender
        for (int g = 1; g < groupSize; ++g) {
            int cand = rng->index(n);            // next contender
            if (betterThan(oldPop[cand], oldPop[bestIdx])) bestIdx = cand;
        }
        newPop.push_back(oldPop[bestIdx]);
    }
    return 1;
}

//---------------------------------------------------------------------------
// Goal:   recombine genotypes in newPop. Each genotype crosses (with the
//         crossover probability) with a random distinct mate, using single- or
//         two-point crossover per gaAlgPar->crossoverPoint.
// Input:  procPar - unused; gaAlgPar - crossover prob/strategy, populationSize.
// Output: 1; crossed genotypes have their cmax reset to 0 (need re-scoring).
int Population::CrossOver(ProcessorsParameters* procPar, GeneticAlgorithmParameters* gaAlgPar)
{
    (void)procPar;
    const int n = (int)newPop.size();
    for (int gc = 0; gc < n; ++gc) {
        const int L = (int)newPop[gc].genes.size();   // chromosome length
        if (gaAlgPar->populationSize > 1 && L >= 2
            && rng->bernoulli(gaAlgPar->crossoverProb, gaAlgPar->populationSize)) {
            int mate;                                  // partner genotype index
            do { mate = rng->index(n); } while (mate == gc);

            std::vector<Gene>& a = newPop[gc].genes;
            std::vector<Gene>& b = newPop[mate].genes;

            if (gaAlgPar->crossoverPoint == 1 && L >= 3) {
                // Two-point: swap the middle gene segment [p1, p2).
                int p1 = rng->index(L - 2) + 1;             // first cut in [1, L-2]
                int p2 = rng->index(L - 1 - p1) + p1 + 1;   // second cut in [p1+1, L-1]
                std::swap_ranges(a.begin() + p1, a.begin() + p2, b.begin() + p1);
            } else {
                // Single-point: swap the tails at one cut.
                int cp = (L == 2) ? 1 : rng->index(L - 2) + 1;
                std::swap_ranges(a.begin() + cp, a.end(), b.begin() + cp);
            }
            newPop[gc].cmax   = 0.0;
            newPop[mate].cmax = 0.0;
        }
    }
    return 1;
}

//---------------------------------------------------------------------------
// Goal:   mutate genes in newPop. Per gene, with the mutation probability,
//         either reassign it to a random processor (PerGene) or swap it with a
//         random other gene (Swap), per gaAlgPar->mutationMethod.
// Input:  procPar - processor count; gaAlgPar - mutation prob/strategy.
// Output: 1; mutated genotypes have their cmax reset to 0.
int Population::Mutation(ProcessorsParameters* procPar, GeneticAlgorithmParameters* gaAlgPar)
{
    for (Genotype& g : newPop) {
        const int L = (int)g.genes.size();          // genes in this chromosome
        for (int j = 0; j < L; ++j) {
            if (rng->bernoulli(gaAlgPar->mutationProb, gaAlgPar->populationSize * L)) {
                if (gaAlgPar->mutationMethod == 1 && L >= 2) {
                    // Swap mutation: exchange this gene's processor with another.
                    int other = rng->index(L);          // partner gene
                    std::swap(g.genes[j].processorNum, g.genes[other].processorNum);
                } else {
                    // Per-gene reset: reassign to a random processor.
                    g.genes[j].processorNum = rng->index(procPar->procNum) + 1;
                }
                g.cmax = 0.0;
            }
        }
    }
    return 1;
}

//---------------------------------------------------------------------------
#endif
