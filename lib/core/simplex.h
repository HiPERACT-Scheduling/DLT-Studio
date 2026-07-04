//---------------------------------------------------------------------------
// core/simplex.h
//
// A small dense Simplex LP solver used as the backend of the DLS schedule
// evaluator. It minimises a linear objective subject to "<=" constraints.
// Equalities and ">=" constraints are expressed via AddConstraintLE (see
// that method). This is generic infrastructure: nothing here knows about
// divisible load scheduling.
//
// The public surface (class name CSimplex, its public methods, and the
// OPTIMAL/UNBOUNDED/INFEASIBLE/FAILURE result codes) is depended on by
// simplex_schedule_evaluator.hpp and must stay stable. Comments were
// translated from the original Polish; internal identifiers were renamed
// from Hungarian to descriptive English names. The algorithm is unchanged.
//---------------------------------------------------------------------------

#ifndef DLS_CORE_SIMPLEX_H
#define DLS_CORE_SIMPLEX_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>

#define MAXREAL     1e100       // a "very large" real used as +infinity
#define OPTIMAL         0       // solve result: optimum found
#define UNBOUNDED       1       // solve result: objective is unbounded
#define INFEASIBLE      2       // solve result: no feasible solution exists
#define FAILURE         3       // solve result: computation failed / no convergence

#define EPSILON      1e-5       // numerical round-off tolerance

#define DEFAULTITER  1000       // default cap on the number of iterations


class CSimplex
{
private:
    int maxIterations;          // cap on the number of Simplex iterations
    int numVars;                // number of decision variables
    int numConstraints;         // number of constraints the problem will hold
    int addedConstraints;       // how many constraints have been added so far
    int* basisVars;             // index of the basic variable in each row
    double* objective;          // objective-function coefficients
    double** table;             // the Simplex tableau
    double** tableSnapshot;     // copy of the tableau, kept for WriteLPTask
    double** tableNext;         // tableau being computed for the next iteration
    double* solution;           // result: [0] = objective value, [i>0] = variable i
    std::mt19937 rng;           // owned RNG for anti-cycling (no global rand())

    // Goal:   draw a uniform integer in [0, n) from the owned RNG.
    // Input:  n - exclusive upper bound (n > 0).
    // Output: an int in [0, n).
    int randIndex(int n) { return static_cast<int>(rng() % static_cast<unsigned>(n)); }

    // Goal:   report a fatal error and abort the process.
    // Input:  message - the text to print.
    // Output: none (does not return; calls exit(1)).
    void Error(const char *message)
    {
        printf("\n%s\n", message);
        exit(1);

        return;
    }

public:
    // Goal:   allocate a solver sized for a given problem.
    // Input:  numberOfVars        - number of decision variables;
    //         numberOfConstraints - number of constraints to be added later;
    //         seed                - seed for the anti-cycling RNG (deterministic).
    // Output: a ready-to-fill solver (use SetObjectiveFunction + AddConstraintLE).
    CSimplex(int numberOfVars, int numberOfConstraints, std::uint64_t seed = 0)
        : rng(seed)
    {
        maxIterations    = DEFAULTITER;
        addedConstraints = 0;
        numVars          = numberOfVars;
        numConstraints   = numberOfConstraints;
        solution      = new double[numVars + 1];
        basisVars     = new int[numConstraints];
        objective     = new double[numVars];
        table         = new double*[numConstraints + 1];
        tableSnapshot = new double*[numConstraints + 1];
        tableNext     = new double*[numConstraints + 1];
        for (int i = 0; i < numConstraints + 1; i++)
        {
            table[i]         = new double[numVars + numConstraints + 1];
            tableSnapshot[i] = new double[numVars + numConstraints + 1];
            tableNext[i]     = new double[numVars + numConstraints + 1];
        }
    }

    // Goal:   release all memory owned by the solver.
    // Input:  none.
    // Output: none.
    ~CSimplex()
    {
        delete[] solution;
        delete[] basisVars;
        delete[] objective;
        for (int i = 0; i < numConstraints + 1; i++)
        {
            delete[] table[i];
            delete[] tableSnapshot[i];
            delete[] tableNext[i];
        }
        delete[] table;
        delete[] tableSnapshot;
        delete[] tableNext;
    }

    // Goal:   forget the constraints added so far so the solver can be reused.
    // Input:  none.
    // Output: none (resets the added-constraint counter).
    void Clear()
    {
        addedConstraints = 0;
    }

    // Goal:   add one constraint of the form  g(x1,..,xn) <= rhs.
    //         g(x1,..,xn) >= rhs  ->  -g(x1,..,xn) <= -rhs
    //         g(x1,..,xn) =  rhs  ->  add both  g <= rhs  and  -g <= -rhs
    // Input:  constraint - array of numVars coefficients;
    //         rhs        - the right-hand-side bound.
    // Output: none (appends a row and adds its slack variable).
    void AddConstraintLE(double *constraint, double rhs)
    {
        table[addedConstraints][0]         = rhs;
        tableSnapshot[addedConstraints][0] = rhs;
        for (int i = 1; i < numVars + numConstraints + 1; i++)
            if (i <= numVars)
            {
                table[addedConstraints][i]         = constraint[i - 1];
                tableSnapshot[addedConstraints][i] = constraint[i - 1];
            }
            else
                if (i == numVars + addedConstraints + 1)
                {
                    table[addedConstraints][i]         = 1;   // this row's slack
                    tableSnapshot[addedConstraints][i] = 1;
                }
                else
                {
                    table[addedConstraints][i]         = 0;
                    tableSnapshot[addedConstraints][i] = 0;
                }
        addedConstraints++;

        return;
    }

    // Goal:   set the objective-function coefficients (minimised).
    // Input:  objectiveCoeffs - array of numVars coefficients.
    // Output: none.
    void SetObjectiveFunction(double* objectiveCoeffs)
    {
        for (int i = 0; i < numVars; i++)
            objective[i] = objectiveCoeffs[i];

        return;
    }

    // Goal:   limit how many Simplex iterations Solve() may run.
    // Input:  maxIter - the iteration cap.
    // Output: none.
    void SetMaxNumberOfIter(int maxIter)
    {
        maxIterations = maxIter;

        return;
    }

    // Goal:   solve the LP defined by the objective and added constraints.
    // Input:  none (uses the previously supplied objective/constraints).
    // Output: OPTIMAL, UNBOUNDED, INFEASIBLE or FAILURE; on OPTIMAL the
    //         result is retrievable via GetSolution().
    int Solve()
    {
        // Prepare the tableau: slack variables form the initial basis.
        for (int i = 0; i < numConstraints; i++)
            basisVars[i] = numVars + i + 1;

        if (addedConstraints != numConstraints)
            Error("Wrong number of constraints");

        table[addedConstraints][0] = 0;
        for (int i = 1; i < numVars + numConstraints + 1; i++)
            if (i <= numVars)
                table[addedConstraints][i] = -objective[i - 1];
            else
                table[addedConstraints][i] = 0;

        // SIMPLEX
        int i, j, k, l;
        int iter, jPivot, iPivot, rhsAllNonNeg;
        double theta;                 // ratio-test value of the chosen pivot
        iter = 0;
        for (;;)
        {
            // Enforce the iteration cap.
            iter++;
            if (iter > maxIterations)
                return FAILURE;

            // Are all RHS values (Xi0) non-negative?
            rhsAllNonNeg = 1;
            for (i = 0; i < numConstraints; i++)
                if (table[i][0] < 0)
                {
                    rhsAllNonNeg = 0;
                    break;
                }

            if (rhsAllNonNeg == 1)     // Xi0 >= 0: optimise the objective row
            {
                double maxReducedCost;    // best (largest) reduced cost found

                // Find max{Zj - Cj}; the entering variable column is k.
                maxReducedCost = -MAXREAL;
                for (j = 1; j < numVars + numConstraints + 1; j++)
                {
                    if (maxReducedCost < table[numConstraints][j])
                    {
                        k = j;
                        maxReducedCost = table[numConstraints][j];
                    }
                }

                if (maxReducedCost <= 0) break;   // Zj - Cj <= 0: optimum reached

                // Choose the leaving row by the minimum-ratio test.
                theta = MAXREAL;
                iPivot = randIndex(numConstraints);          // random start: anti-cycling
                for (i = 0; i < numConstraints; i++)
                {
                    if (table[iPivot][k] > 0)
                        if (theta > table[iPivot][0] / table[iPivot][k])
                        {
                            l = iPivot;
                            theta = table[iPivot][0] / table[iPivot][k];
                        }
                    if (++iPivot == numConstraints) iPivot = 0;
                }
                if (theta == MAXREAL)
                    return UNBOUNDED;                        // problem is unbounded
            }
            else
            {
                double bestImprovement, minRhs, improvement;

                // Find min{Xi0}: the most-infeasible row becomes the pivot row.
                minRhs = MAXREAL;
                for (i = 0; i < numConstraints; i++)
                    if (table[i][0] < minRhs)
                    {
                        l = i;
                        minRhs = table[i][0];
                    }

                // Pick the entering column giving the best objective improvement.
                bestImprovement = -MAXREAL;
                theta = -MAXREAL;
                jPivot = randIndex(numVars + numConstraints) + 1;   // random start: anti-cycling
                for (j = 1; j < numVars + numConstraints + 1; j++)
                {
                    if (table[l][jPivot] < 0)
                    {
                        improvement = table[numConstraints][jPivot] * table[l][0] / table[l][jPivot];
                        if (bestImprovement < improvement)
                        {
                            k = jPivot;
                            theta = table[l][0] / table[l][jPivot];
                            bestImprovement = theta * table[numConstraints][jPivot];
                        }
                    }
                    if (++jPivot == numVars + numConstraints + 1) jPivot = 1;
                }

                if (bestImprovement == -MAXREAL)
                    return INFEASIBLE;   // cannot remove the variable from the basis (Xi0 < 0 and theta < 0)
            }

            // Constraint checks.
            if (theta < 0) return UNBOUNDED;
            if (theta >= MAXREAL) return INFEASIBLE;

            //printf("%2d) ",iter);
            //printf("Theta: %10g  ", theta);
            //printf("Pivot: %10g  ", table[l][k]);
            //printf("Rhs: %10g  ",table[l][0]);
            //printf("Obj: %10g\n", table[numConstraints][0]);

            // Bring the entering variable into the basis on row l.
            basisVars[l] = k;

            // Recompute the tableau (Gauss-Jordan pivot) with round-off control.
            double prod1, prod2, pivoted;
            for (i = 0; i < numConstraints + 1; i++)
            {
                for (j = 0; j < numVars + numConstraints + 1; j++)
                    if (i != l)
                    {
                        prod1 = table[i][j] * table[l][k];
                        prod2 = table[l][j] * table[i][k];
                        pivoted = (prod1 - prod2) / table[l][k];
                        if (fabs(pivoted) < EPSILON)
                            tableNext[i][j] = 0;
                        else
                            tableNext[i][j] = pivoted;
                    }
                    else
                        tableNext[i][j] = table[i][j] / table[l][k];
            }

            // Copy the new tableau back into the working tableau.
            for (i = 0; i < numConstraints + 1; i++)
                for (j = 0; j < numVars + numConstraints + 1; j++)
                    table[i][j] = tableNext[i][j];
        }

        // Extract the solution.
        for (i = 0; i < numVars + 1; i++) solution[i] = 0;
        solution[0] = table[numConstraints][0];
        for (i = 0; i < numConstraints; i++)
            if (basisVars[i] <= numVars)
                solution[basisVars[i]] = table[i][0];

        return OPTIMAL;
    }


    // Goal:   read back a value after a successful Solve().
    // Input:  i - 0 for the objective value, 1..numVars for variable i.
    // Output: the requested value (aborts via Error on out-of-range index).
    double GetSolution(int i)
    {
        if (i < 0 || i > numVars)
            Error("Range check error");
        return solution[i];
    }


    // Goal:   write the LP (objective + constraints) to a text file for
    //         inspection; call before Solve() so the tableau is unmodified.
    // Input:  fileName - path of the output file.
    // Output: none (writes the file).
    void WriteLPTask(char *fileName)
    {
        int i, j;
        FILE* fp = fopen(fileName, "w+");

        fprintf(fp, "min:");
        for (i = 0; i < numVars; i++)
            if (objective[i] > 0)
                fprintf(fp, " +%gx%d", objective[i], i + 1);
            else
                if (objective[i] < 0)
                    fprintf(fp, " %gx%d", objective[i], i + 1);
        fprintf(fp, ";\n");
        for (i = 0; i < numConstraints; i++)
        {
            for (j = 1; j < numVars + 1; j++)
                if (tableSnapshot[i][j] > 0)
                    fprintf(fp, " +%gx%d", tableSnapshot[i][j], j);
                else
                    if (tableSnapshot[i][j] < 0)
                        fprintf(fp, " %gx%d", tableSnapshot[i][j], j);
            fprintf(fp, " <= %g;\n", tableSnapshot[i][0]);
        }
        fprintf(fp, "\n------------------------------------------------------\n");
        fclose(fp);
        return;
    }
};

#endif  // DLS_CORE_SIMPLEX_H
