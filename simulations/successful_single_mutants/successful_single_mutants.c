/*
 * successful_single_mutants.c
 *
 * Created by Michael McLaren (mmclaren@stanford.edu) on Sat Nov 8 2014.
 *
 * Simulate new single-mutant lineages with secondary mutation to estimate the
 * probability of being successful $p_1$ and the drift time $T_1$.
 * Spatial structure can follow the island model or the 1- or 2-dimensional
 * stepping stone model.
 *
 * Compile with:
 * gcc -c successful_single_mutants.c; gcc successful_single_mutants.o -lgsl -lgslcblas -lm -o successful_single_mutants
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_randist.h>

#define ZEPS 0.0001
#define NEPS 0.01
// Number of genotypes
#define GTYPES 3

// Global variables
const gsl_rng_type * T;
gsl_rng * R;

void next_gen(unsigned int L1, unsigned int L2, unsigned int N, 
        double w[GTYPES], double mu, double m_1, double m_2, double m_inf, 
        unsigned int n[L1][L2][GTYPES], unsigned int ntot[GTYPES]) {
    /* Parameters:  
     * n[L1][L2][GTYPES] -- counts of genotypes in deme (i,j)
     * ntot[GTYPES] -- counts of genotypes in the total population
     * w[GTYPES] -- fitnesses of the different genotypes
     * mu -- mutation rate from genotype 1 to 2
     * m_1 -- migration rate across rows
     * m_2 -- migration rate across columns
     * m_inf -- global migration rate
     * L1 -- number of rows
     * L1 -- number of columns
     * N -- local population size
     *
     * Order: Mutation, migration, selection, sampling.
     */
    // Frequencies in deme (i,j) after mut
    double x[L1][L2][GTYPES]; 
    // In a single deme, frequencies at start of generation
    double x0[GTYPES]; 
    // Total frequencies in population
    double xtot[GTYPES] = {0}; 
    // Frequencies in deme (i,j) after migration
    double xmig[L1][L2][GTYPES]; 
    // within deme frequencies after selection
    double X[GTYPES]; 
    // total number of demes
    double L = L1 * L2; 
    int i, j, k;
    // Reset total genotype counts to 0
    for (k = 0; k < GTYPES; k += 1) {
        ntot[k] = 0;
    }
    // Begin life cycle
    // Mutation; Results stored in x[][][]
    for (i = 0; i < L1; i += 1) {
        for (j = 0; j < L2; j += 1) {
            // Calculate haplotype frequencies in deme i,j
            x0[0] = (double)n[i][j][0] / N;
            x0[1] = (double)n[i][j][1] / N;
            x0[2] = (double)n[i][j][2] / N;
            // Mutation from type 1 to type 2
            x[i][j][0] = x0[0];
            x[i][j][1] = x0[1] * (1 - mu);
            x[i][j][2] = x0[2] + x0[1] * mu; 
            // Also keep track of population wide totals
            for (k = 0; k < GTYPES; k += 1) {
                xtot[k] += x[i][j][k];
            }
        }
    }
    // Average frequencies for the population
    // xtot[k] = average of x[i][j][k] over all demes (i,j)
    for (k = 0; k < GTYPES; k += 1) {
        xtot[k] /= L;
    }
    // Migration; results stored in xmig[][][]
    for (i = 0; i < L1; i += 1) {
        for (j = 0; j < L2; j += 1) {
            // Migration; wrap-around boundaries
            for (k = 0; k < 4; k += 1) {
                xmig[i][j][k] = (1 - m_1 - m_2 - m_inf) * x[i][j][k] 
                    + 0.5 * m_1 * (x[(i-1+L1)%L1][j][k] + x[(i+1)%L1][j][k])
                    + 0.5 * m_2 * (x[i][(j-1+L2)%L2][k] + x[i][(j+1)%L2][k])
                    + m_inf * xtot[k];
            }
        }
    }
    // Selection, and sampling (within demes)
    for (i = 0; i < L1; i += 1) {
        for (j = 0; j < L2; j += 1) {
            // Selection
            for (k = 0; k < GTYPES; k += 1) {
                X[k] = xmig[i][j][k] * w[k];
            }
            // Sample within each deme:
            // Normalization of X done in multinomial function, so dividing by
            // average fitness is not necessary
            gsl_ran_multinomial(R, GTYPES, N, X, n[i][j]);
            // Keep track of total genotype counts
            for (k = 0; k < GTYPES; k += 1) {
                ntot[k] += n[i][j][k];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    long SEED;
    // Number of rows, number of columns, total number of demes
    unsigned int L, L1, L2; 
    // Local population size, global population size;
    unsigned int N, Ntot; // 
    // Migration rates: between rows, between col's, and global
    double m_1, m_2, m_inf; 
    // Fitness params
    double delta, s; 
    // forward mutation from type 1 to type 2
    double mu; 
    // Fitnesses of the 3 genotypes
    double w[GTYPES];
    // Total number of runs to simulate
    unsigned int num_runs = 0;
    // Number of runs where lineage is successful
    unsigned int num_success = 0; 
    // Number of runs where lineage is still segregating 
    unsigned int num_seg = 0; 
    // Time in current run
    unsigned long int t = 0;
    // Max time allowed
    unsigned long int tfinal;
    // Times to record to out file (currently only t_first and t_fix implemented)
    unsigned long int t_1, t_est, t_fix = 0;
    // Data files
    FILE *outfile;
    FILE *datafile;
    // Iterator variables
    unsigned int run, i, j, k;

    //Initialize variables:
    if (argc != 14) {
        // Arg no.'s:  0       1  2  3 4   5 6 7  8  9    10       11
        printf("Usage: metapop L1 L2 N mu1 d s m1 m2 minf tfinal num_runs "
        //     12       13
               "ranseed outfile\n");
        return 0; 
    }
    // Read in command line arguments
    L1 = atof(argv[1]);
    L2 = atof(argv[2]);
    N = atof(argv[3]);
    mu = atof(argv[4]);
    delta = atof(argv[5]);
    s = atof(argv[6]);
    m_1 = atof(argv[7]);
    m_2 = atof(argv[8]);
    m_inf = atof(argv[9]);
    tfinal = atof(argv[10]);
    num_runs = atof(argv[11]);
    SEED = atof(argv[12]);
    // Set fitness array
    w[0] = 1;
    w[1] = 1 - delta;
    w[2] = 1 + s;
    // Total number of demes and total population size
    L = L1 * L2;
    Ntot = L * N;

    // Basic checks on parameters
    if (m_1 + m_2 + m_inf >= 1.0000001) {
        printf("Invalid paramters: must have m1+m2+minf <= 1\n");
        return 0; 
    }
    for (k = 0; k < GTYPES; k++) {
        if (w[k] < 0) {
            printf("Invalid paramters: Wrightian fitness must be >= 0\n");
            return 0; 
        }
    }

    // Print out variables to screen:
    /* printf("Using values:\nL = %i\nN = %f\nNtot= %f\n", L, N, Ntot); */
    /* for (i = 0; i < 2; i++)  */
    /*     printf("U%d = %10.9f\n", i, mut[i]); */
    /* printf("d = %f\nh = %f\ns = %f\n", delta, h, s); */
    /* printf("r = %f\ntfinal = %f\nranseed = %ld\n", r, tfinal, SEED); */
    /* printf("m_1 = %f\nm_2 = %f\nm_inf = %f\n", m_1, m_2, m_inf); */
    // FOR DEBUGGING
    /* for (i = 0; i < GTYPES; i++)  */
    /*     printf("%f ", w[i]); */
    /* printf("\n"); */

    // Print out variables to screen:
    printf("***Parameters***\nL1 = %u; L2 = %u; N = %u; Ntot = %u\n", 
            L1, L2, N, Ntot);
    printf("m_1 = %g\nm_2 = %g\nm_inf = %g\n", m_1, m_2, m_inf);
    printf("mu1 = %g; delta = %g; s = %g\n", mu, delta, s);
    printf("tfinal = %g\nnumber of runs = %g\nranseed = %lu\n", 
            (double)tfinal, (double)num_runs, SEED);

    // gsl random setup:
    gsl_rng_env_setup();
    T = gsl_rng_default;
    R = gsl_rng_alloc (T);
    gsl_rng_set(R, SEED);

    // Open summary data file
    outfile = fopen(argv[13], "a");

    for (run = 0; run < num_runs; run += 1) {
        // Initialize population arrays
        // Genotype counts in each deme
        unsigned int n[L1][L2][GTYPES]; 
        // Total numbers in the entire population
        unsigned int ntot[GTYPES]; 
        // Initialize population as wild type + one single mutant
        for (i = 0; i < L1; i+=1) {
            for (j = 0; j < L2; j+=1) {
                n[i][j][0] = N;
                for (k = 1; k < GTYPES; k+=1) {
                    n[i][j][k] = 0;
                }
            }
        }
        n[0][0][0] = N - 1;
        n[0][0][1] = 1;
        ntot[0] = Ntot - 1;
        ntot[1] = 1;
        ntot[2] = 0;
        // Run each replicate until lineage is lost or double mutant fixes 
        for (t = 0; t < tfinal; t += 1) {
            // Evolve for one generation
            next_gen(L1, L2, N, w, mu, m_1, m_2, m_inf, n, ntot);
            // Stop if mutant lineage is extinct
            if (ntot[1] + ntot[2] == 0) {
                break;
            }
            // Record time if there's a new double-mutant lineage:
            // Currently, only recording of t2 is implemented.
            if ((t_1 == 0) && (ntot[2] > 0)) {
                t_1 = t;
            }
            // Reset t_1 to 0 if double-mutant extinct
            else if ((t_1 >= 0) && (ntot[2] == 0)) {
                t_1 = 0;
            }
            // Stop if double mutant fixed
            if (ntot[2] == Ntot) {
                t_fix = t;
                break;
            }
        }
        // Three possibilities: lineage went extinct, lineage fixed, and
        // lineage is still segregating
        // If the lineage is fixed, count as a success even if 2-mutant is not
        // fixed. This should only matter if the population is near the
        // sequential fixation regime.
        if (ntot[2] + ntot[1] == Ntot) {
            num_success += 1;
            printf("SUCCESS! ");
            // Final genotype counts
            for (i = 0; i < GTYPES; i++) {
                fprintf(outfile, "%u ", ntot[i]);
                printf("%u ", ntot[i]);
            }
            // Estimated time that successful 2-mutant is generated, and time the 2-mutant fixes.
            if (ntot[2] == Ntot) {
                fprintf(outfile, "%lu %lu\n", t_1, t_fix);
                printf("%lu %lu", t_1, t_fix);
            }
            // If 2-mutant is not fixed, record t_fix as -1
            else {
                fprintf(outfile, "%lu %i\n", t_1, -1);
                printf("%lu %lu", t_1, t_fix);
            }
            printf("...%u runs to go.\n", num_runs - run);
        }
        // If lineage is still segregating:
        if ( (ntot[2] + ntot[1] > 0) && (ntot[2] + ntot[1] < Ntot) ) {
            num_seg += 1;
            printf("Still segregating. ");
            // Final genotype counts
            for (i = 0; i < GTYPES; i++) {
                fprintf(outfile, "%u ", ntot[i]);
                printf("%u ", ntot[i]);
            }
            // If no double-mutants present
            if (t_1 == 0) {
                fprintf(outfile, "%i %i\n", -1, -1);
                printf("%i %i", -1, -1);
            }
            // If double-mutant present but not fixed
            else if (t_1 > 0) {
                fprintf(outfile, "%lu %i\n", t_1, -1);
                printf("%lu %i", t_1, -1);
            }
            printf("...%u runs to go.\n", num_runs - run);
        }
    }
    // Write number of successful runs and total number of trials
    fprintf(outfile, "# %u %u %u %g %g %g %g %g %g %lu %lu %u %u %u\n",
            L1, L2, N, mu, delta, s, m_1, m_2, m_inf, tfinal, SEED, num_runs,
            num_success, num_seg);
    fclose(outfile);
    return 0;
}
