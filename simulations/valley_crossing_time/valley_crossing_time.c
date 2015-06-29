/*
 * valley_crossing_time.c
 *
 * Created by Michael McLaren (mmclaren@stanford.edu) on Fri Feb 6 2015.
 *
 * Simulation of a subdivided population with recurrent generation of single
 * mutants. Records the approximate time that the successful double mutant is
 * generated and the time that the double mutant is fixed in the total
 * population.
 *
 * Spatial structure can follow the island model or the 1- or 2-dimensional
 * stepping stone model.
 *
 * Compile with:
 * gcc -c valley_crossing_time.c; gcc valley_crossing_time.o -lgsl -lgslcblas -lm -o valley_crossing_time
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
        double w[GTYPES], double mu0, double mu1, 
        double m_1, double m_2, double m_inf, 
        unsigned int n[L1][L2][GTYPES], unsigned int ntot[GTYPES]) {
    /* Parameters:  
     * n[L1][L2][GTYPES] -- counts of genotypes in deme (i,j)
     * ntot[GTYPES] -- counts of genotypes in the total population
     * w[GTYPES] -- fitnesses of the different genotypes
     * mu0 -- mutation rate from genotype 0 to 1
     * mu1 -- mutation rate from genotype 1 to 2
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
    // within deme GENOTYPE frequencies (after mating!)
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
            // Mutation
            x[i][j][0] = x0[0] * (1 - mu0);
            x[i][j][1] = x0[1] * (1 - mu1) + x0[0] * mu0;
            x[i][j][2] = x0[2] + x0[1] * mu1; 
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
    // Mutation rates
    double mu0, mu1; 
    // Fitnesses of the 3 genotypes
    double w[GTYPES];
    // Time in current run
    unsigned long int t = 0;
    // Max time allowed
    unsigned long int tfinal;
    // Times to record to out file (currently only t_first and t_fix implemented)
    unsigned long int t_2, t_est, t_fix = 0;
    // Data files
    FILE *outfile;
    FILE *datafile;
    // Iterator variables
    unsigned int run, i, j, k;

    //Initialize variables:
    if (argc != 14) {
        // Arg no.'s:  0       1  2  3 4   5   6 7 8  9  10   11     12
        printf("Usage: metapop L1 L2 N mu0 mu1 d s m1 m2 minf tfinal ranseed "
        //     13
               "outfile\n");
        return 0; 
    }
    // Read in command line arguments
    L1 = atof(argv[1]);
    L2 = atof(argv[2]);
    N = atof(argv[3]);
    mu0 = atof(argv[4]);
    mu1 = atof(argv[5]);
    delta = atof(argv[6]);
    s = atof(argv[7]);
    m_1 = atof(argv[8]);
    m_2 = atof(argv[9]);
    m_inf = atof(argv[10]);
    tfinal = atof(argv[11]);
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
    printf("***Parameters***\nL1 = %u; L2 = %u; N = %u; Ntot = %u\n", 
            L1, L2, N, Ntot);
    printf("m_1 = %g\nm_2 = %g\nm_inf = %g\n", m_1, m_2, m_inf);
    printf("mu0 = %g; mu1 = %g; delta = %g; s = %g\n", mu0, mu1, delta, s);
    printf("tfinal = %g; ranseed = %lu\n", 
            (double)tfinal, SEED);

    // gsl random setup:
    gsl_rng_env_setup();
    T = gsl_rng_default;
    R = gsl_rng_alloc (T);
    gsl_rng_set(R, SEED);

    // Open summary data file
    outfile = fopen(argv[13], "a");

    // Initialize population arrays
    // Genotype counts in each deme
    unsigned int n[L1][L2][GTYPES]; 
    // Total numbers in the entire population
    unsigned int ntot[GTYPES]; 
    // Initialize population as wild type
    for (i = 0; i < L1; i+=1) {
        for (j = 0; j < L2; j+=1) {
            n[i][j][0] = N;
            n[i][j][1] = 0;
            n[i][j][2] = 0;
        }
    }
    ntot[0] = Ntot;
    ntot[1] = 0;
    ntot[2] = 0;
    // Run each replicate until double mutant fixes 
    for (t = 0; t < tfinal; t += 1) {
        // Evolve for one generation
        next_gen(L1, L2, N, w, mu0, mu1, m_1, m_2, m_inf, n, ntot);
        // Record time if there's a new double-mutant lineage:
        if ((t_2 == 0) && (ntot[2] > 0)) {
            t_2 = t;
        }
        // Reset t_2 to 0 if double-mutant extinct
        else if ((t_2 > 0) && (ntot[2] == 0)) {
            t_2 = 0;
        }
        /* else if ((t_est = 0) && ((double)ntot[2] > s)) { */
        /*     t_est = t; */
        /* } */
        // Stop if double mutant fixed
        if (ntot[2] == Ntot) {
            t_fix = t;
            break;
        }
    }

    // Open summary data file and record data:
    outfile = fopen(argv[13], "a");
    fprintf(outfile, "%ld ", SEED);
    for (i = 0; i < GTYPES; i++) fprintf(outfile, "%u ", ntot[i]);
    // Record the most recent time there was a change from 0 to 1 AB haplotypes
    // Record the final time (whether AB fixed or not)
    fprintf(outfile, "%lu %lu\n", t_2, t);
    fclose(outfile);

    return 0;
}
