/*
 * lineage_tracer.c
 *
 * Created by Michael McLaren (mmclaren@stanford.edu) on Sun 1 Feb 2015.
 *
 * Simulate mutant lineage while tracking n, Fst, and gamma, conditional on
 * non-extinction for a minimum number of generations.
 * Spatial structure can follow the island model or the 1- or 2-dimensional
 * stepping stone model.
 *
 * Compile with:
 * gcc -c lineage_tracer.c; gcc lineage_tracer.o -lgsl -lgslcblas -lm -o lineage_tracer
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

void calc_stats(unsigned int L1, unsigned int L2, unsigned int N, 
        unsigned int n[L1][L2], unsigned int ntot, 
        double *fst_cur, double *gamma_cur, double *f4_cur) {
    // Vars
    unsigned int L = L1 * L2;
    unsigned int Ntot = N * L;
    // Total mutant frequency
    double xbar = (double)ntot / Ntot;
    // Local frequency
    double xij = 0;
    // Identity stats
    double fst = 0;
    double gamma = 0;
    double f4 = 0;
    int i, j, k;
    // Calculate Fst and gamma. Only defined for 0 < ntot < Ntot
    // Else if ntot == 0 or 2*Ntot, set Fst and gamma to -1 to indicate
    // undefined.
    if (ntot > 0 && ntot < Ntot) {
        for (i = 0; i < L1; i += 1) {
            for (j = 0; j < L2; j += 1) {
                xij = (double)n[i][j] / N;
                fst += (xij - xbar) * (xij - xbar);
                gamma += (xij - xbar) * (xij - xbar) * (xij - xbar);
                f4 += (xij - xbar) * (xij - xbar) * (xij - xbar) * (xij - xbar);
            }
        }
        fst = fst / (L * xbar * (1 - xbar));
        gamma = gamma / (L * xbar * (1 - xbar) * (1 - 2 * xbar));
        f4 = f4 / (L * xbar * (1 - xbar) * (1 - 3 * xbar + 3 * xbar * xbar));
    }
    else {
        fst = -1;
        gamma = -1;
        f4 = -1;
    }
    /* printf("xbar: %f; ntot: %i; fst: %f\n", xbar, ntot, fst); */
    *fst_cur = fst;
    *gamma_cur = gamma;
    *f4_cur = f4;
}

void next_gen(unsigned int L1, unsigned int L2, unsigned int N, 
        double s, double m_1, double m_2, double m_inf, 
        unsigned int n[L1][L2], unsigned int *ntot_cur) {
    /* Parameters:  
     * n[L1][L2] -- counts of genotypes in deme (i,j)
     * s -- selection coefficient of the mutation
     * m_1 -- migration rate across rows
     * m_2 -- migration rate across columns
     * m_inf -- global migration rate
     * L1 -- number of rows
     * L1 -- number of columns
     * N -- local population size
     *
     * Order: Migration, selection, sampling.
     */
    // Frequencies in deme (i,j)
    double x[L1][L2]; 
    // Frequencies in deme (i,j) after migration
    double xmig[L1][L2]; 
    // within deme mutant frequency after selection
    double xsel;
    // total number of demes
    double L = L1 * L2;
    // Total mutant frequency in population
    double xtot = (double)*ntot_cur / (N * L);
    int i, j, k;
    // Total number of mutants
    unsigned int ntot = 0;
    // Calculate the frequency within each deme from the count
    for (i = 0; i < L1; i += 1) {
        for (j = 0; j < L2; j += 1) {
            x[i][j] = (double)n[i][j] / N;
        }
    }
    // Migration; results stored in xmig
    for (i = 0; i < L1; i += 1) {
        for (j = 0; j < L2; j += 1) {
            // Migration; wrap-around boundaries
            xmig[i][j] = (1 - m_1 - m_2 - m_inf) * x[i][j]
                + 0.5 * m_1 * (x[(i-1+L1)%L1][j] + x[(i+1)%L1][j])
                + 0.5 * m_2 * (x[i][(j-1+L2)%L2] + x[i][(j+1)%L2])
                + m_inf * xtot;
        }
    }
    // Selection, and sampling (within demes)
    for (i = 0; i < L1; i += 1) {
        for (j = 0; j < L2; j += 1) {
            // Selection
            xsel = xmig[i][j] * (1 + s) / (1 + s * xmig[i][j]);
            // Sample within each deme:
            n[i][j] = gsl_ran_binomial(R, xsel, N);
            // Keep track of total genotype counts
            ntot += n[i][j];
        }
    }
    // Update the total mutant count
    *ntot_cur = ntot;
}

int main(int argc, char *argv[]) {
    long SEED;
    // Number of rows, number of columns, total number of demes
    unsigned int L, L1, L2; 
    // Local population size, global population size;
    unsigned int N, Ntot; // 
    // Migration rates: between rows, between col's, and global
    double m_1, m_2, m_inf; 
    // Initial population-wide frequency
    double x0;
    // selection coefficient
    double s; 
    // Time in current run
    unsigned int t = 0;
    // Only record if survives for at least tmin generations
    unsigned int t_min; 
    // Max number of generations to record
    unsigned int t_max; 
    // Maximum number of tries for a long-lived lineage
    unsigned int max_tries; 
    // Total number of A alleles in the entire population
    unsigned int ntot = 0; 
    // Current Fst value
    double fst = -1; 
    // Current gamma value
    double gamma = -1; 
    // Current F4 value
    double f4 = -1; 
    // Data file for recording lineage trajectory
    FILE *data_file;
    // Iterator variables
    unsigned int run, i, j, k;

    //Initialize variables:
    if (argc != 14) {
        // Arg no.'s:  0       1  2  3 4 5  6  7    8    9    10       11 
        printf("Usage: metapop L1 L2 N s m1 m2 minf tmin tmax maxtries x0 "
        //     12       13     
               "ranseed outfile\n");
        return 0; 
    }
    // Read in command line arguments
    L1 = atof(argv[1]);
    L2 = atof(argv[2]);
    N = atof(argv[3]);
    s = atof(argv[4]);
    m_1 = atof(argv[5]);
    m_2 = atof(argv[6]);
    m_inf = atof(argv[7]);
    t_min = atof(argv[8]);
    t_max = atof(argv[9]);
    max_tries = atof(argv[10]);
    x0 = atof(argv[11]);
    SEED = atof(argv[12]);
    // Total number of demes and total population size
    L = L1 * L2;
    Ntot = L * N;

    // Basic checks on parameters
    if (m_1 + m_2 + m_inf >= 1.0000001) {
        printf("Invalid paramters: must have m1+m2+minf <= 1\n");
        return 0; 
    }
    if (s < -1) {
        printf("Invalid paramters: Wrightian fitness must be >= 0\n");
        return 0; 
    }

    // Print out variables to screen:
    printf("***Parameters***\nL1 = %u; L2 = %u; N = %u; Ntot = %u\n", 
            L1, L2, N, Ntot);
    printf("m_1 = %g; m_2 = %g; m_inf = %g\n", m_1, m_2, m_inf);
    printf("s = %g; x0 = %g\n", s, x0);
    printf("t_min = %g; t_max = %g; max_tries = %g\nrandseed = %lu\n", 
            (double)t_min, (double)t_max, (double)max_tries, SEED);
    if (x0 == 0) {
        printf("Since x0 = 0, starting with single mutant in first deme.\n");
    }

    // gsl random setup:
    gsl_rng_env_setup();
    T = gsl_rng_default;
    R = gsl_rng_alloc (T);
    gsl_rng_set(R, SEED);

    // Array holding counts of the three genotypes in each deme
    unsigned int n[L1][L2];
    // Arrays for recording the trajectory of ntot and Fst
    unsigned int ntot_traj[t_max + 1];
    double fst_traj[t_max + 1];
    double gamma_traj[t_max + 1];
    double f4_traj[t_max + 1];

    // Try up to max_tries times until a run lasting at least t_min generations
    for (run = 0; run < max_tries; run += 1) {
        // If x0 == 0, start lineage with a single individual in the first deme 
        if (x0 == 0) {
            for (i = 0; i < L1; i+=1) {
                for (j = 0; j < L2; j+=1) {
                    n[i][j] = 0;
                }
            }
            n[0][0] = 1;
            ntot = 1;
        }
        else {
            // Initialize population as frequency x0 in all demes
            ntot = 0;
            for (i = 0; i < L1; i+=1) {
                for (j = 0; j < L2; j+=1) {
                    n[i][j] = x0 * N;
                    ntot += n[i][j];
                }
            }
        }
        // Run each replicate until lineage is lost or until t_max
        for (t = 0; (t < t_max) && (ntot != 0); t += 1) {
            // Log ntot and Fst
            calc_stats(L1, L2, N, n, ntot, &fst, &gamma, &f4);
            ntot_traj[t] = ntot;
            fst_traj[t] = fst;
            gamma_traj[t] = gamma;
            f4_traj[t] = f4;
            // Evolve for one generation
            next_gen(L1, L2, N, s, m_1, m_2, m_inf, n, &ntot);
        }
        // Log ntot and Fst of last generation
        calc_stats(L1, L2, N, n, ntot, &fst, &gamma, &f4);
        ntot_traj[t] = ntot;
        fst_traj[t] = fst;
        gamma_traj[t] = gamma;
        f4_traj[t] = f4;
        // End loop and record output if lineage survived until time t_min
        if ((t >= t_min) && (ntot_traj[t_min] > 0)) {
            printf("Found successful lineage. ");
            break;
        }
    }

    if ((t >= t_min) && (ntot_traj[t_min] > 0)) {
    // Open summary data file
        // Open detailed data file and record trajectory of ntot and Fst
        data_file = fopen(argv[13], "w");
        printf("Recording.\n");
        fprintf(data_file, "#L1 L2 N s m1 m2 m t_min t_max max_tries x0 seed\n");
        fprintf(data_file, "# %u %u %u %g %g %g %g %g %g %g %g %lu\n",
                L1, L2, N, s, m_1, m_2, m_inf, (double)t_min, (double)t_max,
                (double)max_tries, x0, SEED);
        for (j = 0; j < t + 1; j += 1) {
            fprintf(data_file, "%u %u %10.9f %10.9f %10.9f\n", j, ntot_traj[j],
                    fst_traj[j], gamma_traj[j], f4_traj[j]);
        }
        fclose(data_file);
    }

    return 0;
}
