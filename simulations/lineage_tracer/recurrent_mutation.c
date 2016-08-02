/*
 * haploid.c
 *
 * Created by Michael McLaren (mmclaren@stanford.edu) on Sun 1 Feb 2015.
 *
 * Simulate mutant lineage while tracking n, Fst, and gamma.
 * Spatial structure can follow the island model or the 1/2-dimensional
 * stepping stone models.
 *
 * Compile with:
 * gcc -c recurrent_mutation.c; gcc recurrent_mutation.o -lgsl -lgslcblas -lm -o tracer_mut
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

void calc_stats(unsigned long L1, unsigned long L2, unsigned long N, 
        unsigned long n[L1][L2], unsigned long ntot, 
        double *fst_cur, double *gamma_cur) {
    // Vars
    unsigned long L = L1 * L2;
    unsigned long Ntot = N * L;
    // Total mutant frequency
    double xbar = (double)ntot / Ntot;
    // Local frequency
    double xij = 0;
    // Fst and gamma
    double fst = 0;
    double gamma = 0;
    long i, j, k;
    // Calculate Fst and gamma. Only defined for 0 < ntot < Ntot
    // Else if ntot == 0 or Ntot, leave Fst and gamma as 0
    if (ntot > 0 && ntot < Ntot) {
        for (i = 0; i < L1; i += 1) {
            for (j = 0; j < L2; j += 1) {
                xij = (double)n[i][j] / N;
                fst += (xij - xbar) * (xij - xbar);
                gamma += (xij - xbar) * (xij - xbar) * (xij - xbar);
            }
        }
        fst = fst / (L * xbar * (1 - xbar));
        gamma = gamma / (L * xbar * (1 - xbar) * (1 - 2 * xbar));
    }
    /* printf("xbar: %f; ntot: %i; fst: %f\n", xbar, ntot, fst); */
    *fst_cur = fst;
    *gamma_cur = gamma;
}

void next_gen(unsigned long L1, unsigned long L2, unsigned long N, 
        double mu, double s, 
        double m_1, double m_2, double m_inf, 
        unsigned long n[L1][L2], unsigned long *ntot_cur) {
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
    // Total mutant frequency in population _after mutation_
    double xtot = 0;
    // Counter variables
    int i, j, k;
    // Total number of mutants
    unsigned int ntot = 0;
    // Calculate the frequency within each deme from the count; 
    for (i = 0; i < L1; i += 1) {
        for (j = 0; j < L2; j += 1) {
            // Before mutation
            x[i][j] = (double)n[i][j] / N;
            // Bidirectional mutation with probability mu
            x[i][j] = x[i][j] * (1 - mu) + (1-x[i][j]) * mu ;
            // Keep track of total frequency
            xtot += x[i][j];
        }
    }
    xtot /= L;
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
    unsigned long L, L1, L2; 
    // Local population size, global population size;
    unsigned long N, Ntot; // 
    // Migration rates: between rows, between col's, and global
    double m_1, m_2, m_inf; 
    // Mutation rate from wt to mutant
    double mu;
    // Initial population-wide frequency
    double x0;
    // selection coefficient
    double s; 
    // Time in current run
    unsigned long t = 0;
    // Max number of generations to record
    unsigned long t_max; 
    // Generations per stat recording
    unsigned long step = 1;
    // Total number of A alleles in the entire population
    unsigned long ntot = 0; 
    // Current Fst value
    double fst = -1; 
    // Current gamma value
    double gamma = -1; 
    // Data file for recording lineage trajectory
    FILE *data_file;
    // Iterator variables
    unsigned long i, j, k;

    //Initialize variables:
    if (argc != 14) {
        // Arg no.'s:  0       1  2  3 4  5  6  7 8    9    10
        printf("Usage: metapop L1 L2 N mu s m1 m2 minf tmax x0 "
        //     11       12      13
               "ranseed outfile step\n");
        return 0; 
    }
    // Read in command line arguments
    L1 = atof(argv[1]);
    L2 = atof(argv[2]);
    N = atof(argv[3]);
    mu = atof(argv[4]);
    s = atof(argv[5]);
    m_1 = atof(argv[6]);
    m_2 = atof(argv[7]);
    m_inf = atof(argv[8]);
    t_max = atof(argv[9]);
    x0 = atof(argv[10]);
    SEED = atof(argv[11]);
    step = atof(argv[13]);
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
    // Prevent segfault when number of generations to record is too large
    if (t_max / step > 260000) {
        printf("Invalid paramters: Must have tmax*step <= 2.6e5\n");
        return 0; 
    }

    // Print out variables to screen:
    printf("***Parameters***\nL1 = %lu; L2 = %lu; N = %lu; Ntot = %g\n", 
            L1, L2, N, (double)Ntot);
    printf("m_1 = %g; m_2 = %g; m_inf = %g\n", m_1, m_2, m_inf);
    printf("mu = %g; s = %g; x0 = %g\n", mu, s, x0);
    printf("t_max = %g\nrandseed = %lu\n", 
            (double)t_max, SEED);

    // gsl random setup:
    gsl_rng_env_setup();
    T = gsl_rng_default;
    R = gsl_rng_alloc (T);
    gsl_rng_set(R, SEED);

    // Array holding counts of the three genotypes in each deme
    unsigned long n[L1][L2];
    // Arrays for recording the trajectory of ntot, Fst, and gamma
    // Note: This will seg fault if t is too large, so need to make sure
    // t_max/step is small enough
    unsigned long ntot_traj[t_max/step + 1];
    double fst_traj[t_max/step + 1];
    double gamma_traj[t_max/step + 1];

    // Initialize population as frequency x0 in all demes
    ntot = 0;
    for (i = 0; i < L1; i+=1) {
        for (j = 0; j < L2; j+=1) {
            n[i][j] = x0 * N;
            ntot += n[i][j];
        }
    }
    // Run until t_max
    for (t = 0; t < t_max; t += 1) {
        if (t % step == 0) {
            // Log ntot, Fst, and gamma
            calc_stats(L1, L2, N, n, ntot, &fst, &gamma);
            ntot_traj[t/step] = ntot;
            fst_traj[t/step] = fst;
            gamma_traj[t/step] = gamma;
        }
        // Evolve for one generation
        next_gen(L1, L2, N, mu, s, m_1, m_2, m_inf, n, &ntot);
    }
    // Log ntot and Fst of last generation
    calc_stats(L1, L2, N, n, ntot, &fst, &gamma);
    if (t % step == 0) {
        k = t / step;
    }
    else {
        k = t / step + 1;
    }
    ntot_traj[k] = ntot;
    fst_traj[k] = fst;
    gamma_traj[k] = gamma;

    // Open detailed data file and record trajectory of ntot and Fst
    data_file = fopen(argv[12], "w");
    printf("Recording.\n");
    fprintf(data_file, "#L1 L2 N mu s m1 m2 m t_max x0 seed\n");
    fprintf(data_file, "# %lu %lu %lu %g %g %g %g %g %g %g %lu\n",
            L1, L2, N, mu, s, m_1, m_2, m_inf, (double)t_max,
            x0, SEED);
    // Record up to the last time
    for (j = 0; j < t; j += step) {
        k = j/step;
        fprintf(data_file, "%lu %lu %10.9f %10.9f\n", k*step,
                ntot_traj[k], fst_traj[k], gamma_traj[k]);
    }
    fclose(data_file);


    return 0;
}
