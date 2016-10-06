"""Predictions for valley crossing in asexual populations.

Can import with
sys.path += ['/home/michael/research/valley_crossing/metapopulations/asexual/'\
        + 'src/analysis/']
import predictions as pred

To reload, can use
from importlib import reload
"""
from __future__ import division

import csv
from scipy import integrate, optimize
from scipy.misc import derivative
from pylab import *
from scipy.special import gamma as gammaf
from scipy.special import erf, erfi, erfcx, hyp1f1

########## Well-mixed populations 
##### Basic pop gen results
def fixation_probability(Ne, s, x):
    """Fixation probability of a mutation starting at initial frequency x.

    Assumes a well-mixed asexual population of size N.
    """
    if s == 0:
        return x
    else:
        return (1-exp(-2*Ne*s*x)) / (1-exp(-2*Ne*s))

##### Valley-crossing results (well-mixed populations)
def p1_wm(alpha, mu1, delta, s, p2=None): 
    """Probability that a 1-mutant is successful in a well-mixed pop'n.
    
    alpha -- drift parameter
    mu1 -- rate of mutation from type 1 to type 2
    delta -- deleterious selection coefficient of 1-mutant
    s -- beneficial selection coefficient of 2-mutant
    p2 -- Optional. Probability that a 2-mutant is successful. If left to
        default, set to p2 = s/alpha.

    The probability $p_1$ that a 1-mutant is successful is calculated from
    Equation ?.
    """
    if p2 == None:
        p2 = s / alpha
    p1 = ( -delta + sqrt(delta**2 + 4 * alpha * mu1 * p2) ) / (2 * alpha)
    return p1

def p1_wm_exact(alpha, mu1, delta, s, p2=None): 
    """Probability that a 1-mutant is successful in a well-mixed pop'n.
    
    alpha -- drift parameter
    mu1 -- rate of mutation from type 1 to type 2
    delta -- deleterious selection coefficient of 1-mutant
    s -- beneficial selection coefficient of 2-mutant
    p2 -- Optional. Probability that a 2-mutant is successful. If left to
        default, set to p2 = s/alpha.

    The probability $p_1$ that a 1-mutant is successful is calculated from
    Equation ?.
    """
    if p2 == None:
        p2 = s / alpha
    p1 = ( -(delta + mu1) + sqrt((delta + mu1)**2 \
            + 4 * (alpha - delta) * mu1 * p2) ) \
            / (2 * (alpha - delta))
    return p1

def drift_time_wm(alpha, mu1, delta, s, p2=None):
    """Expected drift time of a successful 1-mutant in a well-mixed pop'n.

    Parameters: see help for the 'p1_genic' function.

    The expected drift time $E[T_1]$ is calculated from Equation ?.
    """
    if p2 == None:
        p2 = s / alpha
    # Successful single-mutant drift time
    A = sqrt(delta**2 + 4 * alpha * mu1 * p2)
    tau1 = 2 * log(2 / (1 + delta / A )) / (-delta + A)
    return tau1

def drift_time_wm_exact(alpha, mu1, delta, s, p2=None):
    """Expected drift time of a successful 1-mutant in a well-mixed pop'n.

    Parameters:

    The expected drift time $E[T_1]$ is calculated from Equation ?.
    """
    if p2 == None:
        p2 = s / alpha
    # individual birth rate
    b1 = alpha
    # individual death rate
    d1 = alpha + delta
    # Compound parameters
    a_p = (d1 + b1 + mu1 + sqrt((d1 - b1 + mu1)**2 + 4 * b1 * mu1 * p2)) \
            / (2 * b1)
    a_m = (d1 + b1 + mu1 - sqrt((d1 - b1 + mu1)**2 + 4 * b1 * mu1 * p2)) \
            / (2 * b1)
    tau1 = log((a_p - a_m) / (a_p - 1)) / (b1 * (1 - a_m))
    return tau1

def drift_time_cdf_wm(alpha, mu1, delta, s, p2=None, pdf=False):
    """Expected drift time of a successful 1-mutant in a well-mixed pop'n.

    Parameters: see help for the 'p1_genic' function.

    """
    # Probability that a 2-mutant establishes
    if p2 == None:
        p2 = s / alpha
    # individual birth rate
    b1 = alpha - delta
    # individual death rate
    d1 = alpha
    a_p = (d1 + b1 + mu1 + sqrt((d1 - b1 + mu1)**2 + 4 * b1 * mu1 * p2)) \
            / (2 * b1)
    a_m = (d1 + b1 + mu1 - sqrt((d1 - b1 + mu1)**2 + 4 * b1 * mu1 * p2)) \
            / (2 * b1)
    ## Probability successful
    # As a function of time
    p1_t = lambda t: (a_p - 1) * (1 - a_m) * (1 - exp(-b1 * (a_p - a_m)*t))\
            / (a_p - 1 + (1 - a_m) * exp(-b1 * (a_p - a_m) * t))
    # p1 given by limit as t\to \infty
    p1 = 1 - a_m
    ## Drift time cdf or pdf
    t1_cdf = lambda t: p1_t(t) / p1
    if pdf:
        t1_pdf = lambda t: derivative(t1_cdf, t, dx=1e-2)
        return t1_pdf
    else:
        return t1_cdf

def ccdf_wm(N, alpha, mu0, mu1, delta, s, p2=None, pdf=False):
    """Distribution of the time to the first successful double mutant.

    Default: return the ccdf (one minus the cdf)
    Alternatively, return the pdf
    """
    # Probability that a double mutant is successful
    if p2 == None:
        p2 = s / alpha
    # individual birth rate
    b1 = alpha
    # individual death rate
    d1 = alpha + delta
    # Compound parameters
    a_p = (d1 + b1 + mu1 + sqrt((d1 - b1 + mu1)**2 + 4 * b1 * mu1 * p2)) \
            / (2 * b1)
    a_m = (d1 + b1 + mu1 - sqrt((d1 - b1 + mu1)**2 + 4 * b1 * mu1 * p2)) \
            / (2 * b1)
    # Complement of the cumulative distribution function
    ccdf = lambda t: ( (a_p - a_m) \
            / ((1 - a_m) * exp(-b1*(a_p-a_m)*t) + a_p - 1) )**(N * mu0 / b1) \
            * exp(-N * mu0 * (1 - a_m) * t)
    if pdf:
        pdf = lambda t: -derivative(ccdf, t, dx=1)
        return pdf
    else:
        return ccdf

def vc_time_wm(N, alpha, mu0, mu1, delta, s, p2=None):
    """Average time to first successful double mutant.
    """
    p1 = p1_wm(alpha, mu1, delta, s, p2) 
    t1 = drift_time_wm(alpha, mu1, delta, s, p2) 
    ccdf = ccdf_wm(N, alpha, mu0, mu1, delta, s, p2)
    # Set the upper limit for the integration to a time that is at least 20
    # times the expected valley crossing time.
    # 1 / (N*mu0*p1) is the expected time to the first successful 1-mutant
    # t1 is the expected drift time of a successful 1-mutant
    # (1 / (N*mu0*p1) + t1) is therefore an upperbound to E[T]; actual
    # E[T] will be smaller due to would-be succesful lineages competing to be
    # first
    upper_lim = 20 * (1 / (N*mu0*p1) + t1)
    # upper_lim = np.inf
    return integrate.quad(ccdf, 0, upper_lim)[0]

########## Subdivided populations

def p1_sub_deme_bd(N, alpha, mu1, delta, s, m):
    """Probability that a 1-mutant is successful under the deme BD approx'n.

    As given by Equation ?.
    """
    Ne = N / (2 * alpha)
    # Deme birth rate
    b1 = N * m * fixation_probability(Ne, -delta, 1/N)
    # Deme death rate
    d1 = N * m * fixation_probability(Ne, delta, 1/N)
    # Transition rate from 1 to 2
    r1 = N * mu1 * fixation_probability(Ne, (delta + s), 1/N)
    # Probability that a 2-mutant deme spreads through the entire population
    theta = 1 - exp(-2 * Ne * s)
    # Probability of success starting from one 1-mutant deme
    rho = ( -(d1 - b1 + r1) + sqrt((d1 - b1 + r1)**2 + 4 * b1 * r1 * theta) ) \
            / (2 * b1)
    # Probability of success starting from one 1-mutant
    p1 = fixation_probability(Ne, -delta, 1/N) * rho
    return p1

def drift_time_sub_deme_bd(N, alpha, mu1, delta, s, m):
    """Expected drift time under the deme BD approx'n.

    The expected drift time $E[T_1]$ is calculated from Equation ?.
    """
    Ne = N / (2 * alpha)
    # Deme birth rate
    b1 = N * m * fixation_probability(Ne, -delta, 1/N)
    # Deme death rate
    d1 = N * m * fixation_probability(Ne, delta, 1/N)
    # Transition rate from 1 to 2
    r1 = N * mu1 * fixation_probability(Ne, (delta + s), 1/N)
    # Probability that a 2-mutant deme spreads through the entire population
    theta = 1 - exp(-2 * Ne * s)
    A = sqrt((d1 - b1 + r1)**2 + 4 * b1 * r1 * theta)
    tau1 = 2 * log(2 * A / (d1 - b1 + r1 + A)) / (-(d1 - b1 + r1) + A)
    return tau1

def drift_time_cdf_sub_deme_bd(N, alpha, mu1, delta, s, m, pdf=False):
    """
    """
    Ne = N / (2 * alpha)
    # Deme birth rate
    b1 = N * m * fixation_probability(Ne, -delta, 1/N)
    # Deme death rate
    d1 = N * m * fixation_probability(Ne, delta, 1/N)
    # Transition rate from 1 to 2
    r1 = N * mu1 * fixation_probability(Ne, (delta + s), 1/N)
    # Probability that a 2-mutant deme spreads through the entire population
    theta = 1 - exp(-2 * Ne * s)
    # $a_\pm$
    a_p = (d1 + b1 + r1 + sqrt((d1 - b1 + r1)**2 + 4 * b1 * r1 * theta)) \
            / (2 * b1)
    a_m = (d1 + b1 + r1 - sqrt((d1 - b1 + r1)**2 + 4 * b1 * r1 * theta)) \
            / (2 * b1)
    # Probability successful
    p1 = 1 - a_m
    p1_cdf = lambda t: (a_p - 1) * (1 - a_m) * (1 - exp(-b1 * (a_p - a_m)*t))\
            / (a_p - 1 + (1 - a_m) * (1 - exp(-b1 * (a_p - a_m) * t)))
    t1_cdf = lambda t: p1_cdf(t) / p1
    if pdf:
        t1_pdf = lambda t: derivative(t1_cdf, t, dx=1e-2)
        return t1_pdf
    else:
        return t1_cdf

def ccdf_sub_deme_bd(L, N, alpha, mu0, mu1, delta, s, m, pdf=False):
    Ne = N / (2 * alpha)
    # Deme birth rate
    b1 = N * m * fixation_probability(Ne, -delta, 1/N)
    # Deme death rate
    d1 = N * m * fixation_probability(Ne, delta, 1/N)
    # Transition rate from 0 to 1
    r0 = N * mu0 * fixation_probability(Ne, -delta, 1/N)
    # Transition rate from 1 to 2
    r1 = N * mu1 * fixation_probability(Ne, (delta + s), 1/N)
    # Probability that a 2-mutant deme spreads through the entire population
    theta = 1 - exp(-2 * Ne * s)
    a_p = (d1 + b1 + r1 + sqrt((d1 - b1 + r1)**2 + 4 * b1 * r1 * theta)) \
            / (2 * b1)
    a_m = (d1 + b1 + r1 - sqrt((d1 - b1 + r1)**2 + 4 * b1 * r1 * theta)) \
            / (2 * b1)
    ccdf = lambda t: ( (a_p - a_m) \
            / ((1 - a_m) * exp(-b1*(a_p-a_m)*t) + a_p - 1) )**(L * r0 / b1) \
            * exp(-L * r0 * (1 - a_m) * t)
    if pdf:
        pdf = lambda t: -derivative(ccdf, t, dx=1)
        return pdf
    else:
        return ccdf

def vc_time_sub_deme_bd(L, N, alpha, mu0, mu1, delta, s, m):
    Ne = N / (2 * alpha)
    # Deme birth rate
    b1 = N * m * fixation_probability(Ne, -delta, 1/N)
    # Deme death rate
    d1 = N * m * fixation_probability(Ne, delta, 1/N)
    # Transition rate from 0 to 1
    r0 = N * mu0 * fixation_probability(Ne, -delta, 1/N)
    # Transition rate from 1 to 2
    r1 = N * mu1 * fixation_probability(Ne, (delta + s), 1/N)
    # Probability that a 2-mutant deme spreads through the entire population
    theta = 1 - exp(-2 * Ne * s)
    a_p = (d1 + b1 + r1 + sqrt((d1 - b1 + r1)**2 + 4 * b1 * r1 * theta)) \
            / (2 * b1)
    a_m = (d1 + b1 + r1 - sqrt((d1 - b1 + r1)**2 + 4 * b1 * r1 * theta)) \
            / (2 * b1)
    ccdf = lambda t: ( (a_p - a_m) \
            / ((1 - a_m) * exp(-b1*(a_p-a_m)*t) + a_p - 1) )**(L * r0 / b1) \
            * exp(-L * r0 * (1 - a_m) * t)
    # Set the upper limit for the integration to a time that is at least 20
    # times the expected valley crossing time.
    # 1 / (L*r0*(1-a_m)) is the expected time to the first successful 1-mutant
    # 1 / r1 is an upper bound to the expected drift time of a successful 1-mutant
    # (1 / (L*r0*(1-a_m)) + 1 / r1) is therefore an upperbound to E[T]; actual
    # E[T] will be smaller due to effects of competing would-be succesful lineages
    upper_lim = 20 * (1 / (L*r0*(1-a_m)) + 1 / r1)
    # upper_lim = np.inf
    return integrate.quad(ccdf, 0, upper_lim)[0]

def get_eta(N, alpha, delta, s):
    """

    """
    Ne = N / (2 * alpha)
    pfix01 = fixation_probability(Ne, -delta, 1/N)
    pfix12 = fixation_probability(Ne, s + delta, 1/N)
    p2 = 1 - exp(-s/alpha)
    theta = 1 - exp(-N*s/alpha)
    eta = N * pfix01 * pfix12 * theta / p2
    return eta

# Low migration limit, time until first successful double-mutant deme
def sub_tau_no_mig(L, N, alpha, mu0, mu1, delta, s): 
    """Time until the first 2-mutant deme.

    """
    Ne = N / (2 * alpha)
    # Transition rate from 0 to 1
    r0 = N * mu0 * fixation_probability(Ne, -delta, 1/N)
    # Transition rate from 1 to 2
    r1 = N * mu1 * fixation_probability(Ne, (delta + s), 1/N)

    # The full sum is 
    # tau = sum([ prod([(L-j)*r0 for j in range(0, k-1)]) \
    #         / prod([(L-j)*r0 + j*r1 for j in range(0, k)]) \
    #         for k in range(0, L+1)])
    # However this fails b/c the product in the demoninator becomes
    # indistinguishable from 0. Instead, need to stop computing new terms
    # before this point.
    # Algorithm for computing sum:
    # http://profjrwhite.com/matlab_course/pdf_files_ex/infinite_series.pdf
    # Start with the first two terms (the k=0 and k=1 terms)
    total = 1/(L*r0) + 1/((L-1)*r0 + r1)
    epsilon = Inf
    while (epsilon > 1e-5 and k < L+1):
        # Ratio of the k+1-th term to the k-th term
        ratio = (L - (k - 1)) * r0 / ((L - k) * r0 + k * r1)
        # Calulate the next term as current term * ratio of next term to current
        # term
        term = term * ratio
        total = total + term
        epsilon = abs(term / total)
    return tau

def sub_tau_no_mig1(L, N, alpha, mu0, mu1, delta, s, rates=False): 
    """Find tau in the limit of m = 0.

    Specifically, gives the expected time until the first AB deme.
    Found by numerically integrating Pr[T>t] as found by the large L
    approximation.
    N diploid individuals, or 2*N haploid individuals.
    """
    Ne = N / (2 * alpha)
    # Transition rate from 0 to 1
    r0 = N * mu0 * fixation_probability(Ne, -delta, 1/N)
    # Transition rate from 1 to 2
    r1 = N * mu1 * fixation_probability(Ne, (delta + s), 1/N)
    # Probability that a 2-mutant deme spreads through the entire population
    theta = 1 - exp(-N * s / alpha)
    if rates:
        return (r0, r1, theta)
    # Complement of the cumulative distribution function of T, the time until
    # the first 2-mutant deme
    ccdf = lambda t: exp(-L*r0*theta*t + L*r0*theta/r1*(1-exp(-r1*t)))
    # 1 / (L*l0) + 1 / l1 is an upper bound for tau.
    # Assume that no probability mass lies beyond 100 times this value
    upper_lim = 100 * (1 / (L*r0*theta) + 1 / r1)
    return integrate.quad(ccdf, 0, upper_lim)[0]

################

def mse_freq(N, alpha, m, delta, mu_f, mu_b=0, upper_lim=None):
    """Find the MSE frequency of deleterious mutations in the metapop'n.

    N -- population size of a single deme
    alpha -- drift coefficient
    m -- migration rate
    delta -- deleterious selection coefficient
    mu_f -- forward mutation rate
    mu_b -- backward mutation rate (not implemented yet)

    Assumes an infinite metapopulation; that is, infinitely many demes of
    finite size of Ne (effective) haploid individuals per deme. 
    Assumes a single locus with forward mutation rate mu_f and backward
    mutation rate mu_b.

    Uses numerical integration instead of the hyp1f1 function
    """
    def avg_loc_freq(x):
        """The average mutant frequency within a deme.

        x -- the frequency of mutants among migrants

        Returns the mean of Wright's distribution given x

        Calculates by numerical integration using the trick described in
        Kimura, Maruyama, and Crow (1963) "The mutation load in small
        populations" to deal with possible singularities at the endpoints

        Can alternatively calculate the mean as 
        A * hyp1f1(1+A, 1 + A+B, -D) / ((A+B) * hyp1f1(A, A+B, -D))
        However, this fails for large values of N*m/alpha.

        """
        A = N / alpha * (m * x + mu_f)
        B = N / alpha * (m * (1 - x))
        D = N / alpha * delta
        f = lambda y: exp(-D * y)
        phi = lambda y: f(y) * y**(A-1) * (1-y)**(B-1)
        phi_m1 = lambda y: f(y) * y**(A) * (1-y)**(B-1)
        # integrate phi to find normalization constant
        I = integrate.quad(lambda y: phi(y) - y**(A-1) - exp(-D)*(1-y)**(B-1),
                0, 1)[0] + 1/A + exp(-D)/B
        C = 1 / I
        # find mean 
        I = integrate.quad(lambda y: phi_m1(y) - y**A - exp(-D)*y*(1-y)**(B-1),
                0, 1)[0] + 1/(A+1) + exp(-D)/(B*(B+1))
        m1 = C * I
        return m1
    lower_lim = mu_f / (10 * delta)
    # Get the MSE frequency as the migrant frequency x where the mean of
    # Wright's distribution equals x
    if upper_lim == None:
        xhat = optimize.newton(lambda x: avg_loc_freq(x) - x, lower_lim)
    else:
        xhat = optimize.brentq(lambda x: avg_loc_freq(x) - x, lower_lim, upper_lim)
    return xhat

def phi_stats(x, N, alpha, m, delta, mu_f=0):
    """Stats of the distribution phi

    x -- frequency of mutants among migrants
    N -- number of diploid individuals in a single deme
    alpha -- drift coefficient
    m -- migration rate
    delta -- deleterious selection coefficient
    mu_f -- forward mutation rate

    Assumes an infinite metapopulation; that is, infinitely many demes of
    finite size of Ne (effective) haploid individuals per deme. 
    Assumes a single locus with forward mutation rate mu_f and no back mutation.

    Currently inbreeding not implemented

    """
    A = N / alpha * (m * x + mu_f)
    B = N / alpha * (m * (1 - x))
    D = N / alpha * delta
    g = lambda y: exp(-D * y)
    phi = lambda y: g(y) * y**(A-1) * (1-y)**(B-1)
    phi_m1 = lambda y: g(y) * y**(A) * (1-y)**(B-1)
    phi_m2 = lambda y: g(y) * y**(A + 1) * (1-y)**(B-1)
    phi_m3 = lambda y: g(y) * y**(A + 2) * (1-y)**(B-1)
    # Normalization constant C
    I = integrate.quad(lambda y: phi(y) - y**(A-1) - exp(-D)*(1-y)**(B-1),
            0, 1)[0] + 1/A + exp(-D)/B
    C = 1 / I
    # First moment (mean)
    I = integrate.quad(lambda y: phi_m1(y) - y**A - exp(-D)*y*(1-y)**(B-1),
            0, 1)[0] + 1/(A+1) + exp(-D)/(B*(B+1))
    m1 = C * I
    # Second moment
    I = integrate.quad(lambda y: phi_m2(y) - y**(A+1) - exp(-D)*y**2*(1-y)**(B-1),
            0, 1)[0] + 1/(A+2) + exp(-D)*2/(B*(B+1)*(B+2))
    m2 = C * I
    # Third moment
    I = integrate.quad(lambda y: phi_m3(y) - y**(A+2) - exp(-D)*y**3*(1-y)**(B-1),
            0, 1)[0] + 1/(A+3) + exp(-D)*6/(B*(B+1)*(B+2)*(B+3))
    m3 = C * I
    # F=f2 and gamma=f3
    f2 = (m2 - m1**2)/(m1 * (1-m1))
    f3 = (m3 - m1**2 * (m1 + 3*f2*(1-m1))) / (m1*(1-m1)*(1-2*m1))
    return m1, m2, m3, f2, f3

def phi_stats_typeA(x, N, alpha, m, delta, mu_f=0):
    """Stats of the distribution phi for type A lineages only

    x -- frequency of mutants among migrants
    N -- number of diploid individuals in a single deme
    alpha -- drift coefficient
    m -- migration rate
    delta -- deleterious selection coefficient
    mu_f -- forward mutation rate

    Assumes an infinite metapopulation; that is, infinitely many demes of
    finite size of Ne (effective) haploid individuals per deme. 
    Assumes a single locus with forward mutation rate mu_f and no back mutation.

    Currently inbreeding not implemented

    """
    A = N / alpha * (m * x + mu_f)
    B = N / alpha * (m * (1 - x))
    D = N / alpha * delta
    phi = lambda y: exp(-D * y) * y**(A-1) * (1-y)**(B-1) - exp(-D)*(1-y)**(B-1) 
    phi_m1 = lambda y: exp(-D * y) * y**(A) * (1-y)**(B-1) - exp(-D)*y*(1-y)**(B-1)
    phi_m2 = lambda y: exp(-D * y) * y**(A + 1) * (1-y)**(B-1) - exp(-D)*y**2*(1-y)**(B-1)
    phi_m3 = lambda y: exp(-D * y) * y**(A + 2) * (1-y)**(B-1) - exp(-D)*y**3*(1-y)**(B-1)
    # Normalization constant C
    I = integrate.quad(lambda y: phi(y) - y**(A-1),
            0, 1)[0] + 1/A
    C = 1 / I
    # First moment (mean)
    I = integrate.quad(lambda y: phi_m1(y) - y**A,
            0, 1)[0] + 1/(A+1)
    m1 = C * I
    # Second moment
    I = integrate.quad(lambda y: phi_m2(y) - y**(A+1),
            0, 1)[0] + 1/(A+2)
    m2 = C * I
    # Third moment
    I = integrate.quad(lambda y: phi_m3(y) - y**(A+2),
            0, 1)[0] + 1/(A+3)
    m3 = C * I
    # F=f2 and gamma=f3
    f2 = (m2 - m1**2)/(m1 * (1-m1))
    f3 = (m3 - m1**2 * (m1 + 3*f2*(1-m1))) / (m1*(1-m1)*(1-2*m1))
    return m1, m2, m3, f2, f3



def feq_locally_deleterious(N, alpha, delta, m, approx=0, lintype=0):
    """Heuristic approximation for equilibrium Fst when N*delta/alpha >> 1.

    N -- deme size
    alpha -- drift parameter
    delta -- deleterious selection coefficient
    m -- migration rate, or an array of migration rates
    approx -- 0, 1, or 2; specify which approximation to use
    lintype -- If 0, account for lineages that do not and do fix locally; If 1
        or 2, only consider those that do not fix locally or do fix locally,
        respectively.
    """
    # scaled rate parameters
    D = N * delta / alpha
    M = N * m / alpha
    # fixation probabilities
    u01 = (delta / alpha) / (1 - exp(-D))
    u10 = (delta / alpha) / (exp(D) - 1)

    # First type: lineages that do not reach sizes $\gtrsim \alpha/\delta$
    # Choose from one of three approximations:
    #
    # Approximation 0: found by approximating distribution of local frequencies
    # as a gamma distribution. This approximation leads to an underestimate of F at migration rates 
    # $m^* \ll m \ll \alpha/N$
    if approx == 0:
        F1 = 1 / (M + D)
    # This approximation performs better, but overestimates F at some migration
    # rates. Justification is that it agrees with the more exact approximation
    # 2 in the limit M->0
    if approx == 1:
        F1 = 1 / (M + D - 1)
    # Best fitting approximation. Found by using moment closure method to solve
    # for the equilibrium
    if approx == 2:
        def fst_small(N, mu, d, m):
            Hs = 2*mu/d
            A = sqrt((m*N + d*N)**2 + 2*m*N + 1 - 2*N*d)
            Ht = mu/d / (2*m*N) * (3*m*N + 1 - N*d + A)
            return 1 - Hs / Ht
        Ne = N / (2 * alpha)
        F1 = fst_small(Ne, delta/1e2, delta, m) 
    # Expected weight of such a lineage
    W1 = 1 / (delta * (1 - F1))
    # Second type: lineages that fix locally
    # Fst for such lineages
    F2 = 1
    # Expected weight of such a lineage
    W2 = u10 / (m * u01)
    # If lintype>0, only consider the contributions from lineages 1 or 2
    if lintype == 1:
        W2 = 0
    elif lintype == 2:
        W1 = 0
    # Fst found by averaging over both types of lineages
    F = (F1 * W1 + F2 * W2) / (W1 + W2)
    return F


