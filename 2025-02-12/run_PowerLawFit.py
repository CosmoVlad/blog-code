import numpy as np
import matplotlib.pyplot as plt

import emcee
import corner

from scipy.special import logsumexp, erf
from multiprocessing import Pool


from new_utils import generate_plaw_population, Anorm, fac, log_bins

mmin = 1e+2
mmax = 1e+5

def init():
    global rng
    rng = np.random.default_rng()
init()


def detection_probability(mmm, sigma):
    
    corr = erf(np.log(mmax/mmm)/sigma/np.sqrt(2)) - erf(np.log(mmin/mmm)/sigma/np.sqrt(2))
    corr /= 2.
    
    return corr

def detection_probability_numeric(mmm, sigma, nsamples=200):

    log_samples = rng.multivariate_normal(mean=np.log(mmm), cov=np.diag(np.full_like(mmm, sigma**2)), size=nsamples)
    
    cond = np.logical_and(log_samples>np.log(mmin), log_samples<np.log(mmax))
    
    return np.sum(cond, axis=0)/nsamples

def detectable_fraction(alpha, sigma):

    dy_scaled = np.log(mmax/mmin)/sigma/np.sqrt(2)
    gamma = 1-alpha
    A = (mmax**gamma-mmin**gamma)/gamma

    return 1/2*erf(dy_scaled) + 1/2*np.exp((sigma*gamma)**2/2) / A / gamma\
                                * (mmax**gamma*erf(dy_scaled+sigma*gamma/np.sqrt(2)) - mmin**gamma*erf(dy_scaled-sigma*gamma/np.sqrt(2)))\
                                - 1/2*np.exp((sigma*gamma)**2/2) / A / gamma * erf(sigma*gamma/np.sqrt(2)) * (mmax**gamma+mmin**gamma)

def detectable_fraction_numeric(alpha, sigma, nsamples=200):

    pop = generate_plaw_population(nsamples, alpha, mmin, mmax, rng)

    return np.mean(detection_probability(pop, sigma))

def averaged(mm, alpha, sigma):

    gamma = alpha - 1

    return 1/2 * np.exp(1/2*(sigma*gamma)**2) * (
        erf(
            (np.log(mmax/mm) + sigma**2*gamma)/sigma/np.sqrt(2)
        )
        -
        erf(
            (np.log(mmin/mm) + sigma**2*gamma)/sigma/np.sqrt(2)
        )
    ) * mm**-alpha / Anorm(alpha, mmin, mmax)

def log_averaged_numeric(mm, alpha, sigma, nsamples=200):

    N = int(fac*nsamples)

    while True:

        data = rng.multivariate_normal(mean=np.log(mm), cov=np.diag(np.full_like(mm, sigma**2)), size=N)
        cond = np.logical_and(data>np.log(mmin), data<np.log(mmax))
        counts = np.sum(cond, axis=0)
        if np.all(counts > nsamples):
            break
        N = int(fac*N)

    return np.log(np.sum(cond*np.exp(-(alpha-1)*data), axis=0))- np.log(Anorm(alpha, mmin, mmax)) - np.log(N) - np.log(mm)
    


################################################ fiducial values

alpha = 1.3
sigma = 0.5

def parallel_run(size):

    size = int(size)
    N = int(fac*size)
    
    while True:
    
        init_data = generate_plaw_population(N, alpha, mmin, mmax, rng)
        data = np.array([rng.normal(loc=np.log(el), scale=sigma) for el in init_data])
        cond = np.logical_and(data>np.log(mmin), data<np.log(mmax))
        if np.sum(cond) > size:
            break
        N = int(fac*N)
    
    imprecise_pop = rng.choice(np.exp(data[cond]), size=size, replace=False)


    def log_likelihood(theta,num_events):
        
        alpha = theta
    
        if alpha > 0. and alpha < 2.:
            return - num_events*np.log(detectable_fraction(alpha,sigma)) + np.sum(np.log(averaged(imprecise_pop,alpha,sigma)))
            
        return -np.inf

    nwalkers = 32
    pos = np.array(
        [
            1 + 0.1*np.random.randn(nwalkers)
        ]
    ).T
    _, ndim = pos.shape


    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_likelihood, args = (size,),
    )
    sampler.run_mcmc(pos, 10000, progress=True);
    
    flat_samples = sampler.get_chain(discard=1000, thin=20, flat=True)
    
    return flat_samples[:,0],size


if __name__ == '__main__':

	import time as tm

	num_realizations = 48
	sizes = np.repeat(np.logspace(4,11,8, base=2), num_realizations)
	rng.shuffle(sizes)  # for more efficient parallelization

	result = []

	begin = tm.time()

	for size_chunk in [sizes]:

	    with Pool(initializer=init) as p:
	        
	        result_chunk = p.map(parallel_run, size_chunk)

	    result.append(result_chunk)

	test = []
	for el in result:
	    test = [*test, *el]

	all_samples = []
	all_sizes = []

	size_samples, sizes = zip(*sorted(test, key=lambda x: x[1]))
	all_sizes, counts = np.unique(sizes, return_counts=True)
	samples_per_realization = np.array(
	    np.split(
	        np.array(size_samples), indices_or_sections=np.cumsum(counts)[:-1]
	    )
	)

	print('It tooks {:.1f} seconds.'.format(tm.time()-begin))

	np.savez(
	    'PowerLaw_alpha={:.1f}_reals={:d}.npz'.format(alpha, num_realizations),
	    samples_per_realization=samples_per_realization,
	    all_sizes=all_sizes,
	    alpha=alpha
	)