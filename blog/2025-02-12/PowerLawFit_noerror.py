# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import numpy as np
import matplotlib.pyplot as plt

import emcee
import corner

from scipy.special import logsumexp, erf
from multiprocessing import Pool

MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rcdefaults()

#rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rc('text', usetex=True)


plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


# %matplotlib inline

# %%
# fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,4))

# ax.grid(True,linestyle=':',linewidth='1.')
# ax.xaxis.set_ticks_position('both')
# ax.yaxis.set_ticks_position('both')
# ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

# ax.set_xlabel('$x$')
# ax.set_ylabel('$y$')

#fig.tight_layout()
#fig.savefig('test.pdf')

# %% [markdown]
# ## A power-law fit
#
# Consider a power-law in masses
# $$
# f(m) = \frac{m^{-\alpha}}{A(\alpha,m_{\rm min},m_{\rm max})}\,, \qquad m\in[m_{\rm min},m_{\rm max}]\,, 
# $$
# $$
# A(\alpha,m_{\rm min},m_{\rm max}) = \int\limits_{m_{\rm min}}^{m_{\rm max}}{{\rm d}m\,m^{-\alpha}}\,.
# $$
#
# We want to infer $\alpha$ given a population of $N$ masses.
#
# Here we assume that there are **no selection effects**, and we consider the case when:
# - masses of the individual events are known exactly.
#

# %% [markdown]
# ### Population

# %%
from new_utils import generate_plaw_population, Anorm, fac, log_bins

def init():
    global rng
    rng = np.random.default_rng()
init()


# %%
# fiducial values
alpha = 1.3
mmin = 1e+2
mmax = 1e+5


# population
size = 1500
pop = generate_plaw_population(size, alpha, mmin, mmax, rng)


# plotting
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,5))

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)

ax.set_xscale('log')
ax.set_yscale('log')

bins = log_bins(mmin, mmax, nbins=20)
ax.hist(
    pop, 
    bins=bins, density=True, label='pop ($N={:d}$)'.format(size)
)

mm = log_bins(mmin, mmax, nbins=100)
ax.loglog(
    mm, mm**-alpha/Anorm(alpha, mmin, mmax), 
    label='pop model'
)

ax.set_xlabel('$m$ $(M_\\odot)$')
ax.set_ylabel('PDF')

ax.legend()

fig.tight_layout()


# %% [markdown]
# ### No measurement errors
#
# Here we assume that individual masses in the population are measured perfectly. Then, the detection fraction is simply
# $$
# \mathcal{F} = 1\,,
# $$
# and
# \begin{eqnarray}
# \log{p(\alpha|D)} &=& \mbox{const}  -N\log{\mathcal{F}} - \alpha\sum\limits_{i=1}^N{f(m_i)}\,, \\
# &=& \mbox{const} -N\log{A(\alpha,m_{\rm min},m_{\rm max})} - \alpha\sum\limits_{i=1}^N{\log{m_i}}\,.
# \end{eqnarray}
#
# Note that the ends of the mass range can be interpreted as a selection effect. If we only consider data that falls into a narrower range $[m_{\rm min}^\prime, m_{\rm max}^\prime]\subset [m_{\rm min}, m_{\rm max}]$, the log-likelihood has the same form except for the substitutions $m_{\rm min}\to m_{\rm min}^\prime$, $m_{\rm max}\to m_{\rm max}^\prime$, $N\to N^\prime$, where $N'$ refers to data that follows into the narrower range.

# %% [markdown]
# #### Single MCMC run ($N=1500$)

# %%
def log_likelihood(theta,num_events):
    
    alpha = theta

    if alpha > 0. and alpha < 2.:
        return - num_events*np.log(Anorm(alpha, mmin, mmax))\
                - alpha*np.sum(np.log(pop)) 
        
    return -np.inf


# %%
nwalkers = 32
labels = ["$\\alpha$"]
pos = np.array(
    [
        1 + 0.1*np.random.randn(nwalkers)
    ]
).T
_, ndim = pos.shape


sampler = emcee.EnsembleSampler(
    nwalkers, ndim, log_likelihood, args = (size,)
)
sampler.run_mcmc(pos, 10000, progress=True);

# %%
every = 4

fig, axes = plt.subplots(ndim, figsize=(10, 3), sharex=True)

if ndim == 1:
    axes = (axes,)
    
samples = sampler.get_chain()
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, ::every, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    #ax.yaxis.set_label_coords(-0.1, 0.5)

axes[-1].set_xlabel("step number");

flat_samples = sampler.get_chain(discard=1000, thin=20, flat=True)

fig = corner.corner(
            flat_samples,
            labels=labels, truths=[alpha], quantiles=[0.16, 0.5, 0.84], show_titles=True
        );

# %%
fig,ax = plt.subplots(ncols=1, nrows=1, figsize=(6,5))

ax.grid(True,linestyle=':',linewidth='1.')
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params('both',length=3,width=0.5,which='both',direction = 'in',pad=10)


bins = np.linspace(1,2,50)
ax.hist(
    flat_samples[:,0], 
    bins=bins, density=True, 
    alpha=0.5, label='posterior'
)

ax1 = ax.twinx()
ax1.set_ylim(0,1)

pp = np.linspace(*ax1.get_ylim(), 100)
aa = np.full_like(pp, alpha)
ax1.plot(aa,pp, c='black', ls='dashed')


ax.set_xlabel('$\\alpha$')
ax.set_ylabel('PDF')

ax.legend()

fig.tight_layout()


# %% [markdown]
# #### Multiple MCMC runs (for different population sizes $N$ and multiple realizations for each of those $N$)

# %%
def parallel_run_noerror(size):
    
    size = int(size)
    
    pop = generate_plaw_population(size, alpha, mmin, mmax, rng)


    def log_likelihood(theta,num_events):

        alpha = theta

        if alpha > 0. and alpha < 2.:
            
            return - num_events*np.log(Anorm(alpha, mmin, mmax))\
                    + np.sum(
                -alpha*np.log(pop)
            ) 

        return -np.inf

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

# %%
# %%time

## MULTIPLE RUNS

num_realizations = 6
sizes = np.repeat(np.logspace(4,11,8, base=2), num_realizations)
rng.shuffle(sizes)  # for more efficient parallelization

result = []

for size_chunk in np.split(sizes, num_realizations):

    with Pool(initializer=init) as p:
        
        result_chunk = p.map(parallel_run_noerror, size_chunk)

    result.append(result_chunk)

# %%
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

# np.savez(
#     'PowerLaw_noerror_alpha={:.1f}_reals={:d}.npz'.format(alpha, num_realizations),
#     samples_per_realization=samples_per_realization,
#     all_sizes=all_sizes,
#     alpha=alpha
# )

# %%
