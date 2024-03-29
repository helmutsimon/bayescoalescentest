# coding=utf-8

"""
This file contains MCMC and associated routines for Bayesian inference of coalescence times.

Notes:
    1. Multiprocessing does not work on all systems. Ubuntu 18.04.5 yes, MacOS no.
"""

from pymc3.distributions.multivariate import Multinomial, MvNormal, Dirichlet
from pymc3.distributions.discrete import Poisson, Categorical
from pymc3.distributions.continuous import Gamma, Beta, Uniform
from pymc3.model import Model
from pymc3.sampling import sample
from pymc3.step_methods.metropolis import Metropolis, CategoricalGibbsMetropolis
from pymc3.distributions.transforms import StickBreaking
from theano import config, function
import theano.tensor as tt
import arviz
import xarray as xr
from theano.compile.ops import as_op
import numpy as np
import functools
from bisect import bisect
from more_itertools import locate
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


__author__ = "Helmut Simon"
__copyright__ = "© Copyright 2023, Helmut Simon"
__license__ = "BSD-3"
__version__ = "0.5.0"
__maintainer__ = "Helmut Simon"
__email__ = "hsimon@bigpond.net.au"
__status__ = "Test"


@functools.lru_cache(maxsize=512)
def replace_zero(s, j):
    """Replace jth 0 in string s with a 1. First character is given by j=0 etc."""
    i = list(locate(s, lambda x: x == '0'))[j]
    return s[:i] + '1' + s[i + 1:]


@functools.lru_cache(maxsize=512)
def make_row(s):
    """Convert string of zeros (+) and ones (,) to matrix row, i.e. counting partitions by size."""
    c = re.split('1', s)
    return [int(len(x) + 1) for x in c]


def derive_tree_matrix(f):
    """Derive tree matrix from the list f. The first element of f is an integer in [0, n-2], the second
    in [0, n-3] and so on until [0, 1] i.e. Lehmer code."""
    n = len(f) + 2
    s = '0' * (n - 1)
    result = list()
    for j in f:
        s = replace_zero(s, j)
        orow = make_row(s)
        urow = np.bincount(orow, minlength=n + 1)
        urow = urow[1:-1]
        result.append(urow)
    lastrow = [n] + (n - 2) * [0]
    result.append(lastrow)
    mx = np.stack(result, axis=0)
    return mx


def Lehmer_distribution(n):
    permutation = list()
    for i in range(n - 2):
        name = 'index_' + str(i)
        p = np.ones(n - i - 1) / (n - i - 1)
        cat_dist = Categorical(name, p)
        permutation.append(cat_dist)
    return permutation


#Define transforms for MVN (informative) prior
xf = tt.matrix()
out = StickBreaking().forward(xf)
forward_val = function([xf], out)

xb = tt.matrix()
out = StickBreaking().backward(xb)
backward_val = function([xb], out)


def sample_Dirichlet_function(prior):
    def sample_Dirichlet_prior():
        probs = Dirichlet('probs', prior)
        return probs

    return sample_Dirichlet_prior


def sample_MVN_function(n, mu, sigma):
    def sample_MVN_prior():
        mvn_sample = MvNormal('mvn_sample', mu=mu, cov=sigma, shape=(n - 2))
        simplex_sample = StickBreaking().backward(mvn_sample)
        return simplex_sample

    return sample_MVN_prior


def unfolded_probabilities(q0):
    return q0


def probabilities_function_folded(n):
    def folded_probabilities(q0):
        x = q0 + np.flip(q0)
        pol = ((n - 1) % 2) + 1
        ul = (int((n - 1) / 2) + pol - 1)
        pfold = x[: ul]
        mask = np.ones(ul)
        mask[-1] = 1. / pol
        mask = tt.as_tensor(mask)
        return np.multiply(pfold, mask)

    return folded_probabilities


def sample_mutation_rate_Dirichlet_function(seq_mut_rate, sd_mut_rate):
    def sample_mutation_rate_Dirichlet():
        return Beta('mut_rate', mu=seq_mut_rate, sd=sd_mut_rate)

    return sample_mutation_rate_Dirichlet


def sample_mutation_rate_MVN_function(mrate_lower, mrate_upper):
    def sample_mutation_rate_MVN():
        return Uniform('mut_rate', lower=mrate_lower, upper=mrate_upper)

    return sample_mutation_rate_MVN

def sample_ttl_Dirichlet_function():
    def sample_ttl_Dirichlet():
        return Gamma('total_length', alpha=1, beta=1e-10)

    return sample_ttl_Dirichlet


def sample_ttl_MVN_function(ttl_mu, ttl_sigma):
    def sample_ttl_MVN():
        return Gamma('total_length', mu=ttl_mu, sigma=ttl_sigma)

    return sample_ttl_MVN


def Thomson_estimate(sfs, mrate):
    """
    Estimate TMRCA from site frequency spectrum (SFS). The SFS must be unfolded
    :param
    sfs: numpy.ndarray
        An unfolded site frequency spectrum
    mrate: float
        Sequence mutation rate per generation
    :returns
    numpy.float64
        Thomson estimate of TMRCA and standard deviation
    numpy.float64
        standard deviation of Thomson estimate.
    """

    n = len(sfs) + 1
    n_seq = np.arange(1, n)
    thom = np.sum(sfs * n_seq) / (n * mrate)
    var_thom = np.sum(sfs * n_seq * n_seq) / (2 * n * mrate) ** 2
    return thom, np.sqrt(var_thom)

def run_MCMC(n, sfs, variable_name, seq_mut_rate=None, sd_mut_rate=None,
             mrate_lower=None, mrate_upper=None, mu=None, sigma=None, ttl_mu=None, ttl_sigma=None,
             folded=False, draws=50000, progressbar=False, order="random",
             cores=None, tune=None, target_accept=0.9, return_inferencedata=True, concentration=1.0):
    """Define and run MCMC model for coalescent tree branch lengths. This function allows the following options:
       - uninformative (Dirichlet, given by setting variable-name parameter to 'probs') vs model-based
         (MVN, given by setting variable-name parameter to 'mvn_sample') prior distributions; and
       - data is folded (set folded=True) or unfolded SFS (default)."""
    config.compute_test_value = 'raise'
    if folded:
        print('Folded SFS.')
        assert len(sfs) == int(n / 2), 'n inconsistent with length of folded sfs'
        allele_probabilities = probabilities_function_folded(n)
    else:
        print('Unfolded SFS.')
        assert len(sfs) == n - 1, 'n inconsistent with length of unfolded sfs'
        allele_probabilities = unfolded_probabilities
    j_n = np.diag(1 / np.arange(2, n + 1, dtype=np.float32))
    sfs = np.array(sfs)
    sfs = tt.as_tensor(sfs)
    seg_sites = sum(sfs)
    if variable_name == 'probs':  # Dirichlet prior
        prior = concentration * np.ones(n - 1)
        sample_prior = sample_Dirichlet_function(prior)
        assert seq_mut_rate > sd_mut_rate, 'Mutation rate estimate must be greater than standard deviation.'
        sample_mutation_rate = sample_mutation_rate_Dirichlet_function(seq_mut_rate, sd_mut_rate)
        sample_ttl = sample_ttl_Dirichlet_function()
        ttl_est = (seg_sites + 1) / seq_mut_rate
        start = {'total_length': ttl_est.eval()}
    else:
        sample_prior = sample_MVN_function(n, mu, sigma)
        sample_mutation_rate = sample_mutation_rate_MVN_function(mrate_lower, mrate_upper)
        sample_ttl = sample_ttl_MVN_function(ttl_mu, ttl_sigma)
        start = {'total_length': ttl_mu}

    if order == "inc":
        order = np.arange(n - 2)
    elif order == "dec":
        order = np.arange(n - 2)
        order = np.flip(order)
    else:
        order = "random"

    def infer_matrix_shape(fgraph, node, input_shapes):
        return [(n - 1, n - 1)]

    @as_op(itypes=[tt.lvector], otypes=[tt.fmatrix], infer_shape=infer_matrix_shape)
    def jmatrix(i):
        """Derive tree matrix from Lehmer code representation."""
        mx = derive_tree_matrix(i)
        mx = np.reshape(mx, (n - 1, n - 1))
        mx = np.transpose(mx)
        jmx = mx.dot(j_n)
        jmx = jmx.astype('float32')
        return jmx

    with Model() as combined_model:
        # Create sequence of categorical distributions to sample Lehmer code representation.
        permutation = Lehmer_distribution(n)
        permutation_tt = tt.as_tensor(permutation)
        jmx = jmatrix(permutation_tt)

        probs = sample_prior()
        q0 = tt.dot(jmx, probs.T)
        q = allele_probabilities(q0)

        sfs_obs = Multinomial('sfs_obs', n=seg_sites, p=q, observed=sfs)

        total_length = sample_ttl()
        mut_rate = sample_mutation_rate()
        mu = total_length * mut_rate
        seg_sites_obs = Poisson('seg_sites_obs', mu=mu, observed=seg_sites)

    with combined_model:
        step = CategoricalGibbsMetropolis(permutation, order=order)
        if tune is None:
            tune = int(draws / 5)
        trace = sample(draws, tune=tune, step=step, nuts={'target_accept': target_accept},
                    progressbar=progressbar, return_inferencedata=return_inferencedata, start=start, cores=cores)
    # Transform MVN variates back into simplex and include in InferenceData object (trace)
    if variable_name == 'mvn_sample':
        chaindim      = trace.posterior.dims['chain']
        drawdim       = trace.posterior.dims['draw']
        mvn_sampledim = trace.posterior.dims['mvn_sample_dim_0']
        y = np.reshape(trace.posterior.mvn_sample.data, (chaindim * drawdim, mvn_sampledim))
        z = backward_val(y)
        w = np.reshape(z, (chaindim, drawdim, mvn_sampledim + 1))
        pr = xr.DataArray(data=w, dims=({'chain': chaindim, 'draw': drawdim, 'probs_dim_0': mvn_sampledim + 1}))
        trace.posterior['probs'] = pr
    return combined_model, trace


def multiply_variates(stacked):
    """
    Multiply variates for relative branch length and total tree length to obtain variates for absolute branch length.
    variable_name is 'mvn_sample' for multivariate normal prior, 'probs' for flat Dirichlet prior.
    """
    vars_rel = stacked['probs'].values.T
    size = vars_rel.shape[0]
    n0 = vars_rel.shape[1]
    j_n = np.diag(1 / np.arange(2, n0 + 2))
    vars_rel = j_n.dot(vars_rel.T)
    vars_TTL = stacked['total_length'].values
    assert size == len(vars_TTL), 'Mismatch between number of draws for rel branch lengths and TTL.'
    vars_TTL.shape = (size, 1)
    vars_TTL3 = np.repeat(vars_TTL, n0, axis=1).T
    branch_vars = np.multiply(vars_rel, vars_TTL3)
    return branch_vars


def print_pds(pdfname, variates, labels, truevalues=None, savepdf=True, properties=dict(),
              title=None, xlim=None, ylim=None, thom=None):
    """
    Print posterior distributions as pdf.

    """
    n = variates.shape[0] + 1
    sns.set_style("whitegrid")
    cols = sns.husl_palette(n_colors=n - 1, s=0.9, l=0.6)
    with PdfPages(pdfname) as pdf:
        fig = plt.figure(figsize=(15, 6))
        for row, col, label in zip(variates, cols, labels):
            sns.kdeplot(row, color=col, label=label, bw_adjust=5, gridsize=500)
        plt.title(title)
        plt.legend(title='Coalescent event sequence', title_fontsize=14, fontsize=14)
        plt.xlabel('Generations', fontsize=14)
        plt.ylabel('Probability density', fontsize=14, labelpad=25)
        plt.xlim([0, xlim])
        plt.ylim([0, ylim])
        ymax = plt.gca().get_ylim()[1]
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0,0))
        plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
        if truevalues is not None:
            plt.vlines(truevalues, 0, ymax, colors=cols, linestyles='dashed')
        if thom is not None:
            plt.vlines(thom, 0, ymax, colors='black', linestyles='dotted')
        d = pdf.infodict()
        for key in properties:
            d[key] = properties[key]
        if savepdf:
            pdf.savefig(fig, bbox_inches='tight')
    return fig


def ancestral_process(brvars, ntimes, tlim):
    """
    Calculate ancestral process given branch length variates, that is probability of k coalescent points having
    occurred prior to given range of time points.

    brvars: np.array
        Array of branch variates output by MCMC.
    ntimes: integer
        Number of intervals
    tlim: float
        maximum time (approx. TMRCA)

    Returns
    -------
    numpy.array
        Probabilities of coalescence times by time interval

    """
    n = brvars.shape[0] + 1
    anc_proc = np.zeros((n, ntimes))
    r = tlim / ntimes
    branches_rev = np.flipud(brvars)
    coal_times = np.cumsum(branches_rev, axis=0)
    draws = coal_times.shape[1]                             # number of MCMC variates
    for var_ix in range(draws):
        for t in range(ntimes):                             # iterating over time intervals
            k = bisect(coal_times[:,var_ix], t * r)
            anc_proc[k, t] += 1
    anc_proc = anc_proc / draws
    return anc_proc

