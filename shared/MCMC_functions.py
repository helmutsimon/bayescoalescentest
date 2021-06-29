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
#from pymc3 import step_methods
from pymc3.step_methods.metropolis import Metropolis, CategoricalGibbsMetropolis
from pymc3.distributions.transforms import Transform, StickBreaking
from pymc3.theanof import floatX
from theano import config
import theano.tensor as tt
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
__copyright__ = "© Copyright 2021, Helmut Simon"
__license__ = "BSD-3"
__version__ = "0.1.0"
__maintainer__ = "Helmut Simon"
__email__ = "helmut.simon@anu.edu.au"
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


class StickBreaking_bv(Transform):
    """
    Transforms K - 1 dimensional simplex space (k values in [0,1] and that sum to 1) to a K - 1 vector of real values.
    This is a variant of the isometric logratio transformation:
    Adapted from PyMC3 StickBreaking transform, with inclusion of backward_val.
    """

    name = "stick_bv"

    forward = StickBreaking.forward
    backward = StickBreaking.backward
    forward_val = StickBreaking.forward_val

    def backward_val(self, y_):
        y = y_.T
        y = np.concatenate([y, -np.sum(y, 0, keepdims=True)])
        e_y = np.exp(y - np.max(y, 0, keepdims=True))
        x = e_y / np.sum(e_y, 0, keepdims=True)
        return floatX(x.T)

stick_bv = StickBreaking_bv()


def Lehmer_distribution(n):
    permutation = list()
    for i in range(n - 2):
        name = 'index_' + str(i)
        p = np.ones(n - i - 1) / (n - i - 1)
        cat_dist = Categorical(name, p)
        permutation.append(cat_dist)
    return permutation


def run_MCMC_Dirichlet(sfs, seq_mut_rate, sd_mut_rate, draws=50000, progressbar=False, order="random", cores=None,
                       tune=None, step=None, target_accept=0.9, concentration=1.0, use_start=True):
    """Define and run MCMC model for coalescent tree branch lengths using uniform (Dirichlet) prior."""
    config.compute_test_value = 'raise'
    n = len(sfs) + 1
    prior = concentration * np.ones(n - 1)
    j_n = np.diag(1 / np.arange(2, n + 1, dtype=np.float32))
    sfs = np.array(sfs)
    sfs = tt.as_tensor(sfs)
    seg_sites = sum(sfs)
    q_est = sfs + (seg_sites * .001)
    q_est = q_est / tt.sum(q_est)
    ttl_est = (seg_sites + 1) / seq_mut_rate
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
        #Create sequence of categorical distributions to sample Lehmer code representation.
        permutation = Lehmer_distribution(n)
        permutation_tt = tt.as_tensor(permutation)
        jmx = jmatrix(permutation_tt)

        probs = Dirichlet('probs', prior)
        q = tt.dot(jmx, probs.T)
        sfs_obs = Multinomial('sfs_obs', n=seg_sites, p=q, observed=sfs)

        total_length = Gamma('total_length', alpha=1, beta=1e-10)
        assert seq_mut_rate > sd_mut_rate, 'Mutation rate estimate must be greater than standard deviation.'
        mut_rate = Beta('mut_rate', mu=seq_mut_rate, sd=sd_mut_rate)
        mu = total_length * mut_rate
        seg_sites_obs = Poisson('seg_sites_obs', mu=mu, observed=seg_sites)

    with combined_model:
        step1 = Metropolis([probs, mut_rate, total_length])
        step2 = CategoricalGibbsMetropolis(permutation, order=order)
        if tune is None:
            tune = int(draws / 5)
        if use_start:
            start = {'total_length': ttl_est.eval(), 'probs': q_est.eval()}
        else:
            start = None
        if step == "metr":
            step = [step1, step2]
            trace = sample(draws, tune=tune, step=step,
                           progressbar=progressbar, return_inferencedata=False, start=start, cores=cores)
        else:
            step = step2
            trace = sample(draws, tune=tune, step=step, nuts={'target_accept':target_accept},
                           progressbar=progressbar, return_inferencedata=False, start=start, cores=cores)
    return combined_model, trace


def run_MCMC_mvn(sfs, mrate_lower, mrate_upper, mu, sigma, ttl_mu, ttl_sigma, draws=50000,
                 progressbar=False, order="random", cores=None, tune=None, step=None):
    """Define and run MCMC model for coalescent tree branch lengths using a multivariate normal prior."""
    config.compute_test_value = 'raise'
    n = len(sfs) + 1
    j_n = np.diag(1 / np.arange(2, n + 1, dtype=np.float32))
    sfs = np.array(sfs)
    sfs = tt.as_tensor(sfs)
    seg_sites = sum(sfs)
    q_est = sfs + (seg_sites * .001)
    q_est = q_est / tt.sum(q_est)
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

        mvn_sample = MvNormal('mvn_sample', mu=mu, cov=sigma, shape=(n - 2))
        simplex_sample = stick_bv.backward(mvn_sample)
        q = tt.dot(jmx, simplex_sample.T)
        sfs_obs = Multinomial('sfs_obs', n=seg_sites, p=q, observed=sfs)

        total_length = Gamma('total_length', mu=ttl_mu, sigma=ttl_sigma)
        mut_rate = Uniform('mut_rate', lower=mrate_lower, upper=mrate_upper)
        mu = total_length * mut_rate
        seg_sites_obs = Poisson('seg_sites_obs', mu=mu, observed=seg_sites)

    with combined_model:
        step1 = Metropolis([mvn_sample, mut_rate, total_length])
        step2 = CategoricalGibbsMetropolis(permutation, order=order)
        if step == "metr":
            step = [step1, step2]
        else:
            step = step2
        if tune is None:
            tune = int(draws / 5)
        start = {'mvn_sample': q_est.eval()}
        trace = sample(draws, tune=tune, step=step, progressbar=progressbar,
                       cores=cores, start=start, return_inferencedata=False)
    return combined_model, trace


def multiply_variates(trace, variable_name):
    """
    Multiply variates for relative branch length and total tree length to obtain variates for absolute
    branch length.
    variable_name is 'mvn_sample' for multivariate normal prior, 'probs' for flat Dirichlet prior.

    """
    vars_rel = [t[variable_name] for t in trace]
    vars_rel = np.array(vars_rel)
    if variable_name == 'mvn_sample':
        vars_rel = stick_bv.backward_val(vars_rel)
    size = vars_rel.shape[0]
    n0 = vars_rel.shape[1]
    j_n = np.diag(1 / np.arange(2, n0 + 2))
    vars_rel = j_n.dot(vars_rel.T)
    vars_TTL = [t['total_length'] for t in trace]
    vars_TTL = np.array(vars_TTL)
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

