# coding=utf-8

"""
This file contains MCMC and associated routines for Bayesian inference of coalescence times.

"""

import os, sys
from pymc3.distributions.multivariate import Multinomial, MvNormal, Dirichlet
from pymc3.distributions.discrete import Categorical, Poisson
from pymc3.distributions.continuous import Gamma, Beta
from pymc3 import *
from pymc3.model import Model
from pymc3.sampling import sample
from pymc3.step_methods.metropolis import Metropolis
from pymc3.distributions.transforms import Transform
from pymc3.step_methods.arraystep import ArrayStep , metrop_select
import pymc3 as pm
from pymc3 import CategoricalGibbsMetropolis
import arviz
from arviz import summary
from pymc3.step_methods.metropolis import sample_except
from pymc3.theanof import floatX
import theano
from theano import config
import theano.tensor as tt
from theano.compile.ops import as_op
import numpy as np
from bisect import bisect
from scipy.special import binom, factorial, gammaln
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-1])
sys.path.append(projdir)

import tree_matrix_computation

def get_ERM_matrix(n):
    ERM_matrix = np.zeros((n - 1, n - 1))
    for m in range(n - 1):
        for k in range(n - 1):
            ERM_matrix[m, k] = (k + 2) * binom(n - m - 2, k) / binom(n - 1, k + 1)
    return ERM_matrix


def get_tree_details(n, sfs, mx_details):
    tree_matrices = mx_details[0][n]
    tree_matrices = [m.T for m in tree_matrices]
    tree_probabilities = mx_details[1][n]
    tree_probabilities = np.array(tree_probabilities)

    # We reduce the space of tree matrices to only consider matrices compatible with the data.
    indices = list(np.arange(len(tree_matrices)))
    for ix, mx in enumerate(tree_matrices):
        for i in range(n - 1):
            if (not np.any(mx[i, :])) and sfs[i]:
                if ix in indices:
                    indices.remove(ix)
    print('Number of matrices used: '.ljust(25), len(indices), ' out of ', len(tree_matrices))
    tree_matrices = [tree_matrices[ix] for ix in indices]
    tree_matrices = np.array(tree_matrices)
    tree_probabilities = np.array([tree_probabilities[ix] for ix in indices])
    tree_probabilities = tree_probabilities / np.sum(tree_probabilities)

    # If only one valid matrix, use a dummy categorical distribution that always returns the same matrix.
    if len(indices) == 1:
        tree_matrices = [tree_matrices[indices[0]], tree_matrices[indices[0]]]
        tree_matrices = np.array(tree_matrices)
        tree_probabilities = np.array([0.5, 0.5])
    return tree_matrices, tree_probabilities


class StickBreaking_bv(Transform):
    """
    Transforms K - 1 dimensional simplex space (k values in [0,1] and that sum to 1) to a K - 1 vector of real values.
    This is a variant of the isometric logratio transformation:
    Adapted from PyMC3 StickBreaking transform, with inclusion of backward_val.
    """

    name = "stick_bv"

    def forward(self, x_):
        x = x_.T
        n = x.shape[0]
        lx = tt.log(x)
        shift = tt.sum(lx, 0, keepdims=True) / n
        y = lx[:-1] - shift
        return floatX(y.T)

    def forward_val(self, x_, point=None):
        x = x_.T
        n = x.shape[0]
        lx = np.log(x)
        shift = np.sum(lx, 0, keepdims=True) / n
        y = lx[:-1] - shift
        return floatX(y.T)

    def backward(self, y_):
        y = y_.T
        y = tt.concatenate([y, -tt.sum(y, 0, keepdims=True)])
        e_y = tt.exp(y - tt.max(y, 0, keepdims=True))
        x = e_y / tt.sum(e_y, 0, keepdims=True)
        return floatX(x.T)

    def backward_val(self, y_):
        y = y_.T
        y = np.concatenate([y, -np.sum(y, 0, keepdims=True)])
        e_y = np.exp(y - np.max(y, 0, keepdims=True))
        x = e_y / np.sum(e_y, 0, keepdims=True)
        return floatX(x.T)

stick_bv = StickBreaking_bv()


def run_MCMC_mvn(sfs, seq_mut_rate, sd_mut_rate, mx_details, mu, sigma, ttl_mu, ttl_sigma, draws=50000, progressbar=False):
    """
    Define and run MCMC model for coalescent tree branch lengths using multivariate normal prior..

    """

    n = len(sfs) + 1
    j_n = np.diag(1 / np.arange(2, n + 1))
    tree_matrices, tree_probabilities = get_tree_details(n, sfs, mx_details)
    sfs = np.array(sfs)
    seg_sites = sum(sfs)

    with Model() as combined_model:

        tree_index = Categorical('tree_index', tree_probabilities)
        tree_matrices = theano.shared(tree_matrices)
        jmx = tree_matrices[tree_index].dot(j_n)

        mvn_sample = MvNormal('mvn_sample', mu=mu, cov=sigma, shape=(n - 2))
        simplex_sample = stick_bv.backward(mvn_sample)
        conditional_probs = tt.dot(jmx, simplex_sample.T)
        sfs_obs = Multinomial('sfs_obs', n=seg_sites, p=conditional_probs, observed=sfs)

        total_length1 = Gamma('total_length', mu=ttl_mu, sigma=ttl_sigma)
        assert seq_mut_rate > sd_mut_rate, 'Mutation rate estimate must be greater than standard deviation.'
        mut_rate = Beta('mut_rate', mu=seq_mut_rate, sd=sd_mut_rate)
        total_length = total_length1 * mut_rate
        seg_sites_obs = Poisson('seg_sites_obs', mu=total_length, observed=seg_sites)

    with combined_model:
        step = Metropolis(
            [mvn_sample, mut_rate, total_length])
        tune = int(draws / 5)
        trace = sample(draws, tune=tune, step=step, progressbar=progressbar)
    print(summary(trace))
    return combined_model, trace


def run_MCMC_Dirichlet(sfs, seq_mut_rate, sd_mut_rate, mx_details, draws=50000, progressbar=False):
    """
    Define and run MCMC model for coalescent tree branch lengths using flat Dirichlet prior.
    """

    n = len(sfs) + 1
    prior = np.ones(n - 1)
    j_n = np.diag(1 / np.arange(2, n + 1))
    tree_matrices, tree_probabilities = get_tree_details(n, sfs, mx_details)
    sfs = np.array(sfs)
    seg_sites = sum(sfs)

    with Model() as combined_model:
        tree_index = Categorical('tree_index', tree_probabilities)
        tree_matrices = theano.shared(tree_matrices)
        jmx = tree_matrices[tree_index].dot(j_n)

        probs = Dirichlet('probs', prior)
        conditional_probs = tt.dot(jmx, probs.T)
        sfs_obs = Multinomial('sfs_obs', n=seg_sites, p=conditional_probs, observed=sfs)

        total_length1 = Gamma('total_length', alpha=1, beta=1e-10)
        assert seq_mut_rate > sd_mut_rate, 'Mutation rate estimate must be greater than standard deviation.'
        mut_rate = Beta('mut_rate', mu=seq_mut_rate, sd=sd_mut_rate)
        total_length = total_length1 * mut_rate
        seg_sites_obs = Poisson('seg_sites_obs', mu=total_length, observed=seg_sites)

    with combined_model:
        step = Metropolis(
            [probs, mut_rate, total_length])
        tune = int(draws / 5)
        trace = sample(draws, tune=tune, step=step, progressbar=progressbar)
    print(summary(trace))
    return combined_model, trace


class CategoricalGibbsMetropolisX(ArrayStep):
    """A modification of the pymc3 CategoricalGibbsMetropolis step method allowing use of the CategoricalFactorial
    distribution.
    """
    name = "categorical_gibbs_metropolis"

    def __init__(self, vars, proposal="uniform", order="random", model=None):

        model = pm.modelcontext(model)
        vars = pm.inputvars(vars)

        dimcats = []

        for v in vars:
            distr = getattr(v.distribution, "parent_dist", v.distribution)
            k = factorial(distr.k - 1, exact=True)  # Categorical distribution has (k-1)! categories
            start = len(dimcats)
            dimcats += [(dim, k) for dim in range(start, start + v.dsize)]

        if order == "random":
            self.shuffle_dims = True
            self.dimcats = dimcats
        else:
            if sorted(order) != list(range(len(dimcats))):
                raise ValueError("Argument 'order' has to be a permutation")
            self.shuffle_dims = False
            self.dimcats = [dimcats[j] for j in order]

        if proposal == "uniform":
            self.astep = self.astep_unif
        else:
            raise ValueError("Argument 'proposal' should be 'uniform'")

        super().__init__(vars, [model.fastlogp])

    astep_unif = CategoricalGibbsMetropolis.astep_unif
    astep_prop = CategoricalGibbsMetropolis.astep_prop
    metropolis_proportional = CategoricalGibbsMetropolis.metropolis_proportional


class CategoricalFactorial(Discrete):
    """A uniform dicrete (categorical) distribution on (k-1)! values."""

    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.factk = factorial(k - 1, exact=True)
        self.log_p = -gammaln(k)

    def random(self, point=None, size=None):
        k = self.k
        factk = self.factk
        return np.random.choice(factk)

    def logp(self, value):
        return self.log_p


def run_MCMC_Dirichlet_ordered(sfs, seq_mut_rate, sd_mut_rate, draws=50000, progressbar=False):
    """
    Define and run MCMC model for coalescent tree branch lengths using flat Dirichlet prior.
    Matrices are sampled by sampling unordered partitions, allowing sample sizes up to 20.
    """

    config.compute_test_value = 'raise'
    n = len(sfs) + 1
    prior = np.ones(n - 1)
    j_n = np.diag(1 / np.arange(2, n + 1, dtype=np.float32))
    sfs = np.array(sfs)
    sfs = tt.as_tensor(sfs)
    seg_sites = sum(sfs)

    def infer_matrix_shape(node, input_shapes):
        return [(n - 1, n - 1)]

    @as_op(itypes=[tt.lscalar], otypes=[tt.fmatrix], infer_shape=infer_matrix_shape)
    def jmatrix(i):
        """Return tree matrix for given integer"""
        f = tree_matrix_computation.factorize(i, n)
        mx = tree_matrix_computation.derive_tree_matrix(f)
        mx = np.reshape(mx, (n - 1, n - 1))
        mx = np.transpose(mx)
        jmx = mx @ j_n
        jmx = jmx.astype('float32')
        return jmx

    with Model() as combined_model:
        ordered_tree_ix = CategoricalFactorial('ordered_tree_ix', n, dtype='int64', testval=0)
        jmx = jmatrix(ordered_tree_ix)

        probs = Dirichlet('probs', prior)
        conditional_probs = tt.dot(jmx, probs.T)
        sfs_obs = Multinomial('sfs_obs', n=seg_sites, p=conditional_probs, observed=sfs)

        total_length1 = Gamma('total_length', alpha=1, beta=1e-10)
        assert seq_mut_rate > sd_mut_rate, 'Mutation rate estimate must be greater than standard deviation.'
        mut_rate = Beta('mut_rate', mu=seq_mut_rate, sd=sd_mut_rate)
        total_length = total_length1 * mut_rate
        seg_sites_obs = Poisson('seg_sites_obs', mu=total_length, observed=seg_sites)

    with combined_model:
        step = [Metropolis([probs, mut_rate, total_length]), \
                CategoricalGibbsMetropolisX([ordered_tree_ix])]
        tune = int(draws / 5)
        trace = sample(draws, tune=tune, step=step, progressbar=progressbar)
    print(summary(trace))
    return combined_model, trace


def run_MCMC_mvn_ordered(sfs, seq_mut_rate, sd_mut_rate, mu, sigma, ttl_mu, ttl_sigma, draws=50000, progressbar=False):
    """
    Define and run MCMC model for coalescent tree branch lengths using flat Dirichlet prior.
    Matrices are sampled by sampling unordered partitions, allowing sample sizes up to 20.
    """

    config.compute_test_value = 'raise'
    n = len(sfs) + 1
    prior = np.ones(n - 1)
    j_n = np.diag(1 / np.arange(2, n + 1, dtype=np.float32))
    sfs = np.array(sfs)
    sfs = tt.as_tensor(sfs)
    seg_sites = sum(sfs)

    def infer_matrix_shape(node, input_shapes):
        return [(n - 1, n - 1)]

    @as_op(itypes=[tt.lscalar], otypes=[tt.fmatrix], infer_shape=infer_matrix_shape)
    def jmatrix(i):
        """Return tree matrix for given integer"""
        f = tree_matrix_computation.factorize(i, n)
        mx = tree_matrix_computation.derive_tree_matrix(f)
        mx = np.reshape(mx, (n - 1, n - 1))
        mx = np.transpose(mx)
        jmx = mx @ j_n
        jmx = jmx.astype('float32')
        return jmx

    with Model() as combined_model:
        ordered_tree_ix = CategoricalFactorial('ordered_tree_ix', n, dtype='int64', testval=0)
        jmx = jmatrix(ordered_tree_ix)

        mvn_sample = MvNormal('mvn_sample', mu=mu, cov=sigma, shape=(n - 2))
        simplex_sample = stick_bv.backward(mvn_sample)
        conditional_probs = tt.dot(jmx, simplex_sample.T)
        sfs_obs = Multinomial('sfs_obs', n=seg_sites, p=conditional_probs, observed=sfs)

        total_length1 = Gamma('total_length', alpha=1, beta=1e-10)
        assert seq_mut_rate > sd_mut_rate, 'Mutation rate estimate must be greater than standard deviation.'
        mut_rate = Beta('mut_rate', mu=seq_mut_rate, sd=sd_mut_rate)
        total_length = total_length1 * mut_rate
        seg_sites_obs = Poisson('seg_sites_obs', mu=total_length, observed=seg_sites)

    with combined_model:
        step = [Metropolis([mvn_sample, mut_rate, total_length]),
                CategoricalGibbsMetropolisX([ordered_tree_ix])]
        tune = int(draws / 5)
        trace = sample(draws, tune=tune, step=step, progressbar=progressbar)
    print(summary(trace))
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
            sns.kdeplot(row, color=col, label=label, bw='scott', gridsize=500)
        plt.title(title)
        plt.xlabel('Generations')
        plt.ylabel('Frequency', labelpad=25)
        plt.xlim([0, xlim])
        plt.ylim([0, ylim])
        ymax = plt.gca().get_ylim()[1]
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

