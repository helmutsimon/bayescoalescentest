# coding=utf-8

"""
This file contains MCMC and associated routines for Bayesian inference of coalescence times.

"""

from pymc3 import *
from pymc3.distributions.multivariate import Multinomial
from pymc3.distributions.discrete import Categorical
from pymc3.distributions.dist_math import bound, logpow
from pymc3.distributions.special import gammaln
import numpy as np
from bisect import bisect
from scipy.special import binom
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


def get_ERM_matrix(n):
    ERM_matrix = np.zeros((n - 1, n - 1))
    for m in range(n - 1):
        for k in range(n - 1):
            ERM_matrix[m, k] = (k + 2) * binom(n - m - 2, k) / binom(n - 1, k + 1)
    return ERM_matrix


def run_MCMC(sfs, seq_mut_rate, sd_mut_rate, mx_details, draws=50000, prior=None, alpha=None, beta=None, progressbar=False):
    """
    Define and run MCMC model for coalescent tree branch lengths.

    """

    n = len(sfs) + 1
    if prior is None:
        prior = np.ones(n - 1) / (n - 1)
    assert seq_mut_rate > sd_mut_rate, 'Mutation rate estimate must be greater than standard deviation.'
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
    j_n = np.diag(1 / np.arange(2, n + 1))
    j_ERM = get_ERM_matrix(n).dot(j_n)
    sfs = np.array(sfs)
    seg_sites = sum(sfs)

    with Model() as combined_model:
        # Note that upgraded version of pymc3 is required, otherwise pyMC3.Multinomial will not handle cases
        #  where probability =0 and corresponding data count=0 also.

        tree_index = Categorical('tree_index', tree_probabilities)
        tree_matrices = theano.shared(tree_matrices)
        jmx = tree_matrices[tree_index].dot(j_n)

        def dirich_logpdf(value=prior, a=prior):
            """This is a modification of pymc3.distributions.multivariate.Dirichlet.logp."""
            u = tt.dot(j_ERM, value.T)
            return bound(tt.sum(logpow(u, a - 1) - gammaln(a), axis=-1)
                         + gammaln(tt.sum(a, axis=-1)),
                         tt.all(u >= 0), tt.all(u <= 1),
                         tt.all(a > 0),
                         broadcast_conditions=False)

        stick = distributions.transforms.StickBreaking()
        probs = DensityDist('probs', dirich_logpdf, shape=(n - 1), testval=prior, transform=stick)
        conditional_probs = tt.dot(jmx, probs.T)
        sfs_obs = Multinomial('sfs_obs', n=seg_sites, p=conditional_probs, observed=sfs)

        total_length1 = Gamma('total_length', alpha=alpha, beta=beta)
        mut_rate = Beta('mut_rate', mu=seq_mut_rate, sd=sd_mut_rate)
        total_length = total_length1 * mut_rate
        seg_sites_obs = Poisson('seg_sites_obs', mu=total_length, observed=seg_sites)

    with combined_model:
        step = Metropolis(
            [probs, mut_rate, total_length])  # May require njobs=1 if not using pymc-devs:master. See issue #3011.
        tune = int(draws / 5)
        trace = sample(draws, tune=tune, step=step, progressbar=progressbar)
    print(summary(trace))
    return combined_model, trace


def multiply_variates(trace):
    """
    Multiply variates for relative branch length and total tree .ength to obtain variates for absolute branch length.

    """

    vars_rel = [t['probs'] for t in trace]
    vars_rel = np.array(vars_rel)
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

