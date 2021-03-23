# coding=utf-8

"""
    This script performs a number of Wright-Fisher simulations, then estimates the branch lengths using both Dirichlet
    and multivariate normal priors. MSEs are calculated in both cases, as well as TMRCAs.

    Parameters are job no, mutation rate, sequence length, number of trials (simulations),
    number of MCMC draws (-dr optional) and directory for logs and data (-d optional).

    A sample run statement is:

    /Users/helmutsimon/repos/bayescoalescentest/python_scripts/compare_priors.py cp001 5.00E+04 1e-7 1000 5 8 -dr 20000
    > cp001.txt &

    Date: 23 March 2021."""

import numpy as np
import pandas as pd
import os, sys
import pymc3
import theano
import msprime
from pymc3.distributions.multivariate import Multinomial
from pymc3.distributions.discrete import Poisson
from pymc3.distributions.continuous import Exponential, Uniform
from pymc3.model import Model
from pymc3.sampling import sample
from pymc3.step_methods.metropolis import Metropolis, CategoricalGibbsMetropolis
from theano import config
import theano.tensor as tt
from theano.compile.ops import as_op
import arviz
from arviz import summary
from collections import Counter
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-2])
sys.path.append(projdir)

from shared import MCMC_functions, msprime_functions


LOGGER = CachingLogger(create_dir=True)


def run_MCMC_WF(sfs, mrate_lower, mrate_upper, pop_size, draws=50000,
                 progressbar=False, order="random", cores=None):
    """Define and run MCMC model for coalescent tree branch lengths using a multivariate normal prior."""
    config.compute_test_value = 'raise'
    n = len(sfs) + 1
    sfs = np.array(sfs)
    sfs = tt.as_tensor(sfs)
    seg_sites = sum(sfs)
    if order == "inc":
        order = np.arange(n - 2)
    elif order == "dec":
        order = np.arange(n - 2)
        order = np.flip(order)
    else:
        order = "random"
    params = np.arange(2, n + 1) * np.arange(1, n) / (4 * pop_size)

    def infer_matrix_shape(fgraph, node, input_shapes):
        return [(n - 1, n - 1)]

    @as_op(itypes=[tt.lvector], otypes=[tt.fmatrix], infer_shape=infer_matrix_shape)
    def jmatrix(i):
        """Derive tree matrix from Lehmer code representation."""
        mx = MCMC_functions.derive_tree_matrix(i)
        mx = np.reshape(mx, (n - 1, n - 1))
        mx = np.transpose(mx)
        mx = mx.astype('float32')
        return mx

    with Model() as combined_model:
        # Create sequence of categorical distributions to sample Lehmer code representation.
        permutation = MCMC_functions.Lehmer_distribution(n)
        permutation_tt = tt.as_tensor(permutation)
        mx = jmatrix(permutation_tt)

        brlens = Exponential('brlens', params, shape=n - 1)
        ttl = tt.sum(tt.dot(brlens, np.arange(2, n + 1)))
        rbrlens = brlens / ttl
        q = tt.dot(mx, rbrlens.T)
        sfs_obs = Multinomial('sfs_obs', n=seg_sites, p=q, observed=sfs)

        mut_rate = Uniform('mut_rate', lower=mrate_lower, upper=mrate_upper)
        mu = ttl * mut_rate
        seg_sites_obs = Poisson('seg_sites_obs', mu=mu, observed=seg_sites)

    with combined_model:
        step1 = Metropolis([brlens, mut_rate])
        step2 = CategoricalGibbsMetropolis(permutation, order=order)
        tune = int(draws / 5)
        trace = sample(draws, tune=tune, step=[step1, step2], progressbar=progressbar,
                       cores=cores, return_inferencedata=False)
    return combined_model, trace


@click.command()
@click.argument('job_no')
@click.argument('n', type=int)
@click.argument('pop_size', type=float)
@click.argument('mutation_rate', type=float)
@click.argument('length', type=float)
@click.argument('growth_rate', type=float)
@click.argument('size', type=int)
@click.argument('num_replicates', type=int)
@click.option('-co', '--cores', default=None)
@click.option('-dr', '--draws', default=20000, help='Number of MCMC samples.')
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(job_no, n, pop_size, mutation_rate, length, growth_rate, size, num_replicates, cores, draws, dirx):
    start_time = time()
    if not os.path.exists(dirx):
        os.makedirs(dirx)
    length = int(length)
    if cores is not None:
        cores = int(cores)
    pop_size = int(pop_size)
    length = int(length)
    np.set_printoptions(precision=3)
    sfs_array, branch_length_array, rel_branch_length_array = list(), list(), list()
    population_configurations = [msprime.PopulationConfiguration(growth_rate=growth_rate,
                                                                 initial_size=pop_size, sample_size=n)]
    demographic_hist = msprime.DemographyDebugger(population_configurations=population_configurations)
    demographic_hist.print_history()
    replicates = msprime.simulate(population_configurations=population_configurations, mutation_rate=mutation_rate,
                                  length=length, num_replicates=num_replicates + 1)
    for j, tree_sequence in enumerate(replicates):
        shape = tree_sequence.get_num_mutations(), tree_sequence.get_sample_size()
        variant_array = np.empty(shape, dtype="u1")
        for variant in tree_sequence.variants():
            variant_array[variant.index] = variant.genotypes
        occurrences = np.sum(variant_array, axis=1)
        sfs = Counter(occurrences)
        sfs = [sfs[i] for i in range(1, n)]
        sfs_array.append(sfs)
        tree = next(tree_sequence.trees())
        mx, coal_times = msprime_functions.analyse_tree(n, tree)
        coal_times = np.array(coal_times)
        shifted = np.concatenate((np.zeros(1), coal_times))[:-1]
        branch_lengths = np.flipud(coal_times - shifted)
        branch_length_array.append(branch_lengths)
        ttl = np.sum(np.arange(2, n + 1) * branch_lengths)
        rel_branch_lengths = branch_lengths / ttl
        rel_branch_length_array.append(rel_branch_lengths)
        assert tree_sequence.num_sites == np.sum(sfs), 'seg site error'
    sfs_array = np.array(sfs_array)
    branch_length_array = np.array(branch_length_array)
    seg_sites = np.sum(sfs_array, axis=1)
    seg_site_mean = np.mean(seg_sites)
    seg_site_sd = np.std(seg_sites)
    print('Mean & sde segregating sites', seg_site_mean, seg_site_sd)
    sys.stdout.flush()

    seq_mutation_rate = mutation_rate * length
    sd_mutation_rate = seq_mutation_rate * 1e-6
    mrate_lower = seq_mutation_rate - sd_mutation_rate
    mrate_upper = seq_mutation_rate + sd_mutation_rate
    n_seq = np.arange(1, n)
    results, summaries, summaries_mvn = list(), list(), list()
    for sfs, brlens, i in zip(sfs_array, branch_length_array, range(size)):
        print('\nTrue branch lengths = '.ljust(25), brlens)
        tmrca_true = np.sum(brlens)
        mcmc_model, trace = run_MCMC_WF(sfs, mrate_lower, mrate_upper, pop_size, cores=cores, draws=draws)
        summaryx = summary(trace)
        summaries.append(summaryx)
        branch_vars = [t['brlens'] for t in trace]
        bmd = np.mean(branch_vars, axis=0)
        tmrca_est = np.sum(bmd)
        mse = np.sqrt(np.linalg.norm(bmd - brlens))
        print('\nEst. branch lengths = '.ljust(25), bmd)
        thom = np.sum(sfs * n_seq) / (n * length * mutation_rate)
        row = [n, np.sum(sfs), mse, tmrca_true, tmrca_est, thom]
        results.append(row)
        print('\nTrue TMRCA = '.ljust(25), np.sum(brlens))
        print('Est. TMRCA = '.ljust(25), np.sum(bmd))

        sys.stdout.flush()

    LOGGER.log_file_path = dirx + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Script md5sum".ljust(25))
    try:
        LOGGER.log_message(str(os.environ['CONDA_DEFAULT_ENV']), label="Conda environment.".ljust(17))
    except KeyError:
        pass
    LOGGER.log_message('Name = ' + np.__name__ + ', version = ' + np.__version__,
                       label="Imported module".ljust(25))
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__,
                       label="Imported module".ljust(25))
    LOGGER.log_message('Name = ' + pymc3.__name__ + ', version = ' + pymc3.__version__,
                       label="Imported module".ljust(25))
    LOGGER.log_message('Name = ' + arviz.__name__ + ', version = ' + arviz.__version__,
                       label="Imported module".ljust(25))
    LOGGER.log_message('Name = ' + theano.__name__ + ', version = ' + theano.__version__,
                       label="Imported module".ljust(25))
    LOGGER.log_message('Name = ' + msprime.__name__ + ', version = ' + msprime.__version__,
                       label="Imported module".ljust(25))
    summaries = pd.concat(summaries, ignore_index=True)
    csv_name = dirx + '/pm_summaries_Dir_' + job_no + '.csv'
    summaries.to_csv(csv_name, sep=',')
    summary_file = open(csv_name, 'r')
    LOGGER.output_file(summary_file.name)
    summary_file.close()
    columns =['n', 'S_n', 'mse', 'tmrca_true', 'tmrca_est', 'thom']
    result = pd.DataFrame(results, columns=columns)
    fname = dirx + '/compare_priors_result_' + job_no + '.csv'
    result.to_csv(fname, sep=',')
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    LOGGER.log_message("%.3G" % ((time() - start_time) / 60), label="Time (minutes)".ljust(25))


if __name__ == "__main__":
    main()

