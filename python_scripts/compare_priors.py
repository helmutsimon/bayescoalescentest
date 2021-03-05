# coding=utf-8

"""
    This script performs a number of Wright-Fisher simulations, then estimates the branch lengths using both Dirichlet
    and multivariate normal priors. MSEs are calculated in both cases, as well as TMRCAs.

    Parameters are job no, mutation rate, sequence length, number of trials (simulations),
    number of MCMC draws (-dr optional) and directory for logs and data (-d optional).

    A sample run statement is:

    /Users/helmutsimon/repos/bayescoalescentest/python_scripts/compare_priors.py cp001 5.00E+04 1e-7 1000 5 8 -dr 20000
    > cp001.txt &

    Date: 5 June 2020."""

import numpy as np
import pandas as pd
import os, sys
import pymc3
import theano
import msprime
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
        # print(variant_array)
        occurrences = np.sum(variant_array, axis=1)
        # print(occurrences)
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
    rel_branch_length_array = np.array(rel_branch_length_array)
    seg_sites = np.sum(sfs_array, axis=1)
    seg_site_mean = np.mean(seg_sites)
    seg_site_sd = np.std(seg_sites)
    print('Mean & sde segregating sites', seg_site_mean, seg_site_sd)
    sys.stdout.flush()
    ttl_array = branch_length_array.dot(np.arange(2, n + 1).T)
    ttl_mu = np.mean(ttl_array)
    ttl_sigma = np.std(ttl_array)

    seq_mutation_rate = mutation_rate * length
    sd_mutation_rate = seq_mutation_rate * 1e-6
    mrate_lower = seq_mutation_rate - sd_mutation_rate
    mrate_upper = seq_mutation_rate + sd_mutation_rate
    j_n_inv = np.diag(np.arange(2, n + 1))
    simplex_variates = j_n_inv.dot(rel_branch_length_array.T).T
    stick_bv = MCMC_functions.StickBreaking_bv()
    transf_variates = stick_bv.forward_val(simplex_variates)
    mu = np.mean(transf_variates, axis=0)
    sigma = np.cov(transf_variates, rowvar=False)
    n_seq = np.arange(1, n)
    results, summaries_Dir, summaries_mvn = list(), list(), list()
    for sfs, brlens, i in zip(sfs_array, branch_length_array, range(size)):
        print('\nTrue branch lengths = '.ljust(25), brlens)
        tmrca_true = np.sum(brlens)
        mcmc_model, trace = MCMC_functions.run_MCMC_Dirichlet(sfs, seq_mutation_rate, sd_mutation_rate,
                                            cores=cores, draws=draws)
        summaryx = summary(trace)
        summaries_Dir.append(summaryx)
        branch_vars = MCMC_functions.multiply_variates(trace, 'probs')
        bmd = np.mean(branch_vars, axis=1)
        tmrca_d = np.sum(bmd)
        mse_d = np.sqrt(np.linalg.norm(bmd - brlens))
        print('\nEst. branch lengths Dirichlet prior = '.ljust(25), bmd)

        mcmc_model, trace = MCMC_functions.run_MCMC_mvn(sfs, mrate_lower, mrate_upper,
                                            mu, sigma, ttl_mu, ttl_sigma, cores=cores, draws=draws)
        summaryx = summary(trace)
        summaries_mvn.append(summaryx)
        branch_vars = MCMC_functions.multiply_variates(trace, 'mvn_sample')
        bmw = np.mean(branch_vars, axis=1)
        tmrca_w = np.sum(bmw)
        mse_w = np.sqrt(np.linalg.norm(bmw - brlens))
        print('\nEst. branch lengths MVN prior = '.ljust(25), bmw)
        thom = np.sum(sfs * n_seq) / (n * length * mutation_rate)
        row = [n, np.sum(sfs), mse_d, mse_w, tmrca_true, tmrca_d, tmrca_w, thom]
        results.append(row)
        print('\nTrue TMRCA = '.ljust(25), np.sum(brlens))
        print('Est. TMRCA Dirichlet prior = '.ljust(25), np.sum(bmd))
        print('Est. TMRCA MVN prior = '.ljust(25), np.sum(bmw))
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
    summaries_Dir = pd.concat(summaries_Dir, ignore_index=True)
    csv_name = dirx + '/pm_summaries_Dir_' + job_no + '.csv'
    summaries_Dir.to_csv(csv_name, sep=',')
    summary_file = open(csv_name, 'r')
    LOGGER.output_file(summary_file.name)
    summary_file.close()
    summaries_mvn = pd.concat(summaries_mvn, ignore_index=True)
    csv_name = dirx + '/pm_summaries_mvn_' + job_no + '.csv'
    summaries_mvn.to_csv(csv_name, sep=',')
    summary_file = open(csv_name, 'r')
    LOGGER.output_file(summary_file.name)
    summary_file.close()
    columns =['n', 'S_n', 'mse_Dir', 'mse_mvn', 'tmrca_true', 'tmrca_Dir', 'tmrca_mvn', 'thom']
    result = pd.DataFrame(results, columns=columns)
    fname = dirx + '/compare_priors_result_' + job_no + '.csv'
    result.to_csv(fname, sep=',')
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    LOGGER.log_message("%.3G" % ((time() - start_time) / 60), label="Time (minutes)".ljust(25))


if __name__ == "__main__":
    main()

