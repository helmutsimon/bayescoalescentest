# coding=utf-8


""" This script creates synthetic data to test Bayesian inference method. It uses msprime and can take a range of
    demographic parameters. It generates an sfs, but maintains data on the exact sample tree.
    msprime routines come from tree_distribution.py.

    Parameters are job no, (diploid) population size, mutation rate, sequence length, population growth rate,
    sample size, number of replicates, directory for logs and data (-d optional), number of parallel jobs (-j
    optional) and msprime demographic events file (-e optional).

    A sample run statement is:

    nohup python3 ~/helmutsimonpython/helmutsimonpython/bayescoalescentest/python_scripts/simulate_population.py
    sp001 1e6 2e-9 1e3 0. 8 1000 > sp001.txt &

    Date: 6 September 2018."""

import numpy as np
import pandas as pd
import os, sys
import gzip, pickle
import msprime
from collections import Counter
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-2])
sys.path.append(projdir)

from shared import msprime_functions

LOGGER = CachingLogger(create_dir=True)


@click.command()
@click.argument('job_no')
@click.argument('pop_size', type=float)
@click.argument('mutation_rate', type=float)
@click.argument('length', type=float)
@click.argument('growth_rate', type=float)
@click.argument('n', type=int)
@click.argument('num_replicates', type=int)
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
@click.option('-j', '--n_jobs', default=5, type=int, help='Number of parallel jobs.')
@click.option('-e', '--events_file', default=None, help='Name of file containing demographic events')
def main(job_no, pop_size, mutation_rate, length, growth_rate, n, num_replicates, dir, n_jobs, events_file):
    start_time = time()
    if not os.path.exists(dir):
        os.makedirs(dir)
    LOGGER.log_file_path = dir + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Script md5sum".ljust(25))
    LOGGER.log_message('Name = ' + msprime.__name__ + ', version = ' + msprime.__version__,
                       label="Imported module".ljust(25))
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__,
                       label="Imported module".ljust(25))
    pop_size = int(pop_size)
    length = int(length)
    np.set_printoptions(precision=3)
    n_seq = np.arange(1, n)
    sfs_array, branch_length_array = list(), list()
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
        assert tree_sequence.num_sites == np.sum(sfs), 'seg site error'
    sfs_array = np.array(sfs_array)

    branch_length_array = np.array(branch_length_array)
    sample_sfs = sfs_array[0]
    sfs_array = sfs_array[1:,]
    #prior = np.mean(sfs_array, axis=0)
    seg_sites = np.sum(sfs_array, axis=1)
    seg_site_mean = np.mean(seg_sites)
    seg_site_sd = np.std(seg_sites)
    true_branch_lengths = branch_length_array[0]
    branch_length_array = branch_length_array[1:,]
    mean_branch_lengths =  np.mean(branch_length_array, axis=0)
    results = [sample_sfs, true_branch_lengths, mean_branch_lengths, seg_site_mean, seg_site_sd]
    filename = 'data/tree_simulations_' + job_no
    with gzip.open(filename, 'wb') as outfile:
        pickle.dump(results, outfile)
    outfile = open(filename, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    filename = 'data/sfs_' + job_no
    sfs_array = pd.DataFrame(sfs_array)
    sfs_array.to_csv(filename)
    outfile = open(filename, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    filename = 'data/brlens_' + job_no
    branch_length_array = pd.DataFrame(branch_length_array)
    branch_length_array.to_csv(filename)
    outfile = open(filename, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    thom = np.sum(sample_sfs * n_seq) / (n * length * mutation_rate)
    var_thom = np.sum(sample_sfs * n_seq * n_seq) / (2 * n * length * mutation_rate) ** 2
    LOGGER.log_message(str(sample_sfs), label='SFS'.ljust(25))
    LOGGER.log_message(str(true_branch_lengths), label='True branch lengths'.ljust(25))
    LOGGER.log_message(str(mean_branch_lengths), label='Mean branch lengths'.ljust(25))
    #LOGGER.log_message(str(prior), label='Mean of sampled SFSs'.ljust(25))
    LOGGER.log_message(str(seg_site_mean), label='Mean of seg. sites '.ljust(25))
    LOGGER.log_message(str(seg_site_sd), label='Std. dev. of seg sites'.ljust(25))
    LOGGER.log_message(str(np.sum(true_branch_lengths)), label='True TMRCA'.ljust(25))
    LOGGER.log_message("%.4f" % thom, label='Thomson TMRCA'.ljust(25))
    LOGGER.log_message("%.4f" % np.sqrt(var_thom), label="Thomson st. dev.".ljust(25))
    LOGGER.log_message(str(np.mean(branch_length_array, axis=0)), label='Expected branch lengths'.ljust(25))
    LOGGER.log_message("%.3G" % ((time() - start_time) / 60), label="Time (minutes)".ljust(25))


if __name__ == "__main__":
    main()

