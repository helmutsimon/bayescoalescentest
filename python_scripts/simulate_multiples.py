# coding=utf-8


"""
    This script creates synthetic data to test the Bayesian inference method. It uses msprime and can take a range of
    demographic parameters. It generates multiple SFS's and corresponding set of branch lengths, and calculates
    prior distribution parameters from the set of generated trees.

    Parameters are job no, (diploid) population size, mutation rate, sequence length, population growth rate,
    sample size, number of replicates, directory for logs and data (-d optional), number of parallel jobs (-j
    optional) and msprime demographic events file (-e optional).

    A sample run statement is:

    nohup python3 ~/helmutsimonpython/helmutsimonpython/bayescoalescentest/python_scripts/simulate_multiples.py
    sp001 1e6 2e-9 1e3 0. 8 1000 > sm001.txt &

"""

import numpy as np
import pandas as pd
import os, sys
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
@click.argument('n', nargs=-1, type=int)
@click.argument('pop_size', type=float)
@click.argument('mutation_rate', type=float)
@click.argument('length', type=float)
@click.argument('growth_rate', type=float)
@click.argument('num_replicates', type=int)
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
@click.option('-e', '--events_file', default=None, help='Name of file containing demographic events')
def main(job_no, pop_size, mutation_rate, length, growth_rate, n, num_replicates, dir, events_file):
    start_time = time()
    if not os.path.exists(dir):
        os.makedirs(dir)
    LOGGER.log_file_path = dir + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Script md5sum".ljust(25))
    try:
        LOGGER.log_message(str(os.environ['CONDA_DEFAULT_ENV']), label="Conda environment.".ljust(17))
    except KeyError:
        pass
    LOGGER.log_message('Name = ' + msprime.__name__ + ', version = ' + msprime.__version__,
                       label="Imported module".ljust(25))
    LOGGER.log_message('Name = ' + np.__name__ + ', version = ' + np.__version__,
                       label="Imported module".ljust(25))
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__,
                       label="Imported module".ljust(25))
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

    rel_branch_length_array = np.array(rel_branch_length_array)
    mean_rel_branch_lengths = np.mean(rel_branch_length_array, axis=0)
    seg_sites = np.sum(sfs_array, axis=1)
    seg_site_mean = np.mean(seg_sites)
    seg_site_sd = np.std(seg_sites)

    filename = 'data/sfs_' + job_no + '.csv'
    sfs_array = pd.DataFrame(sfs_array)
    sfs_array.to_csv(filename)
    outfile = open(filename, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    filename = 'data/relbrlens_' + job_no + '.csv'
    rel_branch_length_array = pd.DataFrame(rel_branch_length_array)
    rel_branch_length_array.to_csv(filename)
    outfile = open(filename, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    filename = 'data/brlens_' + job_no + '.csv'
    branch_length_array = pd.DataFrame(branch_length_array)
    branch_length_array.to_csv(filename)
    outfile = open(filename, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    LOGGER.log_message(str(mean_rel_branch_lengths), label='Mean relative branch lengths'.ljust(25))
    LOGGER.log_message(str(np.mean(sfs_array, axis=0)), label='Mean of sampled SFSs'.ljust(25))
    LOGGER.log_message(str(seg_site_mean), label='Mean of seg. sites '.ljust(25))
    LOGGER.log_message(str(seg_site_sd), label='Std. dev. of seg sites'.ljust(25))
    mean_branch_lengths = np.mean(branch_length_array, axis=0)
    LOGGER.log_message(str(mean_branch_lengths), label='Mean branch lengths'.ljust(25))
    mean_ttl = sum(mean_branch_lengths * (np.arange(2, n + 1)))
    LOGGER.log_message(str(mean_ttl), label='Mean total tree length'.ljust(25))
    LOGGER.log_message("%.3G" % ((time() - start_time) / 60), label="Time (minutes)".ljust(25))


if __name__ == "__main__":
    main()

