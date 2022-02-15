# coding=utf-8


""" This script creates synthetic data to test Bayesian inference method. It uses msprime and can take a range of
    demographic parameters. It generates an sfs, but maintains data on the exact sample tree.
    msprime routines come from tree_distribution.py.

    Parameters are job no, (diploid) population size, mutation rate, sequence length, population growth rate,
    sample size, samples per population, directory for logs and data (-d optional), number of parallel jobs (-j
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
@click.option('-d', '--dir', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
@click.option('-e', '--events_file', default=None, help='Name of file containing demographic events')
def main(job_no, pop_size, mutation_rate, length, growth_rate, n, dir, events_file):
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
    sample_sfs, matrix, true_branch_lengths, true_tmrca, thom = msprime_functions.simulate_tree_and_sample \
            (n, pop_size, length, 0., mutation_rate, growth_rate, events_file)
    results = [sample_sfs, true_branch_lengths]
    filename = 'data/tree_simulations_' + job_no
    with gzip.open(filename, 'wb') as outfile:
        pickle.dump(results, outfile)
    outfile = open(filename, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    thom = np.sum(sample_sfs * n_seq) / (n * length * mutation_rate)
    var_thom = np.sum(sample_sfs * n_seq * n_seq) / (2 * n * length * mutation_rate) ** 2
    LOGGER.log_message(str(sample_sfs), label='SFS'.ljust(25))
    LOGGER.log_message(str(true_branch_lengths), label='True branch lengths'.ljust(25))
    LOGGER.log_message(str(np.sum(true_branch_lengths)), label='True TMRCA'.ljust(25))
    LOGGER.log_message("%.4f" % thom, label='Thomson TMRCA'.ljust(25))
    LOGGER.log_message("%.4f" % np.sqrt(var_thom), label="Thomson st. dev.".ljust(25))
    LOGGER.log_message("%.3G" % ((time() - start_time) / 60), label="Time (minutes)".ljust(25))


if __name__ == "__main__":
    main()

