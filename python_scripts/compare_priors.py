# coding=utf-8


"""
    This script performs a number of Wright-Fisher simulations, then estimates the branch lengths using both diffuse
    and Wright-Fisher priors. MSEs are calculated in both cases, as well as tMRCAs.

    Parameters are job no, (diploid) population size, mutation rate, sequence length, number of trials (simulations),
    sample size, number of MCMC draws (-dr optional), directory for logs and data (-d optional),
    number of parallel jobs (-j optional).

    A sample run statement is:

    /Users/helmutsimon/repos/bayescoalescentest/python_scripts/compare_priors.py cp001 5.00E+04 1e-7 1000 5 8 -dr 20000
    > cp001.txt &

    Date: 5 June 2020."""

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

from shared import msprime_functions, run_MCMC

LOGGER = CachingLogger(create_dir=True)


@click.command()
@click.argument('job_no')
@click.argument('mutation_rate', type=float)
@click.argument('length', type=float)
@click.argument('size', type=int)
@click.option('-dr', '--draws', default=100000, help='Number of MCMC samples.')
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
@click.option('-j', '--n_jobs', default=5, type=int, help='Number of parallel jobs.')
def main(job_no, mutation_rate, length, size, draws, dirx, n_jobs):
    start_time = time()
    if not os.path.exists(dirx):
        os.makedirs(dirx)
    LOGGER.log_file_path = dirx + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Script md5sum".ljust(25))
    LOGGER.log_message('Name = ' + msprime.__name__ + ', version = ' + msprime.__version__,
                       label="Imported module".ljust(25))
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__,
                       label="Imported module".ljust(25))
    length = int(length)
    mfilename = dirx + '/matrix_details_5-12.pklz'
    with gzip.open(mfilename, 'rb') as mx_details:
        mx_details = pickle.load(mx_details)
    infile = open(mfilename, 'r')
    LOGGER.input_file(infile.name)
    infile.close()

    filename = 'tree_simulations_' + job_no
    file_path = dirx + '/' + filename
    with gzip.open(file_path, 'rb') as results:
        results = pickle.load(results)
    infile = open(file_path, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    sfs_, true_branch_lengths_, mean_branch_lengths, mu, sd = results[:5]
    rel_branch_lengths = mean_branch_lengths / np.sum(mean_branch_lengths)
    n = len(sfs_) + 1
    j_n_inv = np.diag(np.arange(2, n + 1))
    eprior = j_n_inv.dot(rel_branch_lengths)

    filename = dirx + '/sfs_' + job_no
    sfs_array = pd.read_csv(filename, index_col=0)
    sfs_array = np.array(sfs_array)
    infile = open(filename, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    n = sfs_array.shape[1] + 1

    filename = dirx + '/brlens_' + job_no
    branch_length_array = pd.read_csv(filename, index_col=0)
    branch_length_array = np.array(branch_length_array)
    infile = open(filename, 'r')
    LOGGER.input_file(infile.name)
    infile.close()

    seq_mutation_rate = mutation_rate * length
    sd_mutation_rate = seq_mutation_rate * 1e-6
    np.set_printoptions(precision=3)
    n_seq = np.arange(1, n)
    results = list()
    for sfs, brlens, i in zip(sfs_array, branch_length_array, range(size)):
        tmrca_true = np.sum(brlens)
        uprior = np.ones(n - 1)
        alpha = 1
        beta = 1e-10
        mcmc_model, trace = run_MCMC.run_MCMC(sfs, seq_mutation_rate, sd_mutation_rate, mx_details,
                                              draws=draws, prior=uprior, alpha=alpha, beta=beta)
        branch_vars = run_MCMC.multiply_variates(trace)
        bmd = np.mean(branch_vars, axis=1)
        tmrca_d = np.sum(bmd)
        mse_d = np.sqrt(np.linalg.norm(bmd - brlens))
        print('\nEst. branch lengths diffuse prior = '.ljust(25), bmd)

        alpha = (mu / sd) ** 2
        beta = alpha * seq_mutation_rate / mu
        mcmc_model, trace = run_MCMC.run_MCMC(sfs, seq_mutation_rate, sd_mutation_rate, mx_details,
                                              draws=draws, prior=eprior, alpha=alpha, beta=beta)
        branch_vars = run_MCMC.multiply_variates(trace)
        bmw = np.mean(branch_vars, axis=1)
        tmrca_w = np.sum(bmw)
        mse_w = np.sqrt(np.linalg.norm(bmw - brlens))
        print('\nEst. branch lengths demographic prior = '.ljust(25), bmw)
        thom = np.sum(sfs * n_seq) / (n * length * mutation_rate)
        row = [np.sum(sfs), mse_d, mse_w, tmrca_true, tmrca_d, tmrca_w, thom]
        results.append(row)
    columns =['S_n', 'mse_diff', 'mse_dem', 'tmrca_true', 'tmrca_diff', 'tmrca_dem', 'thom']
    result = pd.DataFrame(results, columns=columns)
    fname = dirx + '/compare_priors_result_' + job_no + '.csv'
    result.to_csv(fname, sep=',')
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    LOGGER.log_message("%.3G" % ((time() - start_time) / 60), label="Time (minutes)".ljust(25))


if __name__ == "__main__":
    main()

