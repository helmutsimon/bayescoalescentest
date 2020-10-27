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
import gzip, pickle
import pymc3
import theano
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-2])
sys.path.append(projdir)

from shared import run_MCMC

LOGGER = CachingLogger(create_dir=True)


@click.command()
@click.argument('job_no')
@click.argument('mutation_rate', type=float)
@click.argument('length', type=float)
@click.argument('size', type=int)
@click.option('-dr', '--draws', default=100000, help='Number of MCMC samples.')
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(job_no, mutation_rate, length, size, draws, dirx):
    start_time = time()
    if not os.path.exists(dirx):
        os.makedirs(dirx)
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
    LOGGER.log_message('Name = ' + theano.__name__ + ', version = ' + theano.__version__,
                       label="Imported module".ljust(25))
    length = int(length)
    mfilename = dirx + '/matrix_details_5-12.pklz'
    with gzip.open(mfilename, 'rb') as mx_details:
        mx_details = pickle.load(mx_details)
    infile = open(mfilename, 'r')
    LOGGER.input_file(infile.name)
    infile.close()

    filename = 'relbrlens_' + job_no + '.csv'
    file_path = dirx + '/' + filename
    rbl_variates = pd.read_csv(file_path, index_col=0)
    rbl_variates = np.array(rbl_variates)
    infile = open(file_path, 'r')
    LOGGER.input_file(infile.name)
    infile.close()

    filename = dirx + '/sfs_' + job_no + '.csv'
    sfs_array = pd.read_csv(filename, index_col=0)
    sfs_array = np.array(sfs_array)
    infile = open(filename, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    n = sfs_array.shape[1] + 1

    filename = dirx + '/brlens_' + job_no + '.csv'
    branch_length_array = pd.read_csv(filename, index_col=0)
    infile = open(filename, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    branch_length_array = np.array(branch_length_array)
    ttl_array = branch_length_array.dot(np.arange(2, n + 1).T)
    ttl_mu = np.mean(ttl_array)
    ttl_sigma = np.std(ttl_array)

    seq_mutation_rate = mutation_rate * length
    sd_mutation_rate = seq_mutation_rate * 1e-6
    j_n_inv = np.diag(np.arange(2, n + 1))
    simplex_variates = j_n_inv.dot(rbl_variates.T).T
    stick_bv = run_MCMC.StickBreaking_bv()
    transf_variates = stick_bv.forward_val(simplex_variates)
    mu = np.mean(transf_variates, axis=0)
    sigma = np.cov(transf_variates, rowvar=False)
    np.set_printoptions(precision=3)
    n_seq = np.arange(1, n)
    results = list()
    for sfs, brlens, i in zip(sfs_array, branch_length_array, range(size)):
        print('\nTrue branch lengths = '.ljust(25), brlens)
        tmrca_true = np.sum(brlens)
        mcmc_model, trace = run_MCMC.run_MCMC_Dirichlet(sfs, seq_mutation_rate, sd_mutation_rate, mx_details, draws=draws)
        branch_vars = run_MCMC.multiply_variates(trace, 'probs')
        bmd = np.mean(branch_vars, axis=1)
        tmrca_d = np.sum(bmd)
        mse_d = np.sqrt(np.linalg.norm(bmd - brlens))
        print('\nEst. branch lengths Dirichlet prior = '.ljust(25), bmd)

        mcmc_model, trace = run_MCMC.run_MCMC_mvn(sfs, seq_mutation_rate, sd_mutation_rate, mx_details,
                                            mu, sigma, ttl_mu, ttl_sigma, draws=draws)
        branch_vars = run_MCMC.multiply_variates(trace, 'mvn_sample')
        bmw = np.mean(branch_vars, axis=1)
        tmrca_w = np.sum(bmw)
        mse_w = np.sqrt(np.linalg.norm(bmw - brlens))
        print('\nEst. branch lengths mv-normal prior = '.ljust(25), bmw)
        thom = np.sum(sfs * n_seq) / (n * length * mutation_rate)
        row = [np.sum(sfs), mse_d, mse_w, tmrca_true, tmrca_d, tmrca_w, thom]
        results.append(row)
    columns =['S_n', 'mse_Dir', 'mse_mvn', 'tmrca_true', 'tmrca_Dir', 'tmrca_mvn', 'thom']
    result = pd.DataFrame(results, columns=columns)
    fname = dirx + '/compare_priors_result_' + job_no + '.csv'
    result.to_csv(fname, sep=',')
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    LOGGER.log_message("%.3G" % ((time() - start_time) / 60), label="Time (minutes)".ljust(25))


if __name__ == "__main__":
    main()

