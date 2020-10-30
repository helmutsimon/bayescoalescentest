# coding=utf-8


"""
Run MCMC analysis of dataset in the form produced by simulate_population.py. It saves an array of branch length variates.
A typical run statement is:

nohup python3 /Users/helmutsimon/repos/bayescoalescentest/python_scripts/run_bayesian_model.py 014 tree_simulations_sp020 5e-9 1e3 50000 > rbm014.txt &


"""

import numpy as np
import pandas as pd
import gzip, pickle
import os, sys
from time import time
import click
import scipy
import pymc3
#from pymc3 import *
import theano
from scitrack import CachingLogger, get_file_hexdigest
import matplotlib
import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-2])
sys.path.append(projdir)

from shared import run_MCMC

LOGGER = CachingLogger(create_dir=True)


@click.command()
@click.argument('job_no')
@click.argument('filename')
@click.argument('mutation_rate', type=float)
@click.argument('length', type=float)
@click.argument('mfilename', type=click.Path(exists=True))
@click.option('-f', '--simuljobno', default=None, help='File of relative branch lengths for multivariate prior.')
@click.option('-c', '--cv_mut', type=float, default=1e-6, help='Coefficient of variation of sequence mutation rate.')
@click.option('-d', '--draws', type=float, default=50000)
@click.option('-dir', '--dirx', default='data', help='Directory for data and log files. Default is data')
def main(job_no, filename, mutation_rate, length, mfilename, simuljobno, cv_mut, draws, dirx):
    start_time = time()
    if not os.path.exists(dirx):
        os.makedirs(dirx)
    LOGGER.log_file_path = dirx + "/" + str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
    try:
        LOGGER.log_message(str(os.environ['CONDA_DEFAULT_ENV']), label="Conda environment.".ljust(17))
    except KeyError:
        pass
    LOGGER.log_message('Name = ' + pymc3.__name__ + ', version = ' + pymc3.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + theano.__name__ + ', version = ' + theano.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + scipy.__name__ + ', version = ' + scipy.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + np.__name__ + ', version = ' + np.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + matplotlib.__name__ + ', version = ' + matplotlib.__version__,
                       label="Imported module".ljust(30))
    draws = int(draws)
    seq_mut_rate = mutation_rate * length
    sd_mut_rate = seq_mut_rate * cv_mut
    file_path = dirx + '/' + filename
    with gzip.open(file_path, 'rb') as results:
        results = pickle.load(results)
    infile = open(file_path, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    sfs, true_branch_lengths = results[:2]
    print(sfs, true_branch_lengths)
    n = len(sfs) + 1
    print('SFS ='.ljust(25), sfs)
    np.set_printoptions(precision=1)
    if true_branch_lengths is not None:
        print('True branch lengths = '.ljust(25), true_branch_lengths)
        true_coal_times = np.cumsum(true_branch_lengths[::-1])
        print('True coalescence times = '.ljust(25), true_coal_times)
        print('True TMRCA  = '.ljust(25), '%.1f' % np.sum(true_branch_lengths))
    n_seq = np.arange(1, n)
    thom = np.sum(sfs * n_seq) / (n * seq_mut_rate)
    print('Thomson est. TMRCA  = '.ljust(25), "%.1f" % thom)
    var_thom = np.sum(sfs * n_seq * n_seq) / (2 * n * seq_mut_rate) ** 2
    print('Thomson std. error = '.ljust(25), '%.1f' % np.sqrt(var_thom), '\n')

    with gzip.open(mfilename, 'rb') as mx_details:
        mx_details = pickle.load(mx_details)
    infile = open(mfilename, 'r')
    LOGGER.input_file(infile.name)
    infile.close()

    if simuljobno:
        print('Multivariate-normal prior')
        variable_name = 'mvn_sample'
        filename = dirx + '/relbrlens_' + simuljobno + '.csv'
        rbl_variates = pd.read_csv(filename, index_col=0)
        rbl_variates = np.array(rbl_variates)
        infile = open(file_path, 'r')
        LOGGER.input_file(infile.name)
        infile.close()
        j_n_inv = np.diag(np.arange(2, n + 1))
        simplex_variates = j_n_inv.dot(rbl_variates.T).T
        stick_bv = run_MCMC.StickBreaking_bv()
        transf_variates = stick_bv.forward_val(simplex_variates)
        mu = np.mean(transf_variates, axis=0)
        sigma = np.cov(transf_variates, rowvar=False)
        filename = dirx + '/brlens_' + simuljobno + '.csv'
        branch_length_array = pd.read_csv(filename, index_col=0)
        infile = open(filename, 'r')
        LOGGER.input_file(infile.name)
        infile.close()
        branch_length_array = np.array(branch_length_array)
        ttl_array = branch_length_array.dot(np.arange(2, n + 1).T)
        ttl_mu = np.mean(ttl_array)
        ttl_sigma = np.std(ttl_array)
        mcmc_model, trace = run_MCMC.run_MCMC_mvn(sfs, seq_mut_rate, sd_mut_rate, mx_details, mu, sigma,
                                                  ttl_mu, ttl_sigma, draws=draws)
    else:
        print('Uninformative Dirichlet prior assumed')
        variable_name = 'probs'
        mcmc_model, trace = run_MCMC.run_MCMC_Dirichlet(sfs, seq_mut_rate, sd_mut_rate, mx_details, draws=draws)

    branch_vars = run_MCMC.multiply_variates(trace, variable_name)
    print('\nEst. branch lengths = '.ljust(25), np.mean(branch_vars, axis=1))
    fname = dirx + '/branch_vars_' + job_no + '.pklz'
    with gzip.open(fname, 'wb') as buff:
        pickle.dump(branch_vars, buff)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    print('Est. TMRCA = '.ljust(25), '%.1f' % np.sum(np.mean(branch_vars, axis=1)))

    try:
        matplotlib.logging.getLogger('matplotlib.font_manager').disabled = True
        fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(20, 10))
        pymc3.traceplot(trace, ax=axs)
        plt.savefig(dirx + '/traceplots_' + job_no + '.png')
    except:
        print('Traceplot not saved.')

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()


