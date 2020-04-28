# coding=utf-8


"""
Run MCMC analysis of dataset in the form produced by simulate_population.py. It saves an array of branch length variates.
A typical run statement is:

nohup python3 /Users/helmutsimon/repos/bayescoalescentest/python_scripts/run_bayesian_model.py 014 tree_simulations_sp020 5e-9 1e3 50000 > rbm014.txt &


"""

import numpy as np
import gzip, pickle
import os, sys
from time import time
import click
import scipy
import pymc3
from pymc3 import *
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
@click.option('-c', '--cv_mut', type=float, default=1e-10, help='Coefficient of variation of sequence mutation rate.')
@click.option('-d', '--draws', type=float, default=50000)
@click.option('--up/--ep', default=True,
              help='Choose whether to use uninformative (default) or empirical prior.')
@click.option('-dir', '--dirx', default='data', help='Directory for data and log files. Default is data')
def main(job_no, filename, mutation_rate, length, mfilename, cv_mut, draws, up, dirx):
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
    seq_mutation_rate = mutation_rate * length
    sd_mutation_rate = seq_mutation_rate * cv_mut
    file_path = dirx + '/' + filename
    with gzip.open(file_path, 'rb') as results:
        results = pickle.load(results)
    infile = open(file_path, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    sfs, true_branch_lengths, empirical_prior, mu, sd = results[:5]
    n = len(sfs) + 1
    if up:
        print('Uninformative prior')
        prior = np.ones(n - 1) / (n - 1)
        alpha = 1
        beta = 1e-10
    else:
        print('Empirical prior')
        prior = np.array(empirical_prior)
        shape_zeros = prior < 0.05  # Can't have Dirichlet parameters that are zero
        empirical_prior += shape_zeros * 0.1
        alpha = (mu / sd) ** 2
        beta = alpha * seq_mutation_rate / mu
    print('SFS ='.ljust(25), sfs)

    np.set_printoptions(precision=1)
    if true_branch_lengths is not None:
        print('True branch lengths = '.ljust(25), true_branch_lengths)
        true_coal_times = np.cumsum(true_branch_lengths[::-1])
        print('True coalescence times = '.ljust(25), true_coal_times)
        print('True TMRCA  = '.ljust(25), '%.1f' % np.sum(true_branch_lengths))
    n_seq = np.arange(1, n)
    thom = np.sum(sfs * n_seq) / (n * seq_mutation_rate)
    print('Thomson est. TMRCA  = '.ljust(25), "%.1f" % thom)
    var_thom = np.sum(sfs * n_seq * n_seq) / (2 * n * seq_mutation_rate) ** 2
    print('Thomson std. error = '.ljust(25), '%.1f' % np.sqrt(var_thom), '\n')

    with gzip.open(mfilename, 'rb') as mx_details:
        mx_details = pickle.load(mx_details)
    infile = open(mfilename, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    mcmc_model, trace = run_MCMC.run_MCMC(sfs, seq_mutation_rate, sd_mutation_rate, mx_details,
                                     draws=draws, prior=prior, alpha=alpha, beta=beta)
    branch_vars = run_MCMC.multiply_variates(trace)
    print('\nEst. branch lengths = '.ljust(25), np.mean(branch_vars, axis=1))
    fname = dirx + '/branch_vars_' + job_no + '.pklz'
    with gzip.open(fname, 'wb') as buff:
        pickle.dump(branch_vars, buff)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    print('Est. TMRCA = '.ljust(25), '%.1f' % np.sum(np.mean(branch_vars, axis=1)))

    try:
        fig, axs = plt.subplots(nrows=2, ncols=1)
        axs[0] = forestplot(trace, varnames=['probs'])
        axs[1] = traceplot(trace, varnames=['probs'])
        plt.savefig(dirx + '/traceplots_' + job_no + '.png')
    except:
        pass

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()


