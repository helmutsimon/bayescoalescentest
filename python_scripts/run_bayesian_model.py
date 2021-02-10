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
import pymc3
import arviz
from arviz import summary, plot_trace
from pymc3.backends import tracetab
import theano
from scitrack import CachingLogger, get_file_hexdigest
import matplotlib
import matplotlib.pyplot as plt

abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-2])
sys.path.append(projdir)

from shared import MCMC_functions

LOGGER = CachingLogger(create_dir=True)


@click.command()
@click.argument('job_no')
@click.argument('filename')
@click.argument('mutation_rate', type=float)
@click.argument('length', type=float)
@click.option('-f', '--simuljobno', default=None, help='File of relative branch lengths for multivariate prior.')
@click.option('-c', '--cv_mut', type=float, default=1e-6, help='Coefficient of variation of sequence mutation rate.')
@click.option('-d', '--draws', type=float, default=50000)
@click.option('-o', '--order', default="random")
@click.option('-co', '--cores', default=None)
@click.option('-dir', '--dirx', default='data', help='Directory for data and log files. Default is data')
def main(job_no, filename, mutation_rate, length, simuljobno, cv_mut, draws, order, cores, dirx):
    start_time = time()
    if not os.path.exists(dirx):
        os.makedirs(dirx)
    draws = int(draws)
    seq_mut_rate = mutation_rate * length
    sd_mut_rate = seq_mut_rate * cv_mut
    file_path = dirx + '/' + filename
    with gzip.open(file_path, 'rb') as results:
        results = pickle.load(results)
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
    if simuljobno:
        print('Multivariate-normal prior')
        variable_name = 'mvn_sample'
        filename = dirx + '/relbrlens_' + simuljobno + '.csv'
        rbl_variates = pd.read_csv(filename, index_col=0)
        rbl_variates = np.array(rbl_variates)
        j_n_inv = np.diag(np.arange(2, n + 1))
        simplex_variates = j_n_inv.dot(rbl_variates.T).T
        stick_bv = MCMC_functions.StickBreaking_bv()
        transf_variates = stick_bv.forward_val(simplex_variates)
        mu = np.mean(transf_variates, axis=0)
        sigma = np.cov(transf_variates, rowvar=False)
        filename = dirx + '/brlens_' + simuljobno + '.csv'
        branch_length_array = pd.read_csv(filename, index_col=0)
        branch_length_array = np.array(branch_length_array)
        ttl_array = branch_length_array.dot(np.arange(2, n + 1).T)
        ttl_mu = np.mean(ttl_array)
        ttl_sigma = np.std(ttl_array)
        model, trace = MCMC_functions.run_MCMC_mvn(sfs, seq_mut_rate, sd_mut_rate, mu, sigma,
                                                         ttl_mu, ttl_sigma, draws, order=order, cores=cores)
    else:
        print('Uninformative Dirichlet prior assumed')
        variable_name = 'probs'
        print('Mean and sd of mut_rate;'.ljust(25), seq_mut_rate, sd_mut_rate)
        model, trace = MCMC_functions.run_MCMC_Dirichlet(sfs, seq_mut_rate, sd_mut_rate,
                                                         draws, order=order, cores=cores)
    summaryx = summary(trace)
    print('\n', summaryx)
    csv_name = dirx + '/pm_summary_' + job_no + '.csv'
    summaryx.to_csv(csv_name, sep=',')
    trace_df = tracetab.trace_to_dataframe(trace, include_transformed=True)
    trace_df.to_csv(dirx + '/pm_trace_' + job_no + '.csv')
    mut_rate_vars = [t['mut_rate'] for t in trace]
    mut_rate_vars = np.array(mut_rate_vars)
    print('Mean mutation rate = '.ljust(25), '%.8f' % np.mean(mut_rate_vars))
    print('STD mutation rate =  '.ljust(25), '%.8f' % np.std(mut_rate_vars))
    branch_vars = MCMC_functions.multiply_variates(trace, variable_name)
    print('\nEst. branch lengths = '.ljust(25), np.mean(branch_vars, axis=1))
    fname = dirx + '/branch_vars_' + job_no + '.pklz'
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
    LOGGER.log_message('Name = ' + arviz.__name__ + ', version = ' + arviz.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + np.__name__ + ', version = ' + np.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + matplotlib.__name__ + ', version = ' + matplotlib.__version__,
                       label="Imported module".ljust(30))
    infile = open(csv_name, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    if simuljobno:
        filename = dirx + '/relbrlens_' + simuljobno + '.csv'
        infile = open(filename, 'r')
        LOGGER.input_file(infile.name)
        infile.close()
        filename = dirx + '/brlens_' + simuljobno + '.csv'
        infile = open(filename, 'r')
        LOGGER.input_file(infile.name)
        infile.close()
    with gzip.open(fname, 'wb') as buff:
        pickle.dump(branch_vars, buff)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()
    print('Est. TMRCA = '.ljust(25), '%.1f' % np.sum(np.mean(branch_vars, axis=1)))

    matplotlib.logging.getLogger('matplotlib').setLevel('ERROR')
    matplotlib.logging.getLogger('matplotlib.font_manager').disabled = True
    tp = plot_trace(trace)

    try:
        fig = plt.gcf()
        fig.savefig(dirx + '/traceplots_' + job_no + '.png')
    except Exception as errmsg:
        print('Traceplot not saved: ', str(errmsg))

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()


