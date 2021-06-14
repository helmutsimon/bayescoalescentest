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
@click.option('-m', '--mutation_rate', nargs=2, type=float, default=None,
                help='Estimated mutation rate & standard deviation. Required for uninformative prior.')
@click.option('-f', '--simuljobno', default=None,
                help='File of relative branch lengths for multivariate prior.')
@click.option('-l', '--lims', nargs=2, type=float, default=None,
                help='Limits for uniform distribution on mutation rates. Required for MVN prior.')
@click.option('-d', '--draws', type=float, default=50000)
@click.option('-o', '--order', default="random", help='Parameter for CategoricalGibbsMetropolis step.')
@click.option('-co', '--cores', default=None)
@click.option('-t', '--tune', default=None)
@click.option('-s', '--step', default=None)
@click.option('-ta', '--target_accept', default=0.9)
@click.option('-dir', '--dirx', default='data', help='Directory for data and log files. Default is data')
def main(job_no, filename, mutation_rate, simuljobno, lims, draws, order, cores, tune, step, target_accept, dirx):
    start_time = time()
    if not os.path.exists(dirx):
        os.makedirs(dirx)
    draws = int(draws)
    if cores is not None:
        cores = int(cores)
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
        print('True ttl  = '.ljust(25), '%.1f' % np.sum(true_branch_lengths * np.arange(2, n + 1)))
    if simuljobno:
        print('Model (multivariate-normal) prior.')
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
        model, trace = MCMC_functions.run_MCMC_mvn(sfs, lims[0], lims[1], mu, sigma,
                                ttl_mu, ttl_sigma, draws, order=order, cores=cores, tune=tune, step=step)
    else:
        print('Uninformative priors for relative branch lengths and total tree lngth.')
        variable_name = 'probs'
        seq_mut_rate = mutation_rate[0]
        sd_mut_rate = mutation_rate[1]
        print('Mean and sd of mut_rate;'.ljust(25), seq_mut_rate, sd_mut_rate)
        model, trace = MCMC_functions.run_MCMC_Dirichlet(sfs, seq_mut_rate, sd_mut_rate,
                                draws, order=order, cores=cores, tune=tune, step=step, target_accept=target_accept)
    summaryx = summary(trace)
    print('\n', summaryx)
    csv_name = dirx + '/pm_summary_' + job_no + '.csv'
    summaryx.to_csv(csv_name, sep=',')
    trace_df = tracetab.trace_to_dataframe(trace, include_transformed=True)
    trace_df.to_csv(dirx + '/pm_trace_' + job_no + '.csv')
    mut_rate_vars = [t['mut_rate'] for t in trace]
    mut_rate_vars = np.array(mut_rate_vars)
    mrate = np.mean(mut_rate_vars)
    print('Mean mutation rate = '.ljust(25), '%.8f' % mrate)
    print('STD mutation rate =  '.ljust(25), '%.8f' % np.std(mut_rate_vars))
    n_seq = np.arange(1, n)
    thom = np.sum(sfs * n_seq) / (n * mrate)
    print('Thomson est. TMRCA  = '.ljust(25), "%.1f" % thom)
    var_thom = np.sum(sfs * n_seq * n_seq) / (2 * n * mrate) ** 2
    print('Thomson std. error = '.ljust(25), '%.1f' % np.sqrt(var_thom), '\n')
    branch_vars = MCMC_functions.multiply_variates(trace, variable_name)
    mean_brlens = np.mean(branch_vars, axis=1)
    print('\nEst. branch lengths = '.ljust(25), mean_brlens)
    print('Est. TMRCA = '.ljust(25), '%.1f' % np.sum(mean_brlens))
    print('Est. ttl = '.ljust(25), '%.1f' % np.sum(mean_brlens * np.arange(2, n + 1)))
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
    data_file = open(file_path, 'r')
    LOGGER.input_file(data_file.name)
    data_file.close()
    summary_file = open(csv_name, 'r')
    LOGGER.output_file(summary_file.name)
    summary_file.close()

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

    matplotlib.logging.getLogger('matplotlib').setLevel('ERROR')
    matplotlib.logging.getLogger('matplotlib.font_manager').disabled = True
    arviz.rcParams['plot.max_subplots'] = 200
    tp = plot_trace(trace, compact=False)

    try:
        fig = plt.gcf()
        fig.savefig(dirx + '/traceplots_' + job_no + '.png')
    except Exception as errmsg:
        print('Traceplot not saved: ', str(errmsg))

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()


