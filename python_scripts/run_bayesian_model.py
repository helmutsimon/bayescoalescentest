# coding=utf-8


"""
Run MCMC analysis of dataset in the form produced by simulate_population.py. It saves an array of branch length variates.
A typical run statement is:

nohup python /Users/helmutsimon/repos/bayescoalescentest/python_scripts/run_bayesian_model.py
       014 12 tree_simulations_sp020 -m 5e-9 1e-9 -d 50000 > rbm014.txt &
"""

import numpy as np
import pandas as pd
import gzip, pickle
import os, sys
from time import time
import click
import arviz
import theano
from scitrack import CachingLogger, get_file_hexdigest
import warnings
from packaging.version import parse
import matplotlib
from importlib.metadata import version

abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-2])
sys.path.append(projdir)

from shared import MCMC_functions

#LOGGER = CachingLogger(create_dir=True)        ##wrong location?


def wrap_MCMC(n, filename, mutation_rate, simuljobno, mrate_lims, folded, draws, order,
                cores, tune, target_accept, concentration, return_inferencedata):
    with gzip.open(filename, 'rb') as inputs:
        inputs = pickle.load(inputs)
    sfs, true_branch_lengths = inputs[:2]
    print('SFS ='.ljust(25), sfs)
    np.set_printoptions(precision=1)
    if true_branch_lengths is not None:
        print('True branch lengths = '.ljust(25), true_branch_lengths)
        true_coal_times = np.cumsum(true_branch_lengths[::-1])
        print('True coalescence times = '.ljust(25), true_coal_times)
        print('True TMRCA  = '.ljust(25), '%.1f' % np.sum(true_branch_lengths))
        print('True ttl  = '.ljust(25), '%.1f' % np.sum(true_branch_lengths * np.arange(2, n + 1)))
    mrate_lower = mrate_lims[0]
    mrate_upper = mrate_lims[1]
    seq_mut_rate = mutation_rate[0]
    sd_mut_rate = mutation_rate[1]
    if seq_mut_rate:
        if not folded:
            thom, thom_sd = MCMC_functions.Thomson_estimate(sfs, seq_mut_rate)
            print('Thomson est. TMRCA  = '.ljust(25), "%.1f" % thom)
            print('Thomson std. error = '.ljust(25), '%.1f' % thom_sd)
            print('Thomson estimates above use estimated mutation rate supplied.', '\n')
    mu, sigma, ttl_mu, ttl_sigma = None, None, None, None

    if simuljobno:
        # In this case, we are using MVN prior, so we need to generate parameters for the prior distributions.
        print('Model (multivariate-normal) prior.')
        variable_name = 'mvn_sample'
        filename = 'relbrlens_' + simuljobno + '.csv'
        rbl_variates = pd.read_csv(filename, index_col=0)
        rbl_variates = np.array(rbl_variates)
        j_n_inv = np.diag(np.arange(2, n + 1))
        simplex_variates = j_n_inv.dot(rbl_variates.T).T
        transf_variates = MCMC_functions.forward_val(simplex_variates)
        mu = np.mean(transf_variates, axis=0)
        sigma = np.cov(transf_variates, rowvar=False)
        filename = 'brlens_' + simuljobno + '.csv'
        branch_length_array = pd.read_csv(filename, index_col=0)
        branch_length_array = np.array(branch_length_array)
        ttl_array = branch_length_array.dot(np.arange(2, n + 1).T)
        ttl_mu = np.mean(ttl_array)
        ttl_sigma = np.std(ttl_array)
        print('Mutation rate interval;'.ljust(25), '[', mrate_lower, ',', mrate_upper, ']')
    else:
        print('Uninformative priors for relative branch lengths and total tree length.')
        variable_name = 'probs'
        print('Mean and sd of mut_rate;'.ljust(25), seq_mut_rate, sd_mut_rate)

    model, trace = MCMC_functions.run_MCMC(n, sfs, variable_name, seq_mut_rate=seq_mut_rate, sd_mut_rate=sd_mut_rate,
        mrate_lower=mrate_lower, mrate_upper=mrate_upper, mu=mu, sigma=sigma, folded=folded, ttl_mu=ttl_mu,
        ttl_sigma=ttl_sigma, cores=cores, draws=draws, order=order, tune=tune,
        return_inferencedata=return_inferencedata, target_accept=target_accept, concentration=concentration)
    return model, trace

def analyse_MCMC_results(job_no, trace):
    print(trace.posterior, '\n')
    if parse(version('arviz')) < parse("0.12.0"):
        warnings.warn("Warning: arviz version >= 0.12.1 required.")

    n = len(trace.posterior.probs_dim_0) + 1
    print('Sample size =', n)

    summaryx = arviz.summary(trace)
    print('\n', summaryx)
    csv_name = 'pm_summary_' + job_no + '.csv'
    summaryx.to_csv(csv_name, sep=',')

    stacked = arviz.extract_dataset(trace)
    assert stacked['probs'].values.shape[1] == len(trace.posterior.chain) * len(trace.posterior.draw), \
            'Draws not retrieved from all chains.'
    mut_rate_vars = stacked['mut_rate'].values
    mrate = np.mean(mut_rate_vars)
    print('Mean mutation rate = '.ljust(25), '%.8f' % mrate)
    print('STD mutation rate =  '.ljust(25), '%.8f' % np.std(mut_rate_vars))

    branch_vars = MCMC_functions.multiply_variates(stacked)
    mean_brlens = np.mean(branch_vars, axis=1)
    print('\nEst. branch lengths = '.ljust(25), mean_brlens)
    print('Est. TMRCA = '.ljust(25), '%.1f' % np.sum(mean_brlens))
    print('Est. ttl = '.ljust(25), '%.1f' % np.sum(mean_brlens * np.arange(2, n + 1)))

    # Calculate probabilities for basal splits
    index = stacked['index_0'].values
    counts = np.bincount(index, minlength=n - 1)
    counts = counts / np.sum(counts)
    basal_ = counts + np.flip(counts)
    pol = (n - 1) % 2
    ul = int((n - 1) / 2) + pol
    basal = basal_[: ul]
    if n % 2 == 0:
        basal[-1] = basal[-1] / 2
    np.set_printoptions(precision=4)
    print('Basal split probabilities = '.ljust(25), basal)
    assert np.isclose(np.sum(basal), 1.), 'Basal split probabilities do not total 1'
    csv_name = 'basal_probabilities_' + job_no + '.csv'
    pd.DataFrame(basal).to_csv(csv_name, sep=',')

    fname = 'branch_vars_' + job_no + '.pklz'
    with gzip.open(fname, 'wb') as buff:
        pickle.dump(branch_vars, buff)

    matplotlib.logging.getLogger('matplotlib').setLevel('ERROR')
    matplotlib.logging.getLogger('matplotlib.font_manager').disabled = True
    arviz.rcParams['plot.max_subplots'] = 200
    tp = arviz.plot_trace(trace, compact=False)
    fig = matplotlib.pyplot.gcf()
    fig.savefig('traceplots_' + job_no + '.png')
    return


@click.command()
@click.argument('job_no')
@click.argument('n')
@click.argument('filename')
@click.option('-m', '--mutation_rate', nargs=2, type=float, default=(None, None),
                help='Estimated mutation rate & standard deviation. Required for uninformative prior.')
@click.option('-f', '--simuljobno', default=None,
                help='File of relative branch lengths for multivariate prior.')
@click.option('-l', '--mrate_lims', nargs=2, type=float, default=(None, None),
                help='Limits for uniform distribution on mutation rates. Required for MVN prior.')
@click.option('-fsfs', '--folded', is_flag=True, show_default=True, default=False, help='SFS is folded.')
@click.option('-d', '--draws', type=float, default=50000, help='Number of MCMC draws.')
@click.option('-o', '--order', default="random", help='Parameter for CategoricalGibbsMetropolis step.')
@click.option('-co', '--cores', default=None, help='Number of cores to use')
@click.option('-t', '--tune', default=None)
@click.option('-ta', '--target_accept', default=0.9)
@click.option('-cn', '--concentration', default=1.0)
@click.option('-dir', '--dirx', default='data', help='Directory for data and log files. Default is data.')
def main(job_no, n, filename, mutation_rate, simuljobno, mrate_lims, folded, draws, order,
                cores, tune, target_accept, concentration, dirx):

    start_time = time()
    if not os.path.exists(dirx):
        os.makedirs(dirx)
    if not os.getcwd() == dirx:
        os.chdir(dirx)
    n = int(n)
    draws = int(draws)
    if cores is not None:
        cores = int(cores)
    if tune is not None:
        tune = int(tune)

    return_inferencedata = True
    model, trace = wrap_MCMC(n, filename, mutation_rate, simuljobno, mrate_lims, folded, draws, order,
              cores, tune, target_accept, concentration, return_inferencedata)
    trname = 'pm_trace_' + job_no + '.nc'
    trace.to_netcdf(trname, groups=["posterior", "log_likelihood", "sample_stats"])

    assert len(trace.posterior.draw) == draws, 'Error in number of draws in output.'
    assert len(trace.posterior.probs_dim_0) + 1 == n, 'Error in sample size in output.'

    analyse_MCMC_results(job_no, trace)

    LOGGER = CachingLogger(create_dir=False)
    LOGGER.log_file_path = str(os.path.basename(__file__)) + '_' + job_no + ".log"
    LOGGER.log_args()
    LOGGER.log_message(get_file_hexdigest(__file__), label="Hex digest of script.".ljust(17))
    try:
        LOGGER.log_message(str(os.environ['CONDA_DEFAULT_ENV']), label="Conda environment.".ljust(17))
    except KeyError:
        pass

    LOGGER.input_file(filename)
    LOGGER.output_file('pm_trace_' + job_no + '.nc')
    if simuljobno:
        LOGGER.input_file('relbrlens_' + simuljobno + '.csv')
        LOGGER.input_file('brlens_' + simuljobno + '.csv')
    LOGGER.output_file('pm_summary_' + job_no + '.csv')
    LOGGER.output_file('basal_probabilities_' + job_no + '.csv')
    LOGGER.output_file('branch_vars_' + job_no + '.pklz')

    label = "Imported package".ljust(30)
    LOGGER.log_message('Name = pymc3, version = '      + version('pymc3'),      label=label)
    LOGGER.log_message('Name = theano, version = '     + theano.__version__,    label=label)
    LOGGER.log_message('Name = arviz, version = '      + version('arviz'),      label=label)
    LOGGER.log_message('Name = numpy, version = '      + version('numpy'),      label=label)
    LOGGER.log_message('Name = pandas, version = '     + version('pandas'),     label=label)
    LOGGER.log_message('Name = click, version = '      + version('click'),      label=label)
    LOGGER.log_message('Name = matplotlib, version = ' + version('matplotlib'), label=label)
    LOGGER.log_message('Name = more_itertools, version = ' + version('more_itertools'), label=label)
    LOGGER.log_message('Name = MCMC_functions, version = ' + MCMC_functions.__version__, label="Local module".ljust(30))

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()