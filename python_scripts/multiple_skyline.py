# coding=utf-8

"""
    This script performs a number of Wright-Fisher simulations, then calculates skyline pplots and averages them.

    Parameters are job no, samp[le size, population size, mutation rate, sequence length, growth rate,
    number of trials (simulations), demographic events filename, number of time intervals, number of processor cores,
    number of MCMC draws (-dr optional) and directory for logs and data (-d optional).

    A sample run statement is:

    /Users/helmutsimon/repos/bayescoalescentest/python_scripts/multiple_skyline.py cp001 5.00E+04 1e-7 1000 5 8 -dr 20000
    > cp001.txt &

    Date: 5 June 2020."""

import numpy as np
import pandas as pd
import os, sys
import pymc3
import theano
import msprime
import arviz
from arviz import summary
from collections import Counter
from bisect import bisect
from time import time
import click
from scitrack import CachingLogger, get_file_hexdigest


abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-2])
sys.path.append(projdir)

from shared import MCMC_functions, msprime_functions


LOGGER = CachingLogger(create_dir=True)


def compute_skyline_variates(brvars, ntimes, tlim):
    """
    Calculation for skyline plot.
    Sample from quasi-posterior distribution of population sizes at given time intervals.

    brvars: np.array
        Array of branch variates output by MCMC.
    ntimes: integer
        Number of intervals
    tlim: float
        maximum time (approx. TMRCA)

    Returns
    -------
    pandas.DataFrame
        Samples from quasi-posterior distribution of population sizes by time interval (index).

    """
    n = brvars.shape[0] + 1
    print(n)
    r = tlim / ntimes
    branches_rev = np.flipud(brvars)
    coal_times = np.cumsum(branches_rev, axis=0)
    draws = coal_times.shape[1]    # number of MCMC variates
    Nvars = {t:list() for t in range(ntimes)}
    for var_ix in range(draws):
        for t in range(ntimes):                             # iterating over time intervals
            next_coal_time = bisect(coal_times[:,var_ix], t * r)
            k = n - next_coal_time                         # k is number of ancestors at time t
            if k == 1:
                k = 2
            brlen = brvars[k - 2, var_ix]
            Nvar = k * (k - 1) * brlen / 4
            Nvars[t].append(Nvar)
    result = pd.DataFrame.from_dict(Nvars, orient='index', dtype=float)
    return result



@click.command()
@click.argument('job_no')
@click.argument('n', type=int)
@click.argument('pop_size', type=float)
@click.argument('mutation_rate', type=float)
@click.argument('length', type=float)
@click.argument('growth_rate', type=float)
@click.argument('num_replicates', type=int)
@click.option('-e', '--events_file', default=None, help='Name of file containing demographic events')
@click.option('-nt', '--ntimes', default=200)
@click.option('-co', '--cores', default=None)
@click.option('-dr', '--draws', default=20000, help='Number of MCMC samples.')
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
def main(job_no, n, pop_size, mutation_rate, length, growth_rate, num_replicates, events_file, ntimes, cores,
                    draws, dirx):
    start_time = time()
    np.set_printoptions(formatter={'float': "{0:0.3G}".format})
    if not os.path.exists(dirx):
        os.makedirs(dirx)
    length = int(length)
    if cores is not None:
        cores = int(cores)
    pop_size = int(pop_size)
    length = int(length)
    n_seq = np.arange(1, n)
    np.set_printoptions(precision=3)
    sfs_array = list()
    population_configurations = [msprime.PopulationConfiguration(growth_rate=growth_rate,
                                                                 initial_size=pop_size, sample_size=n)]
    demographic_events = msprime_functions.read_demographic_history(events_file, True)
    demographic_hist = msprime.DemographyDebugger(demographic_events=demographic_events,
                                                  population_configurations=population_configurations)
    demographic_hist.print_history()
    sys.stdout.flush()
    replicates = msprime.simulate(population_configurations=population_configurations, demographic_events=
                demographic_events, mutation_rate=mutation_rate, length=length, num_replicates=num_replicates)
    for j, tree_sequence in enumerate(replicates):
        shape = tree_sequence.get_num_mutations(), tree_sequence.get_sample_size()
        variant_array = np.empty(shape, dtype="u1")
        for variant in tree_sequence.variants():
            variant_array[variant.index] = variant.genotypes
        occurrences = np.sum(variant_array, axis=1)
        sfs = Counter(occurrences)
        sfs = [sfs[i] for i in range(1, n)]
        sfs_array.append(sfs)
    sfs_array = np.array(sfs_array)
    thom_ests = np.sum((sfs_array * n_seq) / (n * length * mutation_rate), axis=0)
    thom_mean = np.mean(thom_ests)
    print('\nMean Thomson TMRCA estimate    = '.ljust(25), "%.3G" % thom_mean)
    seq_mutation_rate = mutation_rate * length
    sd_mutation_rate = seq_mutation_rate * 1e-6

    summaries, skyline_variates = list(), list()
    for sfs, i in zip(sfs_array, range(num_replicates)):
        mcmc_model, trace = MCMC_functions.run_MCMC_Dirichlet(sfs, seq_mutation_rate, sd_mutation_rate,
                                            cores=cores, draws=draws)
        summaryx = summary(trace)
        summaries.append(summaryx)
        branch_vars = MCMC_functions.multiply_variates(trace, 'probs')
        result = compute_skyline_variates(branch_vars, ntimes, thom_mean)
        mean_Ns = np.mean(result, axis=1)
        skyline_variates.append(mean_Ns)
    skyline_variates = np.array(skyline_variates)
    fname = dirx + '/skyvars_' + job_no + '.csv'
    np.savetxt(fname, skyline_variates, delimiter=",")

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
    LOGGER.log_message('Name = ' + arviz.__name__ + ', version = ' + arviz.__version__,
                       label="Imported module".ljust(25))
    LOGGER.log_message('Name = ' + theano.__name__ + ', version = ' + theano.__version__,
                       label="Imported module".ljust(25))
    LOGGER.log_message('Name = ' + msprime.__name__ + ', version = ' + msprime.__version__,
                       label="Imported module".ljust(25))
    summaries = pd.concat(summaries, ignore_index=True)
    csv_name = dirx + '/pm_summaries_' + job_no + '.csv'
    summaries.to_csv(csv_name, sep=',')
    summary_file = open(csv_name, 'r')
    LOGGER.output_file(summary_file.name)
    summary_file.close()
    LOGGER.log_message("%.3G" % ((time() - start_time) / 60), label="Time (minutes)".ljust(25))


if __name__ == "__main__":
    main()