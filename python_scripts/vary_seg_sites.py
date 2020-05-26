# coding=utf-8


"""
    Simulate a coalescent tree using msprime, then apply mutations to it conditioned on a range of segrgating sites and
    generate sample SFS's. Estimate branch lengths from these for comparison with true branch lengths.
    Parameters are job identifier, sample size; population size, sequence length, site mutation rate, population
    growth rate (to present); recombination rate; number of samples, directory (optional), segregating site range
    (optional); number of parallel jobs (for simulation - optional); number of MCMC samples (optional).

    Sample run statements:
    nohup python3 /Users/helmutsimon/repos/bayescoalescentest/python_scripts/vary_seg_sites.py 001 12 5e4 1000 1e-7
                0.0003 -e demog5.csv > vss001.txt &
"""

import numpy as np
import pandas as pd
import gzip, pickle
import os, sys
import msprime
from time import time
import click
import scipy
import pymc3
import theano
from scitrack import CachingLogger, get_file_hexdigest

abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-2])
sys.path.append(projdir)

from shared import msprime_functions, run_MCMC

LOGGER = CachingLogger(create_dir=True)


@click.command()
@click.argument('job_no')
@click.argument('n', type=int)
@click.argument('pop_size', type=float)
@click.argument('length', type=int)
@click.argument('mutation_rate', type=float)
@click.argument('growth_rate', type=float)
@click.option('-r', '--recombination_rate', type=float, default=0)
@click.option('-nes', '--num_emp_samples', type=int, default=1)
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Default is data')
@click.option('-s', '--range', default=(5, 55, 5), help="Segregating sites as np.arange parameters")
@click.option('-j', '--n_jobs', default=5, help='Number of parallel jobs.')
@click.option('-dr', '--draws', default=1000000, help='Number of MCMC samples.')
@click.option('-e', '--events_file', default=None, help='Name of file containing demographic events')
def main(job_no, n, pop_size, length, recombination_rate, mutation_rate, growth_rate, events_file, range, n_jobs,
                        num_emp_samples, dirx, draws):
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
    LOGGER.log_message('Name = ' + msprime.__name__ + ', version = ' + msprime.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + scipy.__name__ + ', version = ' + scipy.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + pymc3.__name__ + ', version = ' + pymc3.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + np.__name__ + ', version = ' + np.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + pd.__name__ + ', version = ' + pd.__version__,
                       label="Imported module".ljust(30))

    pop_size = int(pop_size)
    sample_sfs, matrix, true_branch_lengths, true_tmrca, thom, sfs_mean, seg_site_mean, seg_site_sd = \
        msprime_functions.simulate_tree_and_sample(n, pop_size, length, recombination_rate, mutation_rate,
                                                   growth_rate, events_file, n_jobs, num_emp_samples)
    print(matrix)
    print('TMRCA = ', true_tmrca)
    nseq2 = np.arange(2, n + 1)
    ttl = np.sum(nseq2 * true_branch_lengths)
    print('Total tree length = ', ttl)

    probs = matrix @ true_branch_lengths
    probs = probs / np.sum(probs)

    mfilename = 'data/matrix_details_5-12.pklz'
    with gzip.open(mfilename, 'rb') as mx_details:
        mx_details = pickle.load(mx_details)
    infile = open(mfilename, 'r')
    LOGGER.input_file(infile.name)
    infile.close()
    row = true_branch_lengths.tolist() + [0, true_tmrca, 0]
    rows = [row]
    for seg_sites in np.arange(*range):
        seq_mut_rate = seg_sites / ttl
        sfs = np.random.multinomial(seg_sites, probs)
        print('SFS: ', sfs)
        n_seq = np.arange(1, n)
        thom = np.sum(sfs * n_seq) / (n * seq_mut_rate)
        print('Thomson estimate of TMRCA = ', thom)
        sd_mut_rate = seq_mut_rate * 1e-6
        alpha = 1
        beta = 1e-10
        mcmc_model, trace = run_MCMC.run_MCMC(sfs, seq_mut_rate, sd_mut_rate, mx_details, alpha=alpha, beta=beta,
                          progressbar=False, draws=draws)
        branch_vars = run_MCMC.multiply_variates(trace)
        row = np.mean(branch_vars, axis=1)
        mse = np.linalg.norm(row - true_branch_lengths)
        tmrca = np.sum(row)
        row= list(row)
        row = row + [mse, tmrca, thom]
        rows.append(row)
    index = np.arange(*range).tolist()
    index.insert(0, 'True')
    columns = nseq2.tolist() + ['mse', 'tmrca', 'thom']
    results = pd.DataFrame(rows, index=index,  columns=columns)
    fname = dirx + '/vary_seg_sites_results_' + job_no + '.csv'
    results.to_csv(fname)
    outfile = open(fname, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60.), label="Run duration (minutes)".ljust(50))


if __name__ == "__main__":
    main()


