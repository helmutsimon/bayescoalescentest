# coding=utf-8


"""
Sample tree matrices according to the ERM measure.
Parameters are job number, sample size (n), number of samples to return (size), and output filename.
Sample run statement is:
nohup python3 ~/repos/bayescoalescentest/python_scripts/sample_tree_matrices.py stm001 15 1e6 matrices_smt001 > o.txt &

"""

import numpy as np
import os
import sys
from time import time
import gzip, pickle
import re
import more_itertools
from collections import Counter
import click
from scitrack import CachingLogger, get_file_hexdigest

abspath = os.path.abspath(__file__)
projdir = "/".join(abspath.split("/")[:-2])
sys.path.append(projdir)

from shared import tree_matrix_computation

LOGGER = CachingLogger(create_dir=True)


@click.command()
@click.argument('job_no')
@click.argument('n', type=int)
@click.argument('size', type=float)
@click.argument('sfs', type=int, nargs=-1)
@click.option('-f', '--filename', default=None)
@click.option('-d', '--dirx', default='data', type=click.Path(),
              help='Directory name for data and log files. Defaults is data')
def main(job_no, n, size, sfs, filename, dirx):
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
    LOGGER.log_message('Name = ' + np.__name__ + ', version = ' + np.__version__, label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + re.__name__ + ', version = ' + re.__version__, label="Imported module".ljust(30))
    LOGGER.log_message('Name = ' + more_itertools.__name__ + ', version = ' + more_itertools.__version__,
                       label="Imported module".ljust(30))
    LOGGER.log_message('Name = tree_matrix_computation' + ', version = ' + tree_matrix_computation.__version__,
                       label="Imported module".ljust(30))

    size = int(size)
    matrix_file = tree_matrix_computation.sample_matrices(n, size, sfs)
    matrices = matrix_file[0][n]
    probs = matrix_file[1][n]
    assert len(matrices) == len(probs), 'Lists of matrices and probabilities returned not equal in length.'
    LOGGER.log_message(str(len(matrices)), label="Length of matrix list returned".ljust(30))
    print((time() - start_time) / 60)
    sys.stdout.flush()
    hashmxs = [mx.tostring() for mx in matrices]
    mx_counts = Counter(hashmxs)
    LOGGER.log_message(str(len(mx_counts)), label="Number of different matrices".ljust(30))
    LOGGER.log_message('%.3e' % max(probs), label="Frequency most common matrix".ljust(30))

    if filename is None:
        filename = dirx + '/mxs_' + job_no
    with gzip.open(filename, 'wb') as outfile:          # See https://stackoverflow.com/questions/33562394 for gzip issue
        pickle.dump(matrix_file, outfile)
    outfile = open(filename, 'r')
    LOGGER.output_file(outfile.name)
    outfile.close()

    duration = time() - start_time
    LOGGER.log_message("%.2f" % (duration / 60), label="Run duration (minutes)".ljust(30))


if __name__ == "__main__":
    main()

