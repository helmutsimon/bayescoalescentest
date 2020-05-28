# coding=utf-8


import numpy as np
from math import factorial
import re
from more_itertools import locate
from collections import Counter
import functools

__author__ = "Helmut Simon"
__copyright__ = "© Copyright 2020, Helmut Simon"
__license__ = "BSD-3"
__version__ = "0.0.1"
__maintainer__ = "Helmut Simon"
__email__ = "helmut.simon@anu.edu.au"
__status__ = "Test"

@functools.lru_cache(maxsize=512)
def replace_zero(s, j):
    """Replace jth 0 in string s with a 1. First character is given by j=0 etc."""
    i = list(locate(s, lambda x: x == '0'))[j]
    return s[:i] + '1' + s[i + 1:]


@functools.lru_cache(maxsize=512)
def make_row(s):
    """Convert string of zeros (+) and ones (,) to matrix row, i.e. counting partitions by size."""
    c = re.split('1', s)
    return [int(len(x) + 1) for x in c]


def factorize(i, n):
    """Compute the factoradic decomposition of the integer i, modulo n."""
    quotient = i
    f = list()
    for radix in range(n - 1):
        quotient, remainder = divmod(quotient, radix + 1)
        f.append(remainder)
    return f[::-1]


def derive_tree_matrix(f):
    """Derive tree matrix from the list f. The first element of f is an integer in [0, n-1], the second
    in [0, n-2] and so on."""
    n = len(f) + 1
    s = '0' * (n - 1)
    result = list()
    for j in f:
        s = replace_zero(s, j)
        orow = make_row(s)
        urow = np.bincount(orow, minlength=n + 1)
        urow = urow[1:-1]
        result.append(urow)
    mx = np.stack(result, axis=0)
    return mx


def tree_matrix(i, n):
    """Return tree matrix for given integer"""
    f = factorize(i, n)
    return derive_tree_matrix(f)


def generate_tree_matrices(n):
    """Return listof tree matrices and corresponding probabilities for sample size n."""
    matrices, hashes, probs = list(), list(), list()
    factn = factorial(n - 1)
    c = Counter()
    for i in range(factn):
        mx = tree_matrix(i, n)
        hashm = mx.tostring()
        c[hashm] += 1
    for i in range(factn):
        mx = tree_matrix(i, n)
        hashm = mx.tostring()
        if hashm not in hashes:
            matrices.append(mx)
            hashes.append(hashm)
            probs.append(c[hashm])
    return matrices, probs


def sample_matrices_old(n, size):
    """
        Sample tree matrices for sample size n according to ERM measure.

        Parameters
        ----------
        n: int
            Sample size
        size: int
            Number of samples.

        Returns
        -------
        list
            List of matrices.
        numpy.ndarray
            Probabilities corresponding to matrices
        Counter
            Counts of hashes of matrices.

        """
    c = Counter()
    samples, hashes, matrices = list(), list(), list()
    while len(matrices) < size:
        f = list()
        for i in range(1, n):
            f.append(np.random.choice(i))
        f = f[::-1]
        if f in samples:
            continue
        else:
            samples.append(f)
        mx = derive_tree_matrix(f)
        hashmx = mx.tostring()
        if hashmx not in hashes:
            matrices.append(mx)
            hashes.append(hashmx)
        c[hashmx] += 1
    probs = [c[m.tostring()] for m in matrices]
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    return matrices, probs, c


def sample_matrices(n, size):
    """
        Sample tree matrices for sample size n according to ERM measure.

        Parameters
        ----------
        n: int
            Sample size
        size: int
            Number of samples.

        Returns
        -------
        list
            List of matrices.
        numpy.ndarray
            Probabilities corresponding to matrices
        Counter
            Counts of hashes of matrices.

        """
    c = Counter()
    count, hashes, matrices = 0, list(), list()
    while count < size:
        count += 1
        f = list()
        for i in range(1, n):
            f.append(np.random.choice(i))
        f = f[::-1]
        mx = derive_tree_matrix(f)
        hashmx = mx.tostring()
        if hashmx not in hashes:
            matrices.append(mx)
            hashes.append(hashmx)
        c[hashmx] += 1
    probs = [c[hash] for hash in hashes]
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    matrix_file = [{n: matrices}, {n: probs}]
    return matrix_file


def sample_matching_matrices(n, size, sfs=()):
    """
        Sample tree matrices for sample size n according to ERM measure.

        Parameters
        ----------
        n: int
            Sample size
        size: int
            Number of samples.
        sfs: list
            Site frequency spectrum. Only comp[atible matrices are returned.

        Returns
        -------
        list
            List of matrices.
        numpy.ndarray
            Probabilities corresponding to matrices
        Counter
            Counts of hashes of matrices.

        """
    c = Counter()
    count, hashes, matrices = 0, list(), list()
    while count < size:
        f = list()
        for i in range(1, n):
            f.append(np.random.choice(i))
        f = f[::-1]
        mx = derive_tree_matrix(f)
        reject = False
        if sfs != ():
            for i in range(n - 1):
                if (not np.any(mx[i, :])) and sfs[i]:
                    reject = True
                    break
        if reject:
            continue
        count += 1
        hashmx = mx.tostring()
        if hashmx not in hashes:
            matrices.append(mx)
            hashes.append(hashmx)
        c[hashmx] += 1
    probs = [c[hash] for hash in hashes]
    probs = np.array(probs)
    probs = probs / np.sum(probs)
    matrix_file = [{n: matrices}, {n: probs}]
    return matrix_file