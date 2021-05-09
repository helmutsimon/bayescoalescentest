# coding=utf-8


import sys
import msprime
import numpy as np
import pandas as pd
from collections import Counter
from joblib import Parallel, delayed
from scitrack import CachingLogger

LOGGER = CachingLogger(create_dir=True)



def get_next_coalescent(tree, ancestors):
    """Gets next coalescent pair and time (from present) of coalescent."""
    pair_mrca_list, pair_mrca_times, pair_list = list(), list(), list()
    for i in ancestors:
        for j in ancestors:
            if i < j:
                pair_mrca = tree.get_mrca(i, j)
                pair_mrca_list.append(pair_mrca)
                pair_mrca_times.append(tree.get_time(pair_mrca))
                pair_list.append([i, j])
    coalescence_time = min(pair_mrca_times)
    index = pair_mrca_times.index(coalescence_time)
    pair = pair_list[index]
    mrca = pair_mrca_list[index]
    ancestors = list(set(ancestors) - set(pair))
    ancestors.append(mrca)
    return ancestors, coalescence_time


def get_matrix_row(n, tree, ancestors, sub_sample):
    """Calculate row of tree matrix referring to coalescent level (k)."""
    prev_length = len(ancestors)
    ancestors, coalescence_time = get_next_coalescent(tree, ancestors)
    current_length = len(ancestors)
    assert current_length == prev_length - 1, 'Error: Nodes not decremented by 1'
    matrix_row = np.zeros(n)
    for node in ancestors:
        descendants = set([i for i in tree.leaves(node)]) & set(sub_sample)
        matrix_row[len(descendants) - 1] += 1
    return ancestors, matrix_row, coalescence_time


def analyse_tree(n, tree, sub_sample=None):
    """Get matrix and coalescence times for sample tree."""
    if sub_sample is None:
        root = tree.get_root()
        leaves = [i for i in tree.leaves(root)]
        sub_sample = leaves
    assert all([tree.is_leaf(i) for i in sub_sample]), 'Error: Internal nodes passed to get_matrix'
    ancestors = sub_sample     #i.e. the set of ancestors at each coalescence event
    matrix, coalescence_times = list(), list()
    while len(ancestors) > 1:
        ancestors, matrix_row, coalescence_time = get_matrix_row(n, tree, ancestors, sub_sample)
        matrix_row = np.append(matrix_row, np.zeros(n - len(matrix_row)))         #faster than np.pad
        matrix.append(matrix_row)
        coalescence_times.append(coalescence_time)
    matrix = np.array(matrix)
    r0 = np.zeros(n)
    r0[0] = n
    r0.shape = (1, n)
    mx1 = np.append(r0, matrix, axis=0)
    mx2 = mx1[:-1, :-1]
    matrix = np.flipud(mx2)
    matrix = matrix.T
    return matrix, coalescence_times


def generate_sample_coalescence_times(i, tree, n, variant_array):
    """Generate sample from population and return the associated coalescence times and sfs."""
    np.random.seed(i * 7)
    root = tree.get_root()
    leaves = [i for i in tree.leaves(root)]
    sub_sample = np.random.choice(leaves, size=n, replace=False)
    matrix, coalescence_times = analyse_tree(n, tree, sub_sample)
    sample_sequences = variant_array[:, sub_sample]
    occurrences = np.sum(sample_sequences, axis=1)
    sample_sfs = Counter(occurrences)
    sample_sfs = [sample_sfs[i] for i in range(1, n)]
    return matrix, sample_sfs, coalescence_times


def generate_sample_sfs(i, leaves, n, variant_array):
    """Generate sample from population and return the associated sfs."""
    np.random.seed(i * 7)
    sub_sample = np.random.choice(leaves, size=n, replace=False)
    sample_sequences = variant_array[:, sub_sample]
    occurrences = np.sum(sample_sequences, axis=1)
    sample_sfs = Counter(occurrences)
    sample_sfs = [sample_sfs[i] for i in range(1, n)]
    return sample_sfs


def read_demographic_history(filename, print_history):
    if filename is None:
        return list()
    if isinstance(filename, str):
        if filename[-4:] != '.csv':
            filename = filename + '.csv'
            infile = open(filename, 'r')
            LOGGER.input_file(infile.name)
            infile.close()
        demo_parameters = pd.read_csv(filename)
    elif isinstance(filename, pd.DataFrame):
        demo_parameters = filename
    else:
        raise ValueError('Events_file parameter wrong type: ' + str(type(filename)))
    demographic_events = list()
    for index, row in demo_parameters.iterrows():
        time = row['time']
        initial_size = row['size']
        growth_rate = row['rate']
        if print_history:
            print(time, initial_size, growth_rate)
        ppc = msprime.PopulationParametersChange(time=time, initial_size=initial_size, growth_rate=growth_rate)
        demographic_events.append(ppc)
    return demographic_events


def generate_population_tree(pop_size, sample_size, length, recombination_rate,
                             mutation_rate, growth_rate, events_file, print_history=True):
    """Generate a population using msprime. Return tree sequence and variant array (rows are mutations, columns
    are sample members."""
    demographic_events = read_demographic_history(events_file, print_history)
    population_configurations = [msprime.PopulationConfiguration(growth_rate=growth_rate,
                                                  initial_size=pop_size, sample_size=sample_size)]
    demographic_hist = msprime.DemographyDebugger(demographic_events=demographic_events,
                                                  population_configurations=population_configurations)
    if print_history:
        demographic_hist.print_history()
    sys.stdout.flush()
    tree_sequence = msprime.simulate(Ne=pop_size, population_configurations=population_configurations, demographic_events=
                demographic_events, length=length, recombination_rate=recombination_rate, mutation_rate=mutation_rate)
    shape = tree_sequence.get_num_mutations(), tree_sequence.get_sample_size()
    variant_array = np.empty(shape, dtype="u1")
    for variant in tree_sequence.variants():
        variant_array[variant.index] = variant.genotypes
    return tree_sequence, variant_array


def simulate_tree_and_sample(sample_size, pop_size, length, recombination_rate, mutation_rate, growth_rate, events_file,
                  n_jobs, num_emp_samples):
    """Simulate a (large) population and take a random sample from it. We return the details of this sample tree
    (sfs, branch lengths and matrix) as well as averages from other population samples to provide data for empirical
    prior. In the first statement we generate a tree for the entire population size."""
    tree_sequence, variant_array = generate_population_tree(pop_size, sample_size, length, recombination_rate,
                                                            mutation_rate, growth_rate, events_file)
    tree = next(tree_sequence.trees())
    matrix, sample_sfs, coalescence_times = generate_sample_coalescence_times(1, tree, sample_size, variant_array)
    coalescence_times = np.array(coalescence_times)
    shifted = np.concatenate((np.zeros(1), coalescence_times))[:-1]
    true_branch_lengths = np.flipud(coalescence_times - shifted)
    true_tmrca = np.sum(true_branch_lengths)
    n_seq = np.arange(1, sample_size)
    thom = np.sum(sample_sfs * n_seq) / (sample_size * length * mutation_rate)
    return sample_sfs, matrix, true_branch_lengths, true_tmrca, thom


def get_mode(vars):
    hist, edges = np.histogram(vars, bins=500)
    i = np.argmax(hist)
    return (edges[i] + edges[i + 1]) / 2

