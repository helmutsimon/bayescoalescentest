# bayescoalescentest
This repository contains Python files and data relating to the manuscript: Simon, H. and Huttley, G., 2021 *Bayesian Inference of Joint Coalescence Times for Sampled Sequences*. bioRxiv doi = 10.1101/2021.07.23.453461.

## 1. INSTALLATION (for PyMC3 3.11.2)

### 1.1 Create Conda environment and install PyMC3 (see https://discourse.pymc.io/t/pymc3-final-stable-release-on-conda/8172)

```
conda create --name pymc3.11.2 python=3.9
conda activate pymc3.11.2
conda install -c conda-forge mkl pymc3 theano-pymc
pip3 install jupyter
```

The following packages should be installed:

cogent3, scipy, numpy, pandas, more-itertools, click, scitrack, matplotlib, seaborn, msprime (if simulations are required)

### 1.2 Clone bayescoalescentest repository from GitHub.

## 2. RUN BAYESIAN MODEL (python_scripts/run_bayesian_model.py)


run_bayesian_model.py requires an input file consisting of a Python list, pickled and compressed with gzip. The list contains two elements. The first is a site frequency spectrum comprising a list of integers.
The second is a list of true branch lengths comprising a list of real numbers. True branch lengths are only available for data produced from a simulation, otherwise the second element of the list should be None. 
The input file is zipped and pickled (see nbks/Pre-process_Nuu_Chah_Nulth_data.ipynb for an example of how to create such a file).
The file should reside in ./data directory (or use -dir option for a different directory name).

Mandatory parameters for run_bayesian_model.py are a job_number (used to label output), the sample size, n and the name of the input file described above.

run_bayesian_model.py can use an uninformative prior or a model-based (multivariate normal or MVN) prior.

For the uninformative prior, the following parameter is also required:

    -m   mutation rate: mean and standard deviation of estimate of **sequence** mutation rate, e.g. -m 2.38e-4 1e-5

For the MVN prior, the following parameters are required:

    -f   identifier for file containing samples of relative branch lengths and branch lengths used to generate the parameters for the MVN prior.
	For example, if the identifier is XXX, these files must be called relbrlens_XXX.csv and brlens_XXX.csv and located in the data directory.
	The script simulate_multiples.py is supplied to generate such files using msprime.
	If this parameter is not given, this indicates that the uninformative prior is used.

    -l	lower and upper limits for **sequence** mutation rate (uniform prior distribution), e.g. -l 0 1e-3


Other parameters control how the MCMC process is run. The most useful, because PyMC3 output may suggest they be modified to improve convergence, are:

    -d  number of MCMC draws, default = 50000.
    -ta target acceptance, default=0.9.
    -c  number of cores, default set by system.
    -t  number of tuning steps, default is 20% of number of draws.

Remaining parameters are primarily for experimental purposes.

The primary output is a file containing samples from the posterior distribution of branch lengths. The primary tool to analyse/visualise this output is the Jupiter notebook 'plots_and_tables/plots and tables.ipynb'.

## 3. DATA SIMULATION (python_scripts/simulate_population.py)


simulate_population.py allows simulated data to be generated using msprime, creating an input file for run_bayesian_model.py. It generates an output file containing a list of SFS and 
the true branch lengths of the tree generated by the simulation. Mandatory parameters are:

    - job_no  (used to label output)
    - population size
    - **site** mutation rate
    - length  (length of genome segment being analysed in base pairs
    - initial population growth rate per generation
    - sample size

Optional parameters are:

    -d  directory name for data, default is 'data'
    -e  events_file. This is used for more complex demographies. It is a .csv file with columns headed 'time', 'size' and 'rate'. For each row the 'time' field is the number of generations into the past at which a demographic change occurred. 'Size' and 'rate' are the population size and growth rate from that time point (going backward into past).  The initial population size and growth rate (from time=0) are given by the mandatory parameters.

## 4. USE INFORMATIVE (MVN) PRIOR (simulate_multiples.py)


simulate_multiples.py generates csv files of samples of branch lengths and relative branch lengths as input to run_bayesian_model.py if an MVN prior is used.
Other simulation software could in principle be used for this purpose.
Mandatory parameters for simulate_multiples.py are:

    - job_no  (used to label output)
    - sample size
    - population size
    - **site** mutation rate
    - length  (length of genome segment being analysed in base pairs
    - population growth rate per generation
    - number of samples to generate
    
Optional parameters are:

    -d  directory name for data, default is 'data'
    -e  events_file. As for simulate_population.py

## 5. PRESENT AND VISUALISE RESULTS (plots_and_tables/plots and tables.ipynb)


    This Jupiter notebook allows display of posterior distributions of branch lengths and coalescent times in the form of line graphs and heat maps. It also calculates expected values, as well as calculating and graphing 'ancestral probabilities'.
    
