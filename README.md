# bayescoalescentest

**_See Discussions tab for recent changes._**

## 1. OVERVIEW
bayescoalescentest estimates several parameters for coalescent trees using Bayesian inference on genomic data sampled from a study population (sample size = *n*). The data  may take the form of a folded or unfolded site frequency spectrum (SFS). In the former case, the ancestral states are unknown. In Bayesian style, samples from a posterior distribution are generated for estimated quantities and expected values of these may be taken as point estimates for those quantities.

The software assumes selective neutrality. The coalescent process generating the data can then be taken as determined by the demographic history of the population. The software can be used without any knowledge of this demographic history, by using an uninformative prior distribution. Alternatively, any demographic model that can be simulated (e.g. by Hudson ms or derivatives such as msprime or msms) can also be used to generate a prior distribution using a multivariate normal (MVN) approximation (model-based prior). This repository uses msprime (https://github.com/tskit-dev/msprime/blob/main/README.md) for all simulations. The application of these prior distributions to coalescent inference is novel and extends the scope of the software beyond the standard assumption that data is generated by a Wright-Fisher (constant population size) population genetic model. See below and bioRxiv paper for more on these prior distributions.

The coalescent quantities that can be directly and simultaneously inferred are:

* branch lengths, specifically the joint distribution of branch lengths;
* the basal split (the sizes of the two subsets into which the sample is split by the first bifurcation occurring at the root of the coalescent tree);
* the total tree length; and
* the mutation rate. This can be estimated from a uniform distribution if a model-based prior is used. If an uninformative prior is used, an estimate of mutation rate, together with a standard error is required as input. The mutation rate is then re-estimated by the software.

Other quantities may be derived from the above either by arithmetic operations on posterior distributions, or by finding the mean of some calculated quantity over the posterior distribution:
* coalescent times are a cumulative sum of branch lengths;
* the ancestral process. This is a set of functions of time, indexed by *k* for *k*=1,...n, which give the probability that the sample had *k* ancestors at time *t* before the present;
* the time to the most recent common ancestor (TMRCA). The TMRCA can be estimated as the sum of branch lengths. This can become unreliable if the sample size is high and the number of segregating sites does not increase at the same rate, as is typical of estimates where additional samples increase the number of parameters (in this case branch lengths) being estimated. The result is that the prior distribution comes to have a stronger influence on the posterior distribution, as compared to the relatively sparse data. This is unfortunately often the case with data from natural populations. For sample sizes up to 20, say, results are more often reasonable. It is recommended that the unbiassed estimate of TMRCA introduced by Russell Thomson [[1]](#1) be used as a comparison, and this estimate is generally included in output. Such problems are lessened if a well-founded model-based prior is used;
* probabilities may be calculated for other ad hoc hypotheses: for example, that one particular branch length is greater than another. Like the computation of ancestral probabilities, this is possible because we sample from the joint distribution of branch lengths.

The software also provides methods for visualising the posterior distributions.

Theory and examples are available in the manuscript: Simon, H. and Huttley, G., 2021 *Bayesian Inference of Joint Coalescence Times for Sampled Sequences*. bioRxiv doi = 10.1101/2021.07.23.453461.

## 2. INSTALLATION (for PyMC3 3.11.5)

Create Conda environment and install PyMC3. The following is for MacOS on Intel.  (based on Installation section of https://pypi.org/project/pymc3/3.11.5/)

```
conda create -c conda-forge -n pymc3.11.5 python pymc3 theano-pymc mkl mkl-service arviz=0.12.1 python=3.9
conda activate pymc3.11.5
```

Installation of additional packages is required as follows:

```
conda install click scitrack seaborn jupyter
python -m pip install more_itertools importlib_metadata
```

The following may be required to get the new Conda environment to appear in Jupyter's drop-down list of environments:

```
jupyter kernelspec remove python3
conda install -c anaconda ipykernel
python -m ipykernel install --user --name=pymc3.11.5
```

Then clone bayescoalescentest repository from GitHub.

The software was tested with the following software versions:
 - python 3.9.16
 - pymc3 3.11.5
 - theano 1.1.2
 - arviz 0.12.1 (now minimum requirement)
 - numpy 1.22.1

## 3. USING THE SOFTWARE

### 3.1 Running inference with the model (python_scripts/run_bayesian_model.py)

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

    -l	lower and upper limits for **sequence** mutation rate, e.g. -l 1e-3 2e-3

If data is supplied in the form of a folded SFS, the flag -fsfs is required. 

Other parameters control how the MCMC process is run. The most useful, because PyMC3 output may suggest they be modified to improve convergence, are:

    -d  number of MCMC draws, default = 50000.
    -ta target acceptance, default=0.9.
    -c  number of cores, default set by system.
    -t  number of tuning steps, default is 20% of number of draws.

Remaining parameters are primarily for experimental purposes.

The primary output is a file containing samples from the posterior distribution of branch lengths. The primary tool to analyse/visualise this output is the Jupiter notebook 'plots_and_tables/plots and tables.ipynb'.

### 3.2 DATA SIMULATION (python_scripts/simulate_population.py)

simulate_population.py allows simulated data to be generated using msprime, creating an input file for run_bayesian_model.py. It generates output files containing a Python list withntwo items: the SFS and the true branch lengths of the tree generated by the simulation. A file is created for the unfolded SFS and for the folded SFS. The latter has the letter 'f' appended to its name. Mandatory parameters are:

    - job_no  (used to label output)
    - population size
    - site mutation rate
    - length  (length of genome segment being analysed in base pairs
    - initial population growth rate per generation
    - sample size

Optional parameters are:

    -d  directory name for data, default is 'data'
    -e  events_file. This is used for more complex demographies. It is a .csv file with columns headed 'time', 'size' and 'rate'. For each row the 'time' field is the number of generations into the past at which a demographic change occurred. 'Size' and 'rate' are the population size and growth rate from that time point (going backward into past).  The initial population size and growth rate (from time=0) are given by the mandatory parameters.

### 3.3 GENERATE SAMPLE DATA FOR MODEL-BASED (MVN) PRIOR (simulate_multiples.py)


simulate_multiples.py generates csv files of samples of branch lengths and relative branch lengths as input to run_bayesian_model.py if an MVN prior is used.
Other simulation software could in principle be used for this purpose.
Mandatory parameters for simulate_multiples.py are:

    - job_no  (used to label output)
    - sample size
    - population size
    - site mutation rate
    - length  (length of genome segment being analysed in base pairs
    - population growth rate per generation
    - number of samples to generate
    
Optional parameters are:

    -d  directory name for data, default is 'data'
    -e  events_file. As for simulate_population.py

### 3.4 PRESENT AND VISUALISE RESULTS (plots_and_tables/plots and tables.ipynb)

This Jupiter notebook allows display of posterior distributions of branch lengths and coalescent times in the form of line graphs and heat maps. The latter allows results in the form of posterior distributions for large sample sizes to be displayed in an intelligible way. Notebook cells also calculate expected values, as well as calculating and graphing 'ancestral probabilities'.
    
## 4. RELATION TO PREVIOUS METHODS

The basic method is comparable to the importance sampling method of R.C. Griffiths, S. Tavare and others [[2]](#2) [[3]](#3), which is the basis of the widely-known software program GENETREE [[4]](#4). The GENETREE method effectively generates a posterior distribution of coalescent trees by a stochastic process which works backward in time from unfolded SFS data. Quantities such as branch lengths are estimated by averaging over this ensemble of coalescent trees. The stochastic process generally assumes a Wright-Fisher population model, although extensions have been developed (see [[5]](#5) section 8.4.1.) The major benefit of the current Bayesian approach is to remove this constraint and allow either an uninformative prior distribution, for use where the population model is unknown, or a prior distribution based on any desired population model. Other benefits are the ability to sample tree structures, which allows inference of the basal split; inference from a folded SFS; and the use of a powerful open-source MCMC software (PyMC3) which continues to be improved by an active developer community.

The method of sampling trees and calculating likelihoods was first published in [[6]](#6).

## 5. PRIOR DISTRIBUTIONS

Prior distributions depend on the concept that for a tree corresponding to a given sample size *n*, the probabilities that a mutation in the tree has *i* descendants for *i* = 1, ..., *n*-1 form an (*n*-1)-vector. The set of such vectors for all valid trees form an (*n*-2)-simplex, which is a subset of the unit (*n*-2)-simplex. The "uninformative" distribution, perhaps better termed a diffuse distribution, is equivalent to the uniform distribution on this simplex. Model-based prior distributions are generated by generating a sample of variates in this simplex by simulation. The variates are then transformed into Euclidean (*n*-1)-space using a "stickbreaking" transform and approximated by a multivariate normal (MVN) distribution. Variates sampled from this distribution are then transformed back into the simplex to form the prior distribution.


## References
<a id="1">[1]</a> 
Thomson, R. et a. (2000). 
Recent common ancestry of human Y chromosomes: Evidence from DNA sequence data. 
Proceedings of the National Academy of Sciences, 97(13), 7360-7365.

<a id="2">[2]</a> 
Griffiths, R.C. and Tavare, S. (1994)
Simulating probability distributions in the coalescent.
Theoretical Population Biology 46, 131-159.

<a id="3">[3]</a> 
Griffiths, R.C. and Tavare, S. (1994)
Ancestral Influence in population genetics.
Statistical Science 9(3), 307-319.

<a id="4">[4]</a> 
Bahlo M. and Griffiths R.C. (2000)
Inference from gene trees in a subdivided population.
Theor Popul Biol. 57(2), 79-95.

<a id="5">[5]</a> 
Wakeley, J. (2009). 
Coalescent theory: an introduction. 
Roberts & Company, Greenwood Village Colorado.

<a id="6">[6]</a> 
Sainudiin, R., Thornton, K., Harlow, J., Booth, J., Stillman, M., Yoshida, R., Griffiths, R., Gil, M., and Donnelly, P. (2011). 
Experiments with the site frequency spectrum. 
Bulletin of Mathematical Biology, 73(4): 829-872.
