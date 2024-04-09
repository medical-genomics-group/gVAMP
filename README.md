## Table of contents
* [General info](#general-info)
* [Setup and supported options](#setup-and-supported-options)
* [Docker setup](#docker-setup)

## General info
This repository contains a software implementation of Vector Approximate Message Passing algorithm suitable for doing inference in Genome-Wide Association Studies (GWAS).  


## Setup and supported options

An example of a slurm script that can be used to compile and run the software:

```

module purge

module load gcc openmpi boost

module list 

loc={a path to cpp_vamp folder}


mpic++ /$loc/main_real.cpp /$loc/vamp.cpp /$loc/utilities.cpp /$loc/data.cpp /$loc/options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  /$loc/main_real.exe

mpirun -np {number of MPI workers} /$loc/main_real.exe [input options]

```

Currently supported input options to C++ version of the code:

| Input option | Description |
| --- | --- |
| `run-mode` | 'infere' / 'test' / 'both' |
| `bim-file` | filepath to .bim file including the .bim extension |
| `bed-file` | filepath to .bed file including the .bed extension |
| `bed-file-test` | filepath to .bed file reserved for testing purposes |
| `phen-files` | path to file containing phenotype of interest (only 1 phenotype supportet at the moment) |
| `phen-files-test` | path to phenotype file reserved for testing purposes |
| `cov-file` | filepath to .cov file including the .cov extension (covariates in a probit model) |
| `N` | number of individuals included in the inference process |
| `N-test` | number of individuals included in a testing dataset |
| `Mt-test` | total number of markers included in a testing dataset |
| `Mt` | total number of markers included in the infrence process |
| `out-dir` | output directory for the signal estimates |
| `out-name` | name of the output file |
| `iterations` | maximal number of iterations to be performed |
| `num-mix-comp` | number of gaussian mixture components used (including delta spike at zero) |
| `CG-max-iter` | maximal number of iteration used in conjugate gradient method for solving linear systems |
| `probs` | initial prior mixture coefficients (separated by comma, must sum up to 1) |
| `vars` | initial prior variances (separated by comma) |
| `rho` | initial value of damping factor |
| `EM-err-thr` | relative error threshold within expectation maximization |
| `EM-max-iter` | maximal number of iterations of expectation maximization procedure |
| `stop-criteria-thr` | relative error threshold within expectation maximization |
| `model` | regression model that describes a relationship between effect sizes and phenotypes ('linear' or 'bin_class') |
| `store-pvals` | indicates wheather or not the algorithm stores p-values from association tests of each of the markers |
| `test-iter-range` | indicates the iteration range for which R2 on a test set is calculated |
| `use_XXT_denoiser` | indicates whether or not a denoiser should be calculated based on formula with inversion of NxN matrix or not |
| `use_lmmse_damp` | indicates whether or not damping should be used in LMMSE step calculation |
| `h2` | heritability value used in simulations  |
| `CV` | number of causal variants used in simulations |
| `true_signal_file` | path to file containing true value of the signal |
| `alpha` | scaling factor applied in the normalization of a genotype matrix |
| `learn-vars` | indicates wheather or not prior variances are learnt |
| `C` | indicates the number of covariates in a probit model |
| `alpha-scale` | columns of genotype matrix are scaled with std^(-alpha-scale) |
| `gam1-init` | initial value of gam1 in the restart scenario |
| `gamw-init` | initial value of gamw in the restart scenario |
| `init-est` | indicates whether or not we initialze gVAMP with an estimate provided in estimate-file |
| `use-freeze` | indicates whether or not to freeze certain position in the inference process |
| `freeze-index-file` | a file containing 0/1 assigned to the indices that are being freezed in the inference process |
| `seed` | defines a seed for the gVAMP run (must be a non-negative integer) |

## Docker setup
Running the docker application step-by-step.

First, clone the repository to the local workspace.
```
git clone https://github.com/medical-genomics-group/gVAMP.git
```
Change the working directory.
```
cd gVAMP
```
Modify ``run_gvamp.sh`` script and set paths to your data. In the run time, the data directory can be mounted to a specific location in the container.

Then, build docker image.
```
docker build -t gvamp .
```
Finnaly, run docker application in detached mode (``-d``). ``-v`` allows mounting local data to your container. Note, that you can specify arbitrary mounted directory instead of ``/home/mnt``, but this has to be consistent with the paths specified in ``run_gvamp.sh``.
```
docker run -d -v {local data directory}:/home/mnt gvamp:latest
```
By default, the output logs will show in ``{local data directory}/output/gvamp.log``.