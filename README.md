## Table of contents
* [General info](#general-info)
* [Setup and supported options](#setup)

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
| `bed-file` | filepath to .bed file including the .bed extension |
| `bed-file-test` | filepath to .bed file reserved for testing purposes |
| `phen-files` | path to file containing phenotype of interest (only 1 phenotype supportet at the moment) |
| `phen-files-test` | path to phenotype file reserved for testing purposes |
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
| `use-adap-damp` | indicates wheather or not the algorithm uses adaptive damping procedure (0 or 1) |
| `store-pvals` | indicates wheather or not the algorithm stores p-values from association tests of each of the markers |

