#!/bin/bash

#SBATCH --ntasks 2
#SBATCH --cpus-per-task 1
#SBATCH --mem 10G
#SBATCH --time 00:10:00

module load intel intel-oneapi-mpi

ARCH=intel_ioampi

srun ../build_$ARCH/gmrm \
--bed-file test.bed \
--dim-file test.dim \
--phen-files test1.phen,test1_bis.phen,test1_nas.phen,test2.phen \
--bim-file test.bim \
--ref-bim-file test.bim \
--predict \
