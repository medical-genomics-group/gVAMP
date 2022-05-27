#!/bin/bash

#SBATCH --ntasks 2
#SBATCH --cpus-per-task 1
#SBATCH --mem 10G
#SBATCH --time 00:10:00
# SBATCH --partition debug

#module load gcc/8 mvapich2 boost
module load gcc mvapich2 boost

ARCH=gcc_mvapich2

srun ../build_$ARCH/gmrm \
--bed-file test.bed \
--dim-file test.dim \
--phen-files test1.phen,test1_bis.phen,test1_nas.phen,test2.phen \
--bim-file test.bim \
--ref-bim-file test.bim \
--predict \
--out-dir test1
