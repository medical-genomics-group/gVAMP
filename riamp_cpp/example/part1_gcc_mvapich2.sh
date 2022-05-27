#!/bin/bash

#SBATCH --ntasks 2
#SBATCH --cpus-per-task 1
#SBATCH --mem 10G
#SBATCH --time 00:10:00
#SBTACH --partition debug

#module load gcc/8 mvapich2 boost
module load gcc mvapich2 boost


ARCH=gcc_mvapich2

srun ../build_$ARCH/gmrm \
--bed-file test.bed \
--dim-file test.dim \
--phen-files test1.phen,test1_bis.phen,test1_nas.phen,test2.phen \
--group-index-file test.gri \
--group-mixture-file test.grm \
--shuffle-markers 1 \
--seed 171014 \
--iterations 10 \
--out-dir test1
