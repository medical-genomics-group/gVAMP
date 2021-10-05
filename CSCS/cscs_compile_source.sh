#!/bin/sh

module swap PrgEnv-cray PrgEnv-intel
#module load daint-gpu
#module load intel

NAM=mpi_gibbs

SRC=${HOME}/BayesRRcmd/src

EXE=$SRC/$NAM

cd $SRC

B='-B'
B=''

make EXE=$NAM $B -f Makefile_CSCS || exit 1;


