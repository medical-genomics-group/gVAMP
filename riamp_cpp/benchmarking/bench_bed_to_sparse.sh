#!/bin/bash

set -e
#set -x

#source ../compile_with_intel.sh -B
source ../compile_with_intel.sh

echo HYDRA_ROOT = $HYDRA_ROOT
echo HYDRA_EXE  = $HYDRA_EXE

BENCH_DIR=/work/ext-unil-ctgg/etienne/data_bench/
[ -d $BENCH_DIR ] || (echo "fatal: bench directory not found! $BENCH_DIR" && exit)
echo BENCH_DIR = $BENCH_DIR

OUT_DIR=/scratch/orliac/bench_hydra

NTHREADS=8

export OMP_NUM_THREADS=$NTHREADS

srun -p build -n 4 -t 00:05:00 --mem 20GB --cpus-per-task $NTHREADS $HYDRA_ROOT/bin/$HYDRA_EXE \
    --bfile $BENCH_DIR/test \
    --pheno $BENCH_DIR/test.phen \
    --mcmc-out-dir $OUT_DIR \
    --mcmc-out-name bench_hydra_epfl_intel \
    --bed-to-sparse

