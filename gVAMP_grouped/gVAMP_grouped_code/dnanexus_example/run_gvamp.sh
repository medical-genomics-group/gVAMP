#!/bin/bash

# Set directories
DATA_DIR=/home/dnanexus/mnt
OUT_DIR=/home/dnanexus/mnt/output
EXE_DIR=/home/dnanexus/gVAMP

# Set parallelization according to available CPU resources
NUM_MPI_WORKERS=...
NUM_OMP_THREADS=...

export OMP_NUM_THREADS=${NUM_OMP_THREADS}

mpirun -np ${NUM_MPI_WORKERS} --allow-run-as-root ${EXE_DIR}/main_real.exe \
        --model linear \
        --run-mode infere \
        --bim-file ${DATA_DIR}/... \
        --bed-file ${DATA_DIR}/... \
        --phen-files ${DATA_DIR}/... \
        --N ... \
        --Mt ... \
        --out-dir ${OUT_DIR}/ \
        --out-name gVAMP_example \
        --iterations 30 \
        --num-mix-comp 23 \
	--probs 9.950000000000e-01,2.500000596096e-03,1.250000298048e-03,6.250001490239e-04,3.125000745120e-04,1.562500372560e-04,7.812501862799e-05,3.906250931400e-05,1.953125465700e-05,9.765627328499e-06,4.882813664250e-06,2.441406832125e-06,1.220703207729e-06,6.103518121979e-07,3.051758644323e-07,1.525879113828e-07,7.629397652474e-08,3.814697992903e-08,1.907349204785e-08,9.536743940592e-09,4.768371970296e-09,2.384186401815e-09,1.192093200907e-09 \
	--vars 0,0.0000001,0.0000002238,0.0000005,0.00000112,0.00000251,0.00000561,0.000012565,0.00002812,0.0000629,0.0001408448,0.0003152106,0.0007054413,0.001578778,0.001578778,0.003533305,0.007907536,0.01769706,0.03960603,0.0886383,0.1983725,0.4439577,0.9935773 \
	--use-lmmse-damp 0 \
	--store-pvals 1 \
	--learn-vars 1 \
	--rho 0.05 \
> ${OUT_DIR}/gVAMP_example.log
