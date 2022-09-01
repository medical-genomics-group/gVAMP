#!/bin/bash

# SBATCH --reservation=robingrp_106
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 6
#SBATCH --mem 0
# SBATCH --mem-per-cpu 5gb
#SBATCH --time 1-00:15:00
#SBATCH --output=main_benchmark_out.log
#SBATCH --constraint=gamma

module load gcc openmpi boost

module list 

export OMP_NUM_THREADS=6
export OMP_PROC_BIND=close
export OMP_PLACES=sockets
# setenv OMP_PROC_BIND close
# setenv OMP_PLACES cores

mpic++ main_benchmark.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main_benchmark.exe

time mpirun -np 4 ./main_benchmark.exe --bed-file /nfs/scistore13/robingrp/human_data/geno/ldp08/ukb22828_UKB_EST_v3_ldp005_maf01.bed \
                                       --phen-files /nfs/scistore13/robingrp/human_data/geno/ldp08/ukb_ht_noNA.phen \
                                       --N 438361 \
                                       --Mt 326165 \
                                       --out-dir /nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/sig_estimates/ \
                                       --out-name x1_hat_height_benchmark \
                                       --iterations 5 \
                                       --num-mix-comp 3 \
                                       --probs 0.70412,0.26945,0.02643 \
                                       --vars 0,0.001251585388785e-5,0.606523422454662e-5 \
                                       --rho 0.98

#module load likwid 

#likwid-topology

#time likwid-mpirun -np 4 -pin C0:0,1_C1:0,1 -g FLOPS_DP MEM ./main_benchmark.exe --bed-file /nfs/scistore13/robingrp/human_data/geno/ldp08/ukb22828_UKB_EST_v3_ldp005_maf01.bed \
#                                       --phen-files /nfs/scistore13/robingrp/human_data/geno/ldp08/ukb_ht_noNA.phen \
#                                       --N 438361 \
#                                       --Mt 326165 \
#                                       --out-dir /nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/sig_estimates/ \
#                                       --out-name x1_hat_height_benchmark \
#                                       --iterations 1 \
#                                       --num-mix-comp 3 \
#                                       --probs 0.70412,0.26945,0.02643 \
#                                       --vars 0,0.001251585388785e-5,0.606523422454662e-5 \
#                                       --rho 0.98
                                
                                       
