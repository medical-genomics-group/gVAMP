#!/bin/bash

#SBATCH --ntasks 5
#SBATCH --cpus-per-task 6
#SBATCH --mem 0
# SBATCH --mem-per-cpu 5gb
#SBATCH --time 1-15:01:00
#SBATCH --exclude=bjoern55,bjoern37,bjoern38,bjoern40
#SBATCH --constraint=avx2


## how to run?
# phen_name=ukb_ht_noNA
# bed_name=ukb22828_UKB_EST_v3_ldp005_maf01
# sbatch --output=run_gmrm__${bed_name}__${phen_name}  --wait run_gmrm.sh $bed_name $phen_name 326165 438361

module load gcc openmpi boost

module list 

export OMP_NUM_THREADS=6

mpic++ main_real.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main_real.exe

time mpirun -np 5 ./main_real.exe --bed-file /nfs/scistore13/robingrp/human_data/geno/ldp08/$1.bed \
                                --phen-files /nfs/scistore13/robingrp/human_data/geno/ldp08/$2.phen \
                                --N $3 \
                                --Mt $4 \
                                --out-dir /nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/output/ \
                                --out-name x1_hat_$2_gamwL_CG_10_PL_rho_09_13_9_22 \
                                --iterations 30 \
                                --num-mix-comp 3 \
                                --CG-max-iter 10 \
                                --model linear \
                                --probs 0.70412,0.26945,0.02643 \
                                --vars 0,0.001251585388785e-5,0.606523422454662e-5 \
                                --rho 0.90
