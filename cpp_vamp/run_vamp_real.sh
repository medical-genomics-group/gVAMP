#!/bin/bash

# SBATCH --reservation=robingrp_106
#SBATCH --ntasks 5
#SBATCH --cpus-per-task 6
#SBATCH --mem 0
# SBATCH --mem-per-cpu 5gb
#SBATCH --time 1-15:01:00
#SBATCH --output=emvamp_height_rho098_gamwL_PL_truesignalOn_testset_it10_CG10_01092022.log
# SBATCH --constraint=beta
#SBATCH --exclude=bjoern55
# SBATCH --constraint=avx512

# source /etc/profile.d/modules.sh

# module load gcc openmpi 

# module list 

# export OMP_NUM_THREADS=6

# mpic++ main_real.cpp vamp.cpp utilities.cpp phenotype.cpp -g -o main_real.exe

# mpic++ main_real.cpp vamp.cpp utilities.cpp phenotype.cpp -march=native -DMANVECT -Ofast -fopenmp -g -D_GLIBCXX_DEBUG -o  main_real.exe 

# time mpirun -np 5 ./main_real.exe 0

# mpic++ main.cpp vamp.cpp utilities.cpp phenotype.cpp -fopenmp -g -D_GLIBCXX_DEBUG -o  main.exe

# mpirun -np 6 ./main.exe


module load gcc openmpi boost

module list 

export OMP_NUM_THREADS=6

mpic++ main_real.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main_real.exe

time mpirun -np 5 ./main_real.exe --bed-file /nfs/scistore13/robingrp/human_data/geno/ldp08/ukb22828_UKB_EST_v3_ldp005_maf01.bed \
                                --phen-files /nfs/scistore13/robingrp/human_data/geno/ldp08/ukb_ht_noNA.phen \
                                --N 438361 \
                                --Mt 326165 \
                                --out-dir /nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/sig_estimates/ \
                                --out-name x1_hat_height_gamwL_PL_1_9_22 \
                                --iterations 10 \
                                --num-mix-comp 3 \
                                --CG-max-iter 10 \
                                --probs 0.70412,0.26945,0.02643 \
                                --vars 0,0.001251585388785e-5,0.606523422454662e-5 \
                                --rho 0.98
