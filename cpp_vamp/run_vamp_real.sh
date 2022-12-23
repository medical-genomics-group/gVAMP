#!/bin/bash

# SBATCH --reservation=robingrp_106
#SBATCH --ntasks 18
#SBATCH --cpus-per-task 4
#SBATCH --mem 0
# SBATCH --mem-per-cpu 5gb
#SBATCH --time 1-15:01:00
#SBATCH --output=vamp_height_fixedvars_threads18_rho08_gamwL_PL_truesignalOn_testset_it30_CG50_warm_uncorrnPe_both_13122022.log
# SBATCH --exclude=bjoern55,bjoern37,bjoern38,bjoern40
# SBATCH --constraint=avx2

# source /etc/profile.d/modules.sh

# module load gcc openmpi 

# module list 

# export OMP_NUM_THREADS=6

# mpic++ main_real.cpp vamp.cpp utilities.cpp phenotype.cpp -g -o main_real.exe

# mpic++ main_real.cpp vamp.cpp utilities.cpp phenotype.cpp -march=native -DMANVECT -Ofast -fopenmp -g -D_GLIBCXX_DEBUG -o  main_real.exe 

# time mpirun -np 5 ./main_real.exe 0

# mpic++ main.cpp vamp.cpp utilities.cpp phenotype.cpp -fopenmp -g -D_GLIBCXX_DEBUG -o  main.exe

# mpirun -np 6 ./main.exe


module purge

ml gcc openmpi boost

module list 

export OMP_NUM_THREADS=4

# -DMANVECT
# --probs 0.70412,0.26945,0.02643 \
# --vars 0,0.001251585388785e-5,0.606523422454662e-5 \

mpic++ main_real.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main_real.exe

# /nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/bed_files/ukb22828_UKB_EST_v3_ldp08_fd_maf_thr050_2.bed
# /nfs/scistore13/robingrp/human_data/geno/ldp08/ukb22828_UKB_EST_v3_ldp005_maf01.bed
time mpirun -np 18 ./main_real.exe --bed-file  /nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/bed_files/ukb22828_UKB_EST_v3_ldp08_fd_maf_thr050_3.bed \
                                --phen-files /nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/bed_files/ukb_ht_fd_train.phen \
                                --N 419155 \
                                --Mt 521208 \
                                --out-dir /nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/sig_estimates/ \
                                --out-name x1_hat_height_gamwL_CG_10_PFL_rho_08_13_12_22_both_threads18 \
                                --iterations 12 \
                                --num-mix-comp 3 \
                                --CG-max-iter 50 \
                                --probs 0.7,0.1,0.1,0.1 \
                                --vars 0,1e-6,1e-4,1e-2 \
                                --model linear \
                                --run-mode both \
                                --rho 0.80
