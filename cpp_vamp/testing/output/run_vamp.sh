#!/bin/bash

#SBATCH --ntasks 20
#SBATCH --cpus-per-task 10
#SBATCH --mem 0
# SBATCH --mem-per-cpu 5gb
#SBATCH --time 01-06:00:00
#SBATCH --exclude=bjoern55,bjoern37,bjoern38,bjoern40
#SBATCH --constraint=avx2
# SBATCH --constraint=delta


## how to run?
# phen_name=ukb_train_HT
# bed_name=ukb22828_UKB_EST_v3_all_prunned_080
# sbatch --output=run_vamp__${bed_name}__${phen_name}_31_09_2022_rho_03_CG_30 run_vamp.sh $bed_name $phen_name 458747 8430446 
# sbatch --output=run_vamp__${bed_name}__${phen_name}_07_10_2022_rho_03_CG_30 run_vamp.sh

# source /etc/profile.d/modules.sh

module load gcc openmpi boost

module list 

export OMP_NUM_THREADS=6

loc=nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP

mpic++ /$loc/main_real.cpp /$loc/vamp.cpp /$loc/utilities.cpp /$loc/data.cpp /$loc/options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  /$loc/main_real.exe

prthr=080
i1=ukb22828_UKB_EST_v3_all_prunned_${prthr}
# i1=ukb22828_c21_UKB_EST_v3
i2=ukb_train_HT
i3=458747
# i4=8430446 
# i4=115233
i4=2731356  # thr = 0.8
rho=30

# --bed-file /nfs/scistore13/robingrp/human_data/geno/chr/$i1.bed
time mpirun -np 20 /$loc/main_real.exe --bed-file /nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/bed_files/$i1.bed \
                                --phen-files /nfs/scistore13/robingrp/human_data/pheno/continuous/$i2.phen \
                                --N $i3 \
                                --Mt $i4 \
                                --out-dir /nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/output/vamp_signal_est/ \
                                --out-name x1_hat_${i2}_gamwL_CG_30_PL_rho_${rho}_07_10_22_prunned_${prthr} \
                                --iterations 20 \
                                --num-mix-comp 5 \
                                --CG-max-iter 30 \
                                --model linear \
                                --vars 0.00000,0.00001,0.01,0.1,1 \
                                --probs 0.990559568574484,0.009392809717180,0.000047274997321,0.000000078786398,0.000000267924616 \
                                --rho 0.${rho}

#                                 --probs 0.70412,0.26945,0.02643 \
#                                 --vars 0,0.001251585388785e-5,0.606523422454662e-5 \
#                                 --probs 0.85,0.025,0.025,0.05,0.03,0.01,0.01 \
#                                 --vars 0,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5 \