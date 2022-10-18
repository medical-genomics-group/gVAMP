#!/bin/bash

#SBATCH --ntasks 10
#SBATCH --cpus-per-task 5
#SBATCH --mem 0
# SBATCH --mem-per-cpu 5gb
#SBATCH --time 01-06:00:00
#SBATCH --exclude=bjoern55,bjoern37,bjoern38,bjoern40
#SBATCH --constraint=avx2
# SBATCH --constraint=delta


## how to run?
# phen_name=ukb_train_HT
# bed_name=ukb22828_UKB_EST_v3_all_prunned_060_maf_005
# sbatch --output=run_vamp__${bed_name}__${phen_name}_31_09_2022_rho_01_CG_30 run_vamp.sh $bed_name $phen_name 458747 8430446 
# sbatch --output=run_vamp__${bed_name}__${phen_name}_18_10_2022_rho_005_CG_10_thr_060 run_vamp.sh

# source /etc/profile.d/modules.sh

module load gcc openmpi boost

module list 

export OMP_NUM_THREADS=5

# loc=nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP
loc=nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPBirtyhday/gVAMP/cpp_vamp

mpic++ /$loc/main_real.cpp /$loc/vamp.cpp /$loc/utilities.cpp /$loc/data.cpp /$loc/options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  /$loc/main_real.exe

prthr=036
i1=ukb22828_UKB_EST_v3_all_prunned_${prthr}_maf_005
# i1=ukb22828_c21_UKB_EST_v3
i2=ukb_train_HT
i2T=ukb_test_HT
i3=458747
# i4=8430446 
# i4=115233
# i4=2731356  # thr = 0.8
# i4=4970136   #maf=0.05
# i4=2479074 # the = 0.7
i4=528804
i5=ukb22828_UKB_EST_v3_all_prunned_${prthr}_maf_005_test
rho=05
CG=10

# --bed-file /nfs/scistore13/robingrp/human_data/geno/chr/$i1.bed
time mpirun -np 10 /$loc/main_real.exe --bed-file /nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/bed_files/$i1.bed \
                                --phen-files /nfs/scistore13/robingrp/human_data/pheno/continuous/$i2.phen \
                                --N $i3 \
                                --Mt $i4 \
                                --out-dir /nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/output/vamp_signal_est/ \
                                --out-name x1_hat_${i2}_gamwL_CG_${CG}_PL_rho_0${rho}_18_10_22_prunned_${prthr}_maf_005 \
                                --iterations 10 \
                                --num-mix-comp 4 \
                                --CG-max-iter ${CG} \
                                --model linear \
                                --vars 0.00000,1e-7,1e-4,1e-1\
                                --probs 0.92,0.04,0.02,0.02 \
                                --rho 0.${rho} \
                                --run-mode both \
                                --phen-files-test /nfs/scistore13/robingrp/human_data/pheno/continuous/$i2T.phen \
                                --bed-file-test	/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/bed_files/$i5.bed \
                                --N-test 15000 \
                                --Mt-test $i4 \

#                                 --probs 0.70412,0.26945,0.02643 \
#                                 --vars 0,0.001251585388785e-5,0.606523422454662e-5 \
#                                 --probs 0.85,0.025,0.025,0.05,0.03,0.01,0.01 \
#                                 --vars 0,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5 \