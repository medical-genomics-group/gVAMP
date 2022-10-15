#!/bin/bash

#SBATCH --ntasks 10
# SBATCH --cpus-per-task 1
#SBATCH --mem 0
#SBATCH --time 2-00:15:00
#SBATCH --output=create_prunned_bed_010.log

# 0.99^2 = 0.9801
# 0.95^2 = 0.9025
# 0.9^2 = 0.81
# 0.8^2 = 0.64
# 0.7^2 = 0.49
# 0.6^2 = 0.36
# 0.5^2 = 0.25
# 0.4^2 = 0.16
# 0.3^2 = 0.09
# 0.2^2 = 0.04
# 0.1^2 = 0.01

input_bed_loc=/nfs/scistore13/robingrp/human_data/geno/chr
out_bed_loc=/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/bed_files

module purge
module load plink/1.90
module list

# thr=099
# thr=095
# thr=090
# thr=080
# thr=070
# thr=060
# thr=050
# thr=040
# thr=030
# thr=020
thr=010

# --num_threads 10
# plink --bfile ${input_bed_loc}/ukb22828_UKB_EST_v3_all --indep-pairwise 50 5 0.25 --threads 20 --out ${out_bed_loc}/prunned_${thr}

plink --bfile ${out_bed_loc}/ukb22828_UKB_EST_v3_all_prunned_080 --indep-pairwise 50 5 0.01 --threads 10 --out ${out_bed_loc}/prunned_${thr}

# plink --bfile ${input_bed_loc}/ukb22828_UKB_EST_v3_all -extract prunned_${thr}.prune.in --threads 10 --make-bed --out ${out_bed_loc}/ukb22828_UKB_EST_v3_all_prunned_${thr}

plink --bfile ${out_bed_loc}/ukb22828_UKB_EST_v3_all_prunned_080 -extract prunned_${thr}.prune.in --threads 10 --make-bed --out ${out_bed_loc}/ukb22828_UKB_EST_v3_all_prunned_${thr}



