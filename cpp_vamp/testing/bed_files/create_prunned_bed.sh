#!/bin/bash

#SBATCH --ntasks 10
# SBATCH --cpus-per-task 1
#SBATCH --mem 0
#SBATCH --time 2-00:15:00
#SBATCH --output=create_prunned_bed_050.log

# 0.99^2 = 0.9801
# 0.95^2 = 0.9025
# 0.9^2 = 0.81
# 0.8^2 = 0.64
# 0.7^2 = 0.49
# 0.6^2 = 0.36
# 0.5^2 = 0.25

input_bed_loc=/nfs/scistore13/robingrp/human_data/geno/chr/
out_bed_loc=/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/bed_files

module purge
module load plink/1.90
module list

# thr=099
# thr=095
# thr=090
# thr=080
# thr=049
#thr=036
thr=025

# --num_threads 10
plink --bfile ${input_bed_loc}/ukb22828_UKB_EST_v3_all --indep-pairwise 50 5 0.25 --threads 20 --out ${out_bed_loc}/prunned_${thr}

plink --bfile ${input_bed_loc}/ukb22828_UKB_EST_v3_all -extract prunned_${thr}.prune.in --make-bed --out ${out_bed_loc}/ukb22828_UKB_EST_v3_all_prunned_${thr}



