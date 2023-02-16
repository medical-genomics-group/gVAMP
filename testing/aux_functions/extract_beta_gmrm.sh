#!/bin/bash
#
#SBATCH --job-name=extr_beta
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
# SBATCH --mem 12G
#SBATCH --time=05:00:00

out_loc=/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/output

/nfs/scistore13/robingrp/adepope/PhD/gmrm/gmrm/example/extract_non_zero_betaAll ${out_loc}/$2.bet $4 $3 > ${out_loc}/$2.betlong

awk '{$1+=0}1' ${out_loc}/$2.betlong > ${out_loc}/$2.csv

rm ${out_loc}/$2.betlong
