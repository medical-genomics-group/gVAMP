#!/bin/bash
#SBATCH --partition=defaultp
# SBATCH --reservation=robingrp_89
#SBATCH --job-name=HT_AMP_analysis
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 80gb
#SBATCH --time 0-10:18:00
#SBATCH --output=/nfs/scistore13/robingrp/adepope/PhD/gene2/geneAMP/Height_uncorr.log
# #SBATCH --constraint=intel
# #SBATCH --constraint=avx2
#SBATCH --exclude delta203,delta197,delta198,delta199,delta200,delta201
#SBATCH --constraint=gamma


wd=/nfs/scistore13/robingrp/human_data/adepope_preprocessing
mpi_name=multiEpochs
mpi_dir=../testing
SEED=1234
program_location=bin/hydra_G

module purge

module load openmpi/4.1.1 eigen/3.3.7 boost/1.77.0

module list

 srun $program_location \
    --bfile $wd/ukb_imp_v3_UKB_EST_uncorrpeople \
    --pheno $wd/ukb_imp_v3_UKB_EST_uncorrpeople.fam \
    --failure $wd/ukb_imp_v3_UKB_EST_uncorrpeople.fam \
    --mpibayes bayesWMPI \
    --chain-length 5  \
    --burn-in 0 \
    --thin 5 \
    --mcmc-out-dir $mpi_dir \
    --mcmc-out-name $mpi_name \
    --seed $SEED \
    --shuf-mark 1 \
    --number-markers 111429 \
    --number-individuals 382390 \
    --quad_points 25 \
    --sync-rate 4 \
    --save 5

