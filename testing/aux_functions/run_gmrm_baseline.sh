#!/bin/bash 
#
#SBATCH --job-name=run_baseline_gmrm
#SBATCH --time=9-00:00:00
#SBATCH --mem 0
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 4
#SBATCH --constraint=delta


# printing out the details of the job
echo "bed_name = " $1 
echo "phen_name = " $2 
echo "number of iterations = " $3 

module purge
module load gcc openmpi
module list

bed_file_loc=/nfs/scistore13/robingrp/human_data/geno/ldp08
aux_loc=/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/aux

srun /nfs/scistore13/robingrp/adepope/PhD/gmrm/gmrm/build_gcc_openmpi/gmrm --bed-file ${bed_file_loc}/$1.bed \
                                                                           --dim-file ${aux_loc}/$1.dims \
                                                                           --phen-files ${bed_file_loc}/$2.phen \
                                                                           --group-index-file ${aux_loc}/$1.gris \
                                                                           --group-mixture-file ${aux_loc}/$1.grm \
                                                                           --shuffle-markers 1 \
                                                                           --seed 1510 \
                                                                           --iterations $3     
