#!/bin/bash
#
#SBATCH --job-name=averaging_beta
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 4G
#SBATCH --time=01:00:00

module purge
module load R
module list

# R CMD BATCH options "--args $2_$1" ../aux_functions/extracting_markers.R
Rscript ../aux_functions/extracting_markers.R $2 $4 $3 $5