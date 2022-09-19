#!/bin/bash 
#
#SBATCH --job-name=run_baseline_gmrm
#SBATCH --time=9-00:00:00
#SBATCH --mem 0
#SBATCH --ntasks 4
#SBATCH --cpus-per-task 4
#SBATCH --constraint=delta

## how to run?
# phen_name=ukb_ht_noNA
# bed_name=ukb22828_UKB_EST_v3_ldp005_maf01
# sbatch --output=run_gmrm__${bed_name}__${phen_name}  --wait run_gmrm.sh $bed_name $phen_name 10 5 326165

# printing out the details of the job
echo "bed_name = " $1 
echo "phen_name = " $2

#  1. running gmrm 
 sbatch --output=run_gmrm_baseline__$1__$2 --wait ../aux_functions/run_gmrm_baseline.sh $1 $2 $3
echo "finished with gmrm"

#  2. extracting betas
 sbatch --output=extract_betas__$1__$2 --wait ../aux_functions/extract_beta_gmrm.sh $1 $2 $3 $4
echo "finished with betas extraction" 

#  3. averging effect sizes throughout itrations 
sbatch --output=averaging_effects__$1__$2 --wait ../aux_functions/averaging_effect_over_iter.sh $1 $2 $3 $4 $5
echo "finished with averaging effect sizes"
