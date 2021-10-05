#!/bin/bash

#SBATCH --job-name=annotmafLD
#SBATCH --account=ext-unil-ctgg
#SBATCH --mem=0
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=4
#SBATCH --time 3-00:00:00
#SBATCH --partition=parallel
#SBATCH --output /work/ext-unil-ctgg/tdaniel/groups72_mix4_unrelated_cpus4_tasks8_nodes10_mpisync10_height_4_unrelated_148covadjusted_w35520NA.log
#SBATCH --error /work/ext-unil-ctgg/tdaniel/groups72_mix4_unrelated_cpus4_tasks8_nodes10_mpisync10_height_4_unrelated_148covadjusted_w35520NA.err

module purge
module load intel intel-mpi eigen boost
module list

mkdir -p /scratch/tdaniel/ukb_height_72groups/

export OMP_NUM_THREADS=4

env | grep SLURM

echo SEED = 214096108

start_time="$(date -u +%s)"

cmd=" /work/ext-unil-ctgg/common_software/hydra/src/hydra  --number-individuals 382466 --number-markers 8430446 --mpibayes bayesMPI --pheno /work/ext-unil-ctgg/marion/ukb_imp_v3_UKB_EST_oct19_pheno_w35520NA/ukb_imp_v3_UKB_EST_oct19_height_wNA_unrelated_148covadjusted_w35520NA.phen --chain-length 10005 --thin 5 --save 1000 --mcmc-out-dir /scratch/tdaniel/ukb_height_72groups --mcmc-out-name groups72_mix4_unrelated_cpus4_tasks8_nodes10_mpisync10_height_4_unrelated_148covadjusted_w35520NA --seed 214096108 --shuf-mark 1 --sync-rate 10 --groupIndexFile /work/ext-unil-ctgg/robinson/annot/ukb_imp_v3_UKB_EST_oct19_unrelated_annot12_maf3_ld2_bins.group --groupMixtureFile /work/ext-unil-ctgg/robinson/annot/ukb_imp_v3_UKB_EST_oct19_unrelated_annot12_maf3_ld2_bins_cpn_4.cva --bfile /work/ext-unil-ctgg/thanasis/ukb_imp_v3_UKB_EST_oct19/ukb_imp_v3_UKB_EST_oct19_unrelated --sparse-dir /scratch/athanasi/ukb_imp_v3_UKB_EST_oct19_unrelated/ --sparse-basename ukb_imp_v3_UKB_EST_oct19_unrelated --threshold-fnz 0.060 --sparse-sync "

echo
echo $cmd
echo 

srun $cmd

end_time="$(date -u +%s)"
elapsed="$(($end_time-$start_time))"
echo "Total time in sec: $elapsed"

