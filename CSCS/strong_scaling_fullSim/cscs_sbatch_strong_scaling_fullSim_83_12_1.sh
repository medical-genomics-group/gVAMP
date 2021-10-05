#!/bin/bash

#SBATCH --nodes=83
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --job-name=strong_scaling_fullSim_83_12_1
#SBATCH --output=strong_scaling_fullSim_83_12_1.out
#SBATCH --error=strong_scaling_fullSim_83_12_1.err
#SBATCH --mem=0
#
# SBATCH -p debug
#SBATCH -t 02:00:00
#SBATCH -C gpu

sol=${SLURM_JOB_NAME}

env | grep SLURM

module swap PrgEnv-cray PrgEnv-intel
module load daint-gpu
module load intel
module list

hostname
free -g
lscpu

NAM=mpi_gibbs

EXE=${HOME}/BayesRRcmd/src/$NAM
ls -l $EXE

if [ ! -f $EXE ]; then
    echo Fatal: binary $EXE not found!
    exit
fi


S="1.0,0.1"

DS=3

if [ $DS == 0 ]; then
    datadir=./test/data
    dataset=uk10k_chr1_1mb
    phen=test
    sparsedir=$datadir
    sparsebsn=${dataset}_uint
    NUMINDS=3642
    NUMSNPS=6717
elif [ $DS == 1 ]; then
    datadir=/scratch/orliac/testM100K_N5K_missing
    dataset=memtest_M100K_N5K_missing0.01
    phen=memtest_M100K_N5K_missing0.01
    phenNA=memtest_M100K_N5K_missing0.01_NA
    sparsedir=$datadir
    sparsebsn=${dataset}_uint
    NUMINDS=5000
    NUMSNPS=117148
    NUMSNPS=1000
elif [ $DS == 2 ]; then
    datadir=/scratch/snx3000/eorliac/testN500K
    dataset=testN500K
    phen=$dataset
    sparsedir=$datadir
    sparsebsn=${dataset}_uint
    NUMINDS=500000
    NUMSNPS=1270420
    NUMSNPS=400000
elif [ $DS == 3 ]; then
    sparsedir=/scratch/snx3000/eorliac/fullSim
    sparsebsn=sim_unlinked_freq_ukb_imp_rel_QC_I08_dup_MAF0002_acgt_sparse
    phen=sim_unlinked_freq_ukb_imp_rel_QC_I08_dup_MAF0002_acgt_Mc50K_h2_0.5
    NUMINDS=458783
    NUMSNPS=14794840
    #NUMSNPS=180000 # to fill one CSCS node => 83 nodes for the full dataset
    S="0.00001,0.0001,0.001,0.01"
fi


echo 
echo "======================================"
echo "        RUNNING THE APPLICATION ON:   "
echo "datadir   :" $datadir
echo "dataset   :" $dataset
echo "sparse dir:" $sparsedir
echo "sparse bsn:" $sparsebsn
echo "S         :" $S
echo "======================================"
echo

CL=10
SEED=1222
SR=5
SM=1
SAVE=10000

outdir=`pwd`

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export KMP_AFFINITY=verbose

cmd="srun $EXE --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin 1  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --number-markers $NUMSNPS --number-individuals $NUMINDS --sparse-dir $sparsedir --sparse-basename $sparsebsn --save $SAVE"
echo ----------------------------------------------------------------------------------
echo $cmd
echo ----------------------------------------------------------------------------------
$cmd
