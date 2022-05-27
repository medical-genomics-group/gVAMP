#!/bin/bash

#SBATCH --job-name="weak_scaling_1_1"
#SBATCH --output=weak_scaling_1_1.%j.out
#SBATCH --error=weak_scaling_1_1.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -p debug
#SBATCH -t 00:05:00
#SBATCH -C gpu


module swap PrgEnv-cray PrgEnv-intel
module load daint-gpu
module load intel
module list

hostname
free -g

NAM=mpi_gibbs

EXE=${HOME}/BayesRRcmd/src/$NAM

if [ ! -f $EXE ]; then
    echo Fatal: binary $EXE not found!
    exit
fi


S="1.0,0.1"

DS=2

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
    sparsedir=/scratch/orliac/UKBgen/
    sparsebsn=epfl_test_data_sparse
    phen=epfl_test_data
    NUMINDS=457810
    NUMSNPS=8430446
    NUMSNPS=50000
    S="0.00001,0.0001,0.001,0.01"
fi

outdir=/scratch/snx3000/eorliac/test0

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

CL=3
SEED=1222
SR=0
SM=1
SAVE=10000

sol=mpi1tsparse
cmd="srun $EXE --mpibayes bayesMPI --pheno $datadir/${phen}.phen --chain-length $CL --thin 1  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --number-markers $NUMSNPS --number-individuals $NUMINDS --sparse-dir $sparsedir --sparse-basename $sparsebsn --save $SAVE"
echo ----------------------------------------------------------------------------------
echo $cmd
echo ----------------------------------------------------------------------------------
$cmd
