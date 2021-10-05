#!/bin/bash

# To be run within a SLURM allocation (salloc)

module purge
module load intel intel-mpi intel-mkl boost eigen

source ./compile_code.sh $1


datadir=""; dataset=""; sparsedir=""; sparsebsn=""; phen=""


DS=4

if [ $DS == 0 ]; then 
    datadir=./test/data
    dataset=uk10k_chr1_1mb
    sparsedir=$datadir
    sparsebsn=${dataset}_uint
    phen=test
elif [ $DS == 1 ]; then
    datadir=/scratch/orliac/testM100K_N5K_missing
    dataset=memtest_M100K_N5K_missing0.01
    sparsedir=$datadir
    sparsebsn=${dataset}_uint
    phen=memtest_M100K_N5K_missing0.01
elif [ $DS == 2 ]; then
    datadir=/scratch/orliac/testN500K
    dataset=testN500K
    phen=$dataset
    sparsedir=$datadir
    sparsebsn=${dataset}_uint_test
elif [ $DS == 3 ]; then
    datadir=/scratch/orliac/test_B2S/
    dataset=ukb_imp_v3_UKB_EST_oct19_unrelated_chr22
    phen=ukb_imp_v3_UKB_EST_oct19_BMI_wNA_unrelated_148covadjusted_w35520NA
    phen=BMI_noNA.phen
    sparsedir=$datadir
    sparsebsn=${dataset}_test2
elif [ $DS == 4 ]; then
    datadir=/work/ext-unil-ctgg/etienne/test_bw
    dataset=t_M50K_N5K
    phen=$dataset
    sparsedir=$datadir
    sparsebsn=${dataset}
fi

echo datadir: $datadir

N=1
TPN=10
BPR=1

echo
echo Convert BED to SPARSE
echo

srun -N $N --ntasks-per-node=$TPN $EXE --bed-to-sparse --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --blocks-per-rank $BPR --sparse-dir $sparsedir --sparse-basename $sparsebsn
