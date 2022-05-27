#!/bin/bash
#
# Author : E. Orliac, DCSR, UNIL
# Date   : 2019/05/21
# Purpose: Compare solutions when reading from BED file or SPARSE representation files.
#          Results should be strictly equivalent.
#
# Warning: needs to be in an active slurm allocation, execution via srun!
#
# Warning: the block definition file (.blk) and job setup must match (wrt the number of tasks)
#


module purge
module load intel intel-mpi intel-mkl boost eigen zlib
module list

NAM=hydra

EXE=./src/$NAM

# COMPILATION
cd ./src
B='-B'
B=''
make $B EXE=$NAM -f Makefile || exit 1;
cd ..

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
    sparsedir=$datadir
    sparsebsn=${dataset}_uint
    NUMINDS=5000
    NUMSNPS=117148
elif [ $DS == 2 ]; then
    datadir=/scratch/orliac/testN500K
    dataset=testN500K
    phen=$dataset
    sparsedir=$datadir
    sparsebsn=${dataset}_uint_test
    NUMINDS=500000
    NUMSNPS=1270420
    NUMSNPS=10000
elif [ $DS == 3 ]; then
    sparsedir=/scratch/orliac/UKBgen/
    sparsebsn=epfl_test_data_sparse
    phen=epfl_test_data
    NUMINDS=457810
    NUMSNPS=8430446
    #NUMSNPS=34000
    S="0.00001,0.0001,0.001,0.01"
fi


echo 
echo "======================================"
echo "        RUNNING THE APPLICATION ON:   "
echo "datadir:   " $datadir
echo "dataset:   " $dataset
echo "sparse dir:" $sparsedir
echo "sparse bsn:" $sparsebsn
echo "S         :" $S
echo "======================================"
echo

CL=1
SEED=9
SR=0
SM=1
THIN=3
SAVE=3
TOCONV_T=$((($CL - 1) / $THIN))
echo TOCONV_T $TOCONV_T
N=1
TPN=1
BLK="--marker-blocks-file $datadir/${dataset}.blk"
BLKFILE=$sparsedir/block_def_3.blk
BLK="--marker-blocks-file $BLKFILE"
#BLK=""

#TPN=$(wc -l "$BLKFILE")
#echo TPN = $TPN

sol=RAM_simulation
#cmd="-N $N --ntasks-per-node=$TPN  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn $BLK --check-RAM"
#UKBgen@block3
#cmd="--ntasks=$TPN  --ntasks-per-node=10 $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn $BLK --check-RAM --check-RAM-tasks 500 --check-RAM-tasks-per-node 9"
#UKBgen@block2
BLKFILE=$sparsedir/block_def_2.blk
BLK="--marker-blocks-file $BLKFILE"
cmd="--ntasks=$TPN  --ntasks-per-node=10 $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn $BLK --check-RAM --check-RAM-tasks 500 --check-RAM-tasks-per-node 17"
echo $cmd
srun $cmd || exit 1
