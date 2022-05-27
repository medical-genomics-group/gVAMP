#!/bin/bash
#
# $1 : pass -B from command line to force recompilation e.g.
#

source ./compile_code_G.sh $1

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
    datadir=/scratch/orliac/testN500K
    dataset=testN500K
    phen=$dataset
    sparsedir=$datadir
    sparsebsn=${dataset}_uint
    NUMINDS=500000
    NUMSNPS=1270420
    NUMSNPS=20000
elif [ $DS == 3 ]; then
    sparsedir=/scratch/orliac/UKBgen/
    sparsebsn=epfl_test_data_sparse
    phen=epfl_test_data
    NUMINDS=457810
    NUMSNPS=8430446
    NUMSNPS=500
    S="0.00001,0.0001,0.001,0.01"
fi

outdir=/home/orliac/DCSR/CTGG/BayesRRcmd/output_tests/

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
SM=0

# If you change those, do not expect compatibility
N=1
TPN=1

echo 
echo
echo "@@@ Official (sequential) solution (reading from BED file) @@@"
echo
cmd="srun -N $N --ntasks-per-node=$TPN $EXE --bayes bayesMmap --bfile $datadir/$dataset --pheno $datadir/${phen}.phen   --chain-length $CL --burn-in 0 --thin 1 --mcmc-out-dir $outdir --mcmc-out-name refG --shuf-mark $SM --seed $SEED --S $S --number-markers $NUMSNPS"
#cmd="srun -N $N --ntasks-per-node=$TPN $EXE --bayes bayesMmap --bfile $datadir/$dataset --pheno $datadir/${phenNA}.phen --chain-length $CL --burn-in 0 --thin 1 --mcmc-out refNA --shuf-mark $SM --seed $SEED --S $S --number-markers $NUMSNPS"
#--covariates $datadir/scaled_covariates.csv
echo ----------------------------------------------------------------------------------
echo $cmd
echo ----------------------------------------------------------------------------------
$cmd || exit 1

echo
echo
echo "@@@ MPI 1-task solution reading from  BED file @@@"
echo
sol=mpi1tbedG
cmd="srun -N $N --ntasks-per-node=$TPN $EXE --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin 1  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --read-from-bed-file --number-markers $NUMSNPS --number-individuals $NUMINDS"
echo ----------------------------------------------------------------------------------
echo $cmd
echo ----------------------------------------------------------------------------------
$cmd || exit 1


echo
echo
echo "@@@ MPI 1-task solution reading from SPARSE files @@@"
echo
sol=mpi1tsparseG
cmd="srun -N $N --ntasks-per-node=$TPN $EXE --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin 1  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --number-markers $NUMSNPS --number-individuals $NUMINDS --sparse-dir $sparsedir --sparse-basename $sparsebsn"
echo ----------------------------------------------------------------------------------
echo $cmd
echo ----------------------------------------------------------------------------------
$cmd || exit 1
#--marker-blocks-file $datadir/${dataset}.blk_1 
#--covariates $datadir/scaled_covariates.csv

