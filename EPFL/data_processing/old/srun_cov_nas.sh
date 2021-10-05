#!/bin/bash
#
# $1 : pass -B from command line to force recompilation e.g.
#

source ./compile_code.sh $1


S="1.0,0.1"

DS=4

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
    NUMSNPS=47005
elif [ $DS == 3 ]; then
    sparsedir=/scratch/orliac/UKBgen/
    sparsebsn=epfl_test_data_sparse
    phen=epfl_test_data
    NUMINDS=457810
    NUMSNPS=8430446
    NUMSNPS=400000
    S="0.00001,0.0001,0.001,0.01"
elif [ $DS == 4 ]; then
    datadir=/scratch/orliac/sim1_M100K_N10K
    dataset=sim1_M100K_N10K
    sparsedir=$datadir
    sparsebsn=${dataset}
    NUMINDS=10000
    NUMSNPS=100000
    NUMSNPS=100000
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

CL=5
SEED=1111
SR=0
SM=1

# If you change those, do not expect compatibility
N=1
TPN=8

sol=test_cov_nas

# Convert BED to SPARSE
if [ 1 == 0 ]; then
    BPR=1
    cmd="srun -N $N --ntasks-per-node=$TPN  $EXE  --bed-to-sparse  --bfile $datadir/$dataset  --blocks-per-rank $BPR --sparse-dir $sparsedir --sparse-basename $sparsebsn --mcmc-out-dir $outdir  --mcmc-out-name ${sol}_Bed2Sparse"    
    echo ----------------------------------------------------------------------------------
    echo $cmd
    echo ----------------------------------------------------------------------------------
    $cmd || exit 1    
fi

# Case NAs in both .phen and .cov
# -------------------------------
phen=sim1_M100K_N10K_withNA.phen;  cov=sim1_M100K_N10K_cov_scaled_withfamcol_withNA.cov
COV=""
sol=test_nocov_nas
cmd="srun -N $N  --ntasks-per-node=$TPN  $NUMACTL $EXE  --mpibayes bayesMPI  --pheno $sparsedir/${phen} $COV --chain-length $CL  --thin 1  --mcmc-out-dir $outdir  --mcmc-out-name $sol  --seed $SEED  --shuf-mark $SM  --mpi-sync-rate $SR  --S $S  --number-markers $NUMSNPS  --number-individuals $NUMINDS  --sparse-dir $sparsedir  --sparse-basename $sparsebsn"
echo ----------------------------------------------------------------------------------
echo $cmd
echo ----------------------------------------------------------------------------------
$cmd || exit 1

COV="--covariates $sparsedir/${cov}"
sol=test_cov_nas
cmd="srun -N $N  --ntasks-per-node=$TPN  $NUMACTL $EXE  --mpibayes bayesMPI  --pheno $sparsedir/${phen} $COV --chain-length $CL  --thin 1  --mcmc-out-dir $outdir  --mcmc-out-name $sol  --seed $SEED  --shuf-mark $SM  --mpi-sync-rate $SR  --S $S  --number-markers $NUMSNPS  --number-individuals $NUMINDS  --sparse-dir $sparsedir  --sparse-basename $sparsebsn"
echo ----------------------------------------------------------------------------------
echo $cmd
echo ----------------------------------------------------------------------------------
$cmd || exit 1
