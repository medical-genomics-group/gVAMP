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
module load intel intel-mpi intel-mkl boost eigen
#module load valgrind
module list

# COMPILATION
# -----------
NAM=hydra
EXE=./src/$NAM
cd ./src
B='-B'
B=''
make $B EXE=$NAM -f Makefile || exit 1;
cd ..
if [ ! -f $EXE ]; then
    echo Fatal: binary $EXE not found!
    exit
fi

if [ 1  == 1 ]; then 
    icc beta_converter.c       -o beta_converter
    icc epsilon_converter.c    -o epsilon_converter
    icc components_converter.c -o components_converter
fi    

COV=""


# DATASETS DEFINITION
# -------------------
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
    #phen=memtest_M100K_N5K
    COV="--covariates ${datadir}/memtest_M100K_N5K.cov"
    sparsedir=$datadir
    sparsebsn=${dataset}_uint
    NUMINDS=5000
    NUMSNPS=117148
    NUMSNPS=10000
elif [ $DS == 2 ]; then
    datadir=/scratch/orliac/testN500K
    dataset=testN500K
    phen=$dataset
    sparsedir=$datadir
    sparsebsn=${dataset}_uint_test
    NUMINDS=500000
    NUMSNPS=1270420
    NUMSNPS=1000
elif [ $DS == 3 ]; then
    sparsedir=/scratch/orliac/UKBgen/
    sparsebsn=epfl_test_data_sparse
    phen=epfl_test_data
    NUMINDS=457810
    NUMSNPS=8430446
    NUMSNPS=3600
    S="0.00001,0.0001,0.001,0.01"
elif [ $DS == 4 ]; then
    E=nm
    datadir=/scratch/orliac/testNA
    dataset=test_$E
    sparsedir=$datadir
    sparsebsn=${dataset}_sparse
    phen=test_$E
    phen=test_m;#1miss
    NUMINDS=20000
    NUMSNPS=50000
    S="0.00001,0.0001,0.001,0.01"
    outdir=$datadir/results
    sol=$outdir/test_${E}7
    sol2=$outdir/test_${E}7_sparse
elif [ $DS == 5 ]; then
    datadir=/scratch/orliac/testing_missing
    dataset=testing_missing
    sparsedir=$datadir
    sparsebsn=${dataset}_sparse
    phen=testing
    phen_na=testing_missing
    NUMINDS=10000
    NUMSNPS=10000
    NUMSNPS=1000
    S="0.00001,0.0001,0.001,0.01"
    outdir=$datadir/results/test_restart
    outnam=fail
    sol=$outdir/$outnam
    sol2=$outdir/${outnam}_sparse
fi

if [ ! -d $outdir ]; then
    mkdir -p -v $outdir || exit 1;
fi

echo 
echo "======================================"
echo "        RUNNING THE APPLICATION ON:   "
echo "datadir:   " $datadir
echo "dataset:   " $dataset
echo "sparse dir:" $sparsedir
echo "sparse bsn:" $sparsebsn
echo "S         :" $S
echo "output dir:" $outdir
echo "======================================"
echo

CLR=17; CLF=7;
if [ "$CLF" -ge "$CLR" ]; then
    echo "FATAL: failing iteration CLF(=$CLF) must be strictly lower than full chain length CLR(=$CLR)!"
    exit 1;
fi


SEED=13
SR=0
SM=1
THIN=3
SAVE=6

if [ "$SAVE" -ge "$CLF" ]; then
    echo "FATAL: failing iteration CLF(=$CLF) must be strictly greater that SAVE(=$SAVE) rate!"
    exit 1;
fi


TOCONV_T=$((($CLR - 1) / $THIN))
echo TOCONV_T $TOCONV_T
N=1
TPN=11
BPR=5


# Select what to run ----------------------
bed_to_sparse=0;  run_bed=1;  run_sparse=0;
# -----------------------------------------


# Convert bed to sparse
if [ $bed_to_sparse == 1 ]; then
    cmd="srun -N $N --ntasks-per-node=$TPN $EXE --bed-to-sparse --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --blocks-per-rank $BPR --sparse-dir $sparsedir --sparse-basename $sparsebsn"
    echo $cmd
    $cmd
fi

BLK="--marker-blocks-file $datadir/${dataset}.blk"
BLK=""
VALGRIND="valgrind -v --leak-check=yes";
VALGRIND="valgrind -v --tool=exp-sgcheck";
VALGRIND=""

if [ $run_bed == 1 ]; then
    echo; echo
    echo "@@@ Solution reading from  BED file @@@"
    echo

    outnam=ref; sol=$outdir/$outnam;  CL=$CLR;
    cmd="srun -N $N --ntasks-per-node=$TPN  $VALGRIND  $EXE           --number-individuals $NUMINDS  --number-markers $NUMSNPS --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out-dir $datadir/results --mcmc-out-name $outnam  --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --read-from-bed-file $COV $BLK"
    printf '=%.0s' {1..100}; echo
    echo $cmd
    printf '=%.0s' {1..100}; echo
    $cmd || exit 1

    outnam=fail; sol=$outdir/$outnam; CL=$CLF;
    cmd="srun -N $N --ntasks-per-node=$TPN  $VALGRIND  $EXE           --number-individuals $NUMINDS  --number-markers $NUMSNPS --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out-dir $datadir/results --mcmc-out-name $outnam  --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --read-from-bed-file $COV $BLK"
    printf '=%.0s' {1..100}; echo
    echo $cmd
    printf '=%.0s' {1..100}; echo
    $cmd >/dev/null || exit 1
    #$cmd || exit 1


    outnam=fail; sol=$outdir/$outnam; CL=$CLR;
    cmd="srun -N $N --ntasks-per-node=$TPN  $VALGRIND  $EXE --restart --number-individuals $NUMINDS  --number-markers $NUMSNPS --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out-dir $datadir/results --mcmc-out-name $outnam --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --read-from-bed-file $COV $BLK"
    printf '=%.0s' {1..100}; echo
    echo $cmd
    printf '=%.0s' {1..100}; echo
    $cmd || exit 1

    #./beta_converter       $sol".bet" $TOCONV_T > $sol".bet.txt"
    #./epsilon_converter    $sol".eps"           > $sol".eps.txt"
    #./components_converter $sol".cpn" $TOCONV_T > $sol".cpn.txt"
fi


if [ $run_sparse == 1 ]; then
    echo; echo
    echo "@@@ Solution reading from SPARSE files @@@"
    echo
    cmd="srun -N $N --ntasks-per-node=$TPN  $VALGRIND  $EXE  --number-individuals $NUMINDS  --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CLR --thin $THIN --save $SAVE --mcmc-out $sol2 --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn $COV $BLK"
    printf '=%.0s' {1..100}; echo
    echo $cmd
    printf '=%.0s' {1..100}; echo
    $cmd || exit 1
    ./beta_converter       $sol2".bet" $TOCONV_T > $sol2".bet.txt"
    ./epsilon_converter    $sol2".eps"           > $sol2".eps.txt"
    ./components_converter $sol2".cpn" $TOCONV_T > $sol2".cpn.txt"
fi
