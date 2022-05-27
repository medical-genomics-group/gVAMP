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
    NUMSNPS=2000
elif [ $DS == 3 ]; then
    sparsedir=/scratch/orliac/UKBgen/
    sparsebsn=epfl_test_data_sparse
    phen=epfl_test_data
    NUMINDS=457810
    NUMSNPS=8430446
    NUMSNPS=500
    S="0.00001,0.0001,0.001,0.01"
elif [ $DS == 4 ]; then
    datadir=/scratch/orliac/TESTdataset_groups_mpi
    dataset=sim1_M100K_N10K
    phen=sim1_M100K_N10K_h2_0.5_Mc_1000
    grp=sim1_M100K_N10K_h2_0.5_Mc_1000.groups #.1
    mix=sim1_M100K_N10K_h2_0.5_Mc_1000.S #.1
    NUMINDS=10000
    NUMSNPS=100000
    NUMSNPS=100000
fi


outdir=./output_tests/

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

CL=22
SEED=1222
SR=0
SM=1

# If you change those, do not expect compatibility
N=1
TPN=17

echo 
echo
echo "@@@ Official (sequential) solution (reading from BED file) @@@"
echo
cmd="srun -N $N --ntasks-per-node=$TPN $EXE --bayes bayesMmap --bfile $datadir/$dataset --pheno $datadir/${phen}.phen   --chain-length $CL --burn-in 0 --thin 1 --mcmc-out-dir $outdir --mcmc-out-name ref --shuf-mark $SM --seed $SEED --S $S --number-markers $NUMSNPS"
echo ----------------------------------------------------------------------------------
#echo $cmd
echo ----------------------------------------------------------------------------------
#$cmd || exit 1

echo
echo
echo "@@@ MPI 1-task solution reading from  BED file @@@"
echo
sol=ref; CL=14;
cmd="srun -N $N --ntasks-per-node=$TPN $EXE --mpiBayesGroups --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin 1  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --read-from-bed-file --number-markers $NUMSNPS --number-individuals $NUMINDS --groupIndexFile $datadir/$grp --groupMixtureFile $datadir/$mix"
echo ----------------------------------------------------------------------------------
echo $cmd
echo ----------------------------------------------------------------------------------
$cmd || exit 1
#
sol=fail; CL=7;
cmd="srun -N $N --ntasks-per-node=$TPN $EXE --mpiBayesGroups --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin 1  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --read-from-bed-file --number-markers $NUMSNPS --number-individuals $NUMINDS --groupIndexFile $datadir/$grp --groupMixtureFile $datadir/$mix --save 5"
echo ----------------------------------------------------------------------------------
echo $cmd
echo ----------------------------------------------------------------------------------
$cmd || exit 1
#
# RESTART previous failed chain
sol=fail; CL=14;
cmd="srun -N $N --ntasks-per-node=$TPN $EXE --mpiBayesGroups --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin 1  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --read-from-bed-file --number-markers $NUMSNPS --number-individuals $NUMINDS --groupIndexFile $datadir/$grp --groupMixtureFile $datadir/$mix --save 5 --restart"
echo ----------------------------------------------------------------------------------
echo $cmd
echo ----------------------------------------------------------------------------------
$cmd || exit 1

exit 0;

echo
echo
echo "@@@ MPI 1-task solution reading from SPARSE files @@@"
echo
sol=mpi1tsparse
cmd="srun -N $N --ntasks-per-node=$TPN $EXE --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin 1  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --sync-rate $SR --number-markers $NUMSNPS --number-individuals $NUMINDS --sparse-dir $sparsedir --sparse-basename $sparsebsn"
echo ----------------------------------------------------------------------------------
echo $cmd
echo ----------------------------------------------------------------------------------
$cmd || exit 1


echo
echo
echo "@@@ MPI 1-task solution reading from SPARSE files and sparse sync @@@"
echo
sol=mpi1tsparse_sparseSync
cmd="srun -N $N --ntasks-per-node=$TPN $EXE --mpibayes bayesMPI --sparse-sync --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin 1  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --number-markers $NUMSNPS --number-individuals $NUMINDS --sparse-dir $sparsedir --sparse-basename $sparsebsn"
echo ----------------------------------------------------------------------------------
echo $cmd
echo ----------------------------------------------------------------------------------
$cmd || exit 1
#--marker-blocks-file $datadir/${dataset}.blk_1 
#--covariates $datadir/scaled_covariates.csv

