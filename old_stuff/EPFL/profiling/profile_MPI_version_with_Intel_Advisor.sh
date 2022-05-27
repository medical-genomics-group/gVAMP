#!/bin/bash
#
# Author : E. Orliac, DCSR, UNIL
# Date   : 2019/08/26
# Purpose: Run Intel profiling tools: Advisor
#
# Warning: needs to be in an active slurm allocation, execution via srun!
#
# Warning: the block definition file (.blk) and job setup must match (wrt the number of tasks)
#

module purge
module load intel intel-mpi intel-mkl boost eigen
module list


NAM=hydra

PROJ=/home/orliac/DCSR/CTGG/BayesRRcmd
EXE=$PROJ/src/$NAM

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
    NUMSNPS=20000
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

CL=5
SEED=9
SR=0
SM=0
THIN=3
SAVE=3
TOCONV_T=$((($CL - 1) / $THIN))
echo TOCONV_T $TOCONV_T
N=1
TPN=8
BLK=""

#TPN=$(wc -l "$BLKFILE")
#echo TPN = $TPN


outdir=/scratch/orliac/profiling/Intel_Advisor/profile_1_node
sol=test

### -genv OMP_NUM_THREADS=4 -genv I_MPI_PIN_DOMAIN=omp


SINGLE=1
MULTI=0
ADVISOR=0
THREADS=1

if [ $ADVISOR == 1 ]; then
    unset I_MPI_PMI_LIBRARY
    export SLURM_CPU_BIND=none
    echo "/!\ Switched to mpirun!"
    source ~/load_Intel_Advisor.sh
fi

if [ $SINGLE == 1 ]; then

    if [ $ADVISOR == 1 ]; then
        
        export OMP_NUM_THREADS=$THREADS
        export KMP_AFFINITY=verbose
        export I_MPI_PIN_DOMAIN=omp

        cmd1="mpirun -np $TPN -ppn $TPN  advixe-cl –-collect=survey            --project-dir=$outdir/Advisor/single_trait -no-auto-finalize -- numactl --membind=0 $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out-dir $outdir/results --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn $BLK"
        echo
        echo $cmd1
        echo
        $cmd1 || exit 1
        
        cmd2="mpirun -np $TPN -ppn $TPN  advixe-cl –-collect=tripcounts --flop --project-dir=$outdir/Advisor/single_trait -no-auto-finalize -- numactl --membind=0 $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out-dir $outdir/results --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn $BLK"
        echo
        echo $cmd2
        echo
        $cmd2 || exit 1
    else
        #for T in 1 2 4 8 16; do
        for T in 1 4; do
            export OMP_NUM_THREADS=$T
            export KMP_AFFINITY=verbose            
            cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$T  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn $BLK"
            echo; echo $cmd; echo; $cmd || exit 1
        done
    fi

    echo; echo @@@@@ END SINGLE-TRAIT SOLUTION; echo
fi


if [ $MULTI == 1 ]; then
    
    if [ $ADVISOR == 1 ]; then

        T=$THREADS
        export OMP_NUM_THREADS=$T
        export KMP_AFFINITY=verbose
        export I_MPI_PIN_DOMAIN=omp

        cmd1="mpirun -np $TPN -ppn $TPN  advixe-cl –-collect=survey            --project-dir=$outdir/Advisor/multi_trait -no-auto-finalize -- numactl --membind=0 $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen,$sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out-dir $outdir/results --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn $BLK"
        echo; echo $cmd1; echo; $cmd1 || exit 1
        
        cmd2="mpirun -np $TPN -ppn $TPN  advixe-cl –-collect=tripcounts --flop --project-dir=$outdir/Advisor/multi_trait -no-auto-finalize -- numactl --membind=0 $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen,$sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE --mcmc-out-dir $outdir/results --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn $BLK"
        echo; echo $cmd2; echo; $cmd2 || exit 1
        
    else

        phenList="$sparsedir/${phen}.phen,$sparsedir/${phen}.phen"

        #for T in 1 2 4 8 16; do
        for T in 1; do
            
            export OMP_NUM_THREADS=$T
            #export KMP_AFFINITY=verbose
            
            cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$T $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phenList                         --chain-length $CL --thin $THIN --save $SAVE --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn $BLK"
            echo; echo $cmd; echo; $cmd || exit 1

            # Interleaved 
            #cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$T $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phenList --interleave-phenotypes --chain-length $CL --thin $THIN --save $SAVE --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --mpi-sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn $BLK"
            #echo; echo $cmd; echo; $cmd || exit 1

        done
    fi

    echo; echo @@@@@ END MULTI-TRAITS SOLUTION; echo
fi
