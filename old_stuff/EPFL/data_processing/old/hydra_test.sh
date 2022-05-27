#!/bin/bash
##SBATCH --partition parallel
# SBATCH --time=00:20:00
# SBATCH --mem=0
# SBATCH --nodes=1
# SBATCH --ntasks=1
# SBATCH --cpus-per-task=1

module purge
module load intel intel-mpi intel-mkl boost eigen
module list

DS=$1; SR=$2
if [ -z $DS ]; then echo "Fatal. DS as first arg" && exit 1; fi
if [ -z $SR ]; then echo "Fatal. SR as second arg" && exit 1; fi
echo DS = $DS; echo SR = $SR

#env | grep SLURM

#EXE

export EXE=hydra
cd ./src
B='-B'
B=''
make $B -f Makefile || exit 1;
cd ..

EXE=./src/$EXE

if [ ! -f $EXE ]; then
    echo Fatal: binary $EXE not found!
    exit
fi

datadir=/work/ext-unil-ctgg/marion/benchmark_simulation/data
dataset=ukb_chr2_N_QC

sparsedir=/work/ext-unil-ctgg/marion/benchmark_simulation/data
sparsebsn=ukb_chr2_N_QC_sparse

phen=/work/ext-unil-ctgg/marion/benchmark_simulation/phen/sim_1/data.noHEAD.phen 

NUMINDS=20000
NUMSNPS=328383
NUMSNPS=1000
SEED=1234
#SEED=34776899
#SEED=628326244
#SR=1
SM=1
CL=10
THIN=4
SAVE=8
N=1
TPN=1
CPT=6 # CPUs per task
export OMP_NUM_THREADS=$CPT
export KMP_AFFINITY=verbose
export KMP_AFFINITY=noverbose


# select what to run
run_bed_sync_dp=0;   run_sparse_sync_dp=0;    run_mixed_sync_dp=0;    run_sparse_sync_sparse=0;   run_mixed_sync_sparse=0;
run_bed_sync_dp_groups=0;   run_sparse_sync_dp_groups=0;    run_mixed_sync_dp_groups=0;    run_sparse_sync_sparse_groups=0;   run_mixed_sync_sparse_groups=0;


if [ $DS == 0 ]; then
    S="0.0001,0.001,0.01"
    outdir=/work/ext-unil-ctgg/marion/code_test/bed_dp
    output=hydra_test_bed_dp
    run_bed_sync_dp=1
    
elif [ $DS == 1 ]; then
    S="0.0001,0.001,0.01"
    outdir=/work/ext-unil-ctgg/marion/code_test/sparse_dp
    output=hydra_test_sparse_dp
    run_sparse_sync_dp=1

elif [ $DS == 2 ]; then
    S="0.0001,0.001,0.01"
    outdir=/work/ext-unil-ctgg/marion/code_test/mixed_dp
    output=hydra_test_mixed_dp
    run_mixed_sync_dp=1

elif [ $DS == 3 ]; then
    S="0.0001,0.001,0.01"
    outdir=/scratch/orliac/marion/code_test/sparse_sparse/
    output=hydra_test_sparse_sparse
    run_sparse_sync_sparse=1

elif [ $DS == 4 ]; then
    S="0.0001,0.001,0.01"
    outdir=/scratch/orliac/marion/code_test/mixed_sparse
    output=hydra_test_mixed_sparse
    run_mixed_sync_sparse=1
    
elif [ $DS == 5 ]; then
    outdir=/scratch/orliac/marion/code_test/bed_dp_groups
    output=hydra_test_bed_dp_groups
    grp=/work/ext-unil-ctgg/marion/benchmark_simulation/ukb_chr2_N_QC_groups7.group
    mix=/work/ext-unil-ctgg/marion/benchmark_simulation/0001_001_01_groups7.cva
    run_bed_sync_dp_groups=1
        
elif [ $DS == 6 ]; then
    outdir=/work/ext-unil-ctgg/marion/code_test/sparse_dp_groups
    output=hydra_test_sparse_dp_groups
    grp=/work/ext-unil-ctgg/marion/benchmark_simulation/ukb_chr2_N_QC_groups7.group
    mix=/work/ext-unil-ctgg/marion/benchmark_simulation/0001_001_01_groups7.cva
    run_sparse_sync_dp_groups=1

elif [ $DS == 7 ]; then
    outdir=/work/ext-unil-ctgg/marion/code_test/mixed_dp_groups
    output=hydra_test_mixed_dp_groups
    grp=/work/ext-unil-ctgg/marion/benchmark_simulation/ukb_chr2_N_QC_groups7.group
    mix=/work/ext-unil-ctgg/marion/benchmark_simulation/0001_001_01_groups7.cva
    run_mixed_sync_dp_groups=1

elif [ $DS == 8 ]; then
     outdir=/scratch/orliac/marion/code_test/sparse_sparse_groups
     output=hydra_test_sparse_sparse_groups
     grp=/work/ext-unil-ctgg/marion/benchmark_simulation/ukb_chr2_N_QC_groups7.group
     mix=/work/ext-unil-ctgg/marion/benchmark_simulation/0001_001_01_groups7.cva     
     run_sparse_sync_sparse_groups=1

elif [ $DS == 9 ]; then
    outdir=/scratch/orliac/marion/code_test/mixed_sparse_groups
     output=hydra_test_mixed_sparse_groups
     grp=/work/ext-unil-ctgg/marion/benchmark_simulation/ukb_chr2_N_QC_groups7.group
     mix=/work/ext-unil-ctgg/marion/benchmark_simulation/0001_001_01_groups7.cva
     run_mixed_sync_sparse_groups=1
  
fi


echo
echo "======================================"
echo "        RUNNING THE APPLICATION ON:   "
echo "DS:       " $DS
echo "outdir:   " $outdir
echo "output:   " $sparsedir
echo "======================================"
echo

mkdir -p  $outdir/



### BED data processing

if [ $run_bed_sync_dp == 1 ]; then
    echo; echo
    echo "@@@ BED + DP @@@"
    echo
    cmd="srun $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $output --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S"
    echo $cmd; echo
    $cmd || exit 1
fi

if [ $run_bed_sync_dp_groups == 1 ]; then
    echo; echo
    echo "@@@ BED + DP + GROUPS @@@"
    echo
    cmd="srun $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $output --seed $SEED --shuf-mark $SM --sync-rate $SR --groupIndexFile $grp --groupMixtureFile $mix"
    echo $cmd; echo
    $cmd || exit 1
fi



### SPARSE data processing

if [ $run_sparse_sync_dp == 1 ]; then
    echo; echo; echo "@@@ SPARSE + DP @@@"
    echo
    cmd="srun $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $output --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn"
    echo $cmd; echo
    $cmd || exit 1
fi

if [ $run_sparse_sync_dp_groups == 1 ]; then
    echo; echo; echo "@@@ SPARSE + DP + GROUPS @@@"
    echo
    cmd="srun $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $output --seed $SEED --shuf-mark $SM --sync-rate $SR --groupIndexFile $grp --groupMixtureFile $mix --sparse-dir $sparsedir  --sparse-basename $sparsebsn"
    echo $cmd; echo
    $cmd || exit 1
fi

if [ $run_sparse_sync_sparse == 1 ]; then
    echo; echo; echo "@@@ SPARSE + SPARSE @@@"
    echo

    if [ 1 == 0 ]; then
        cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $output --seed $SEED --shuf-mark $SM --sync-rate $SR --sparse-dir $sparsedir  --sparse-basename $sparsebsn --sparse-sync --S $S"
        echo $cmd; echo
        $cmd || exit 1
    else
        echo "@@@@@ REFERENCE"
        cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen --chain-length 33 --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name ${output}_ref --seed $SEED --shuf-mark $SM --sync-rate $SR --sparse-dir $sparsedir  --sparse-basename $sparsebsn --sparse-sync --S $S"
        echo $cmd; echo
        $cmd || exit 1

        echo; echo "@@@@@ FAIL"
        cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen --chain-length 10  --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $output --seed $SEED --shuf-mark $SM --sync-rate $SR --sparse-dir $sparsedir  --sparse-basename $sparsebsn --sparse-sync --S $S"
        echo $cmd; echo
        $cmd &>/dev/null || exit 1

        echo; echo "@@@@@ RESTART + FAIL"
        cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen --chain-length 19 --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $output --seed $SEED --shuf-mark $SM --sync-rate $SR --sparse-dir $sparsedir  --sparse-basename $sparsebsn --sparse-sync --S $S --restart"
        echo $cmd; echo
        $cmd #&>/dev/null || exit 1

        echo; echo "@@@@@ RE-RE-START"
        cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen --chain-length 33 --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name ${output}_rs --seed $SEED --shuf-mark $SM --sync-rate $SR --sparse-dir $sparsedir  --sparse-basename $sparsebsn --sparse-sync --S $S --restart --ignore-xfiles"
        echo $cmd; echo
        $cmd || exit 1

     fi
fi

if [ $run_sparse_sync_sparse_groups == 1 ]; then
    echo; echo; echo "@@@ SPARSE + SPARSE + GROUPS @@@"
    echo
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $output --shuf-mark $SM --sync-rate $SR --groupIndexFile $grp --groupMixtureFile $mix --sparse-dir $sparsedir  --sparse-basename $sparsebsn --sparse-sync --seed $SEED"
    echo $cmd; echo
    $cmd || exit 1
fi


### MIXED-representation data processing

if [ $run_mixed_sync_dp == 1 ]; then
    echo; echo; echo "@@@ MIXED + DP @@@"; echo
    cmd="srun $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $output --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --bfile $datadir/$dataset --sparse-dir $sparsedir  --sparse-basename $sparsebsn --threshold-fnz 0.06"
    echo $cmd; echo
    $cmd || exit 1
fi

if [ $run_mixed_sync_dp_groups == 1 ]; then
    echo; echo; echo "@@@ MIXED + DP + GROUPS @@@"; echo
    cmd="srun $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $output --seed $SEED --shuf-mark $SM --sync-rate $SR --groupIndexFile $grp --groupMixtureFile $mix --bfile $datadir/$dataset --sparse-dir $sparsedir  --sparse-basename $sparsebsn --threshold-fnz 0.06"
    echo $cmd; echo
    $cmd || exit 1
fi


if [ $run_mixed_sync_sparse == 1 ]; then
    echo; echo; echo "@@@ MIXED + SPARSE @@@"; echo
    if [ 1 == 0 ]; then
        cmd="srun $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $output --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --bfile $datadir/$dataset --sparse-dir $sparsedir  --sparse-basename $sparsebsn --threshold-fnz 0.06 --sparse-sync"
    else
        ampdir=/scratch/orliac/tmp_vtune_profile
        rm -r $ampdir
        amp="srun amplxe-cl –c hotspots –r $ampdir -data-limit=1000 -- "
        cmd="$amp  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $output --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --bfile $datadir/$dataset --sparse-dir $sparsedir  --sparse-basename $sparsebsn --threshold-fnz 0.06 --sparse-sync"
    fi
    echo $cmd; echo
    $cmd || exit 1
fi

if [ $run_mixed_sync_sparse_groups == 1 ]; then
    echo; echo; echo "@@@ MIXED + SPARSE + GROUPS @@@"; echo 
    cmd="srun $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $output --seed $SEED --shuf-mark $SM --sync-rate $SR --groupIndexFile $grp --groupMixtureFile $mix --bfile $datadir/$dataset --sparse-dir $sparsedir  --sparse-basename $sparsebsn --threshold-fnz 0.06 --sparse-sync"
    echo $cmd; echo
    $cmd || exit 1
fi

