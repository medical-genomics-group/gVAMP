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
module list


# COMPILATION
cd ./src
B='-B'
B=''
export EXE=hydra
make $B -f Makefile || exit 1;
cd ..

EXE=./src/$EXE

if [ ! -f $EXE ]; then
    echo Fatal: binary $EXE not found!
    exit
fi

COV=""
BLK=""
S="1.0,0.1"

DS=0

if [ $DS == 0 ]; then
    sparse_dir=/work/ext-unil-ctgg/robinson/
    sparse_bsn=ukb_imp_v3_UKB_EST_oct19_unrelated
    phen_file=/work/ext-unil-ctgg/marion/ukb_imp_v3_UKB_EST_oct19_pheno_w35520NA/ukb_imp_v3_UKB_EST_oct19_height_wNA_unrelated_148covadjusted_w35520NA.phen
    #--mcmc-out-name groups36_mix4_unrelated_cpus4_tasks8_nodes28_mpisync10_height_1_unrelated_148covadjusted_w35520NA
    group_index_file=/work/ext-unil-ctgg/marion/annot/ukb_annot_6_maf_3_ld_2_bins/ukb_imp_v3_UKB_EST_oct19_unrelated_annot_6_maf_3_ld_2_bins.group
    group_mixture_file=/work/ext-unil-ctgg/marion/annot/ukb_annot_6_maf_3_ld_2_bins/ukb_imp_v3_UKB_EST_oct19_unrelated_annot_6_maf_3_ld_2_bins_cpn_4.cva
    NUMINDS=382466
    NUMSNPS=8430446
    NUMSNPS=10000

    out_dir=/scratch/orliac/ukb_height_36groups    
fi

N=1;       TPN=6;    CPT=6;

CL=2;      SEED=10
SR=1;      SM=1
THIN=3;    SAVE=4

FNZ=0.060;

export OMP_NUM_THREADS=$CPT
export KMP_AFFINITY=verbose
export KMP_AFFINITY=noverbose


# Select what to run
run_bed_sync_dp=0;      run_sparse_sync_dp=0;      run_mixed_sync_dp=0;
run_bed_sync_sparse=0;  run_sparse_sync_sparse=0;  run_mixed_sync_sparse=1;
run_bed_sync_bed=0;     run_sparse_sync_bed=0;     run_mixed_sync_bed=0;


CMD_COMMON="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $phen_file --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $out_dir --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S"

if [ $run_sparse_sync_sparse == 1 ]; then
    echo; echo; echo "@@@ SPARSE + SPARSE @@@"; echo
    sol2=sparse_sync_sparse
    cmd="$CMD_COMMON  --mcmc-out-name $sol2 --sparse-dir $sparse_dir --sparse-basename $sparse_bsn --sparse-sync"
    echo $cmd; echo
    $cmd || exit 1

    echo; echo; echo "@@@ SPARSE + SPARSE + 36 GROUPS @@@"; echo
    sol2=sparse_sync_sparse
    cmd="$CMD_COMMON  --mcmc-out-name $sol2 --sparse-dir $sparse_dir --sparse-basename $sparse_bsn --sparse-sync --groupIndexFile $group_index_file --groupMixtureFile $group_mixture_file"
    echo $cmd; echo
    $cmd || exit 1
fi


if [ $run_mixed_sync_sparse == 1 ]; then
    echo; echo; echo "@@@ MIXED + SPARSE + 36 GROUPS @@@"; echo
    sol2=mixed_sync_sparse_ref
    cmd="$CMD_COMMON  --chain-length 23 --mcmc-out-name $sol2 --sparse-dir $sparse_dir --sparse-basename $sparse_bsn --sparse-sync --groupIndexFile $group_index_file --groupMixtureFile $group_mixture_file --bfile ./fake.bed --threshold-fnz $FNZ"
    echo $cmd; echo
    $cmd || exit 1

    sol2=mixed_sync_sparse
    cmd="$CMD_COMMON  --chain-length 17 --mcmc-out-name $sol2 --sparse-dir $sparse_dir --sparse-basename $sparse_bsn --sparse-sync --groupIndexFile $group_index_file --groupMixtureFile $group_mixture_file --bfile ./fake.bed --threshold-fnz $FNZ"
    echo $cmd; echo
    $cmd || exit 1

    cmd="$CMD_COMMON  --chain-length 23 --mcmc-out-name $sol2 --sparse-dir $sparse_dir --sparse-basename $sparse_bsn --sparse-sync --groupIndexFile $group_index_file --groupMixtureFile $group_mixture_file --bfile ./fake.bed --threshold-fnz $FNZ --restart"
    echo $cmd; echo
    $cmd || exit 1
    

    echo; echo; echo "@@@ MIXED + SPARSE @@@"; echo
    cmd="$CMD_COMMON  --mcmc-out-name $sol2 --sparse-dir $sparse_dir --sparse-basename $sparse_bsn --sparse-sync --bfile ./fake.bed --threshold-fnz $FNZ"
    echo $cmd; echo
    #$cmd || exit 1
fi


exit 0




### BED data processing

if [ $run_bed_sync_dp == 1 ]; then
    echo; echo
    echo "@@@ BED + DP @@@"
    echo
    sol=test_mnm2
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $out_dir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi

if [ $run_bed_sync_sparse == 1 ]; then
    echo; echo
    echo "@@@ BED + SPARSE @@@"
    echo
    sol=test_mnm2
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --sparse-sync $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi

if [ $run_bed_sync_bed == 1 ]; then
    echo; echo
    echo "@@@ BED + BED @@@"
    echo
    sol=test_mnm2
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --bfile $datadir/$dataset --pheno $datadir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --bed-sync $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi


### SPARSE data processing

if [ $run_sparse_sync_dp == 1 ]; then
    echo; echo; echo "@@@ SPARSE + DP @@@"
    echo
    sol2=sparse_sync_dp
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol2 --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn --sparse-sync  $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi


if [ $run_sparse_sync_bed == 1 ]; then
    echo; echo; echo "@@@ SPARSE + BED @@@"; echo
    sol2=sparse_sync_bed
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol2 --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --sparse-dir $sparsedir  --sparse-basename $sparsebsn --bed-sync  $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi


### MIXED-representation data processing

if [ $run_mixed_sync_dp == 1 ]; then
    echo; echo; echo "@@@ MIXED + DP @@@"; echo
    sol2=mixed_sync_dp
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol2 --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --bfile $datadir/$dataset --sparse-dir $sparsedir  --sparse-basename $sparsebsn --threshold-fnz $FNZ $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi


AMPLIFIER=0;  ampdir=/scratch/orliac/tmp_vtune_profile;
ADVISOR=0;    advdir=/scratch/orliac/tmp_advis_profile;

if [ $run_mixed_sync_sparse == 1 ]; then
    echo; echo; echo "@@@ MIXED + SPARSE @@@"; echo
    sol2=mixed_sync_sparse

     CMD="$EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol2 --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --bfile $datadir/$dataset --sparse-dir $sparsedir  --sparse-basename $sparsebsn --threshold-fnz $FNZ --sparse-sync $COV $BLK"

     PRE="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT "

    if [ $AMPLIFIER == 1 ]; then
        echo "@@@ AMPLIFIER"
        [ -d $ampdir ] && rm -r $ampdir/*
        cmd_prefix=$PRE" amplxe-cl –c hotspots –r $ampdir -data-limit=1000 -- "

        cmd=$cmd_prefix$CMD
        echo $cmd; echo
        $cmd || exit 1

        amplxe-gui $ampdir/tmp_vtune_profile.amplxeproj &

    elif [ $ADVISOR == 1 ]; then
        echo "@@@ ADVISOR"
        source ~/load_Intel_Advisor.sh
        [ -d $advdir ] && rm -r $advdir/*

        cmd_prefix="advixe-cl -v -collect survey           --project-dir=$advdir -no-auto-finalize -data-limit=0 --search-dir all:=./src -- "
        cmd=$PRE$cmd_prefix$CMD
        echo $cmd; echo
        $cmd || exit 1

        cmd_prefix="advixe-cl -v -collect tripcounts -flop --project-dir=$advdir -no-auto-finalize -data-limit=0 --search-dir all:=./src -- "
        cmd=$PRE$cmd_prefix$CMD
        echo $cmd; echo
        $cmd || exit 1
        
        #cmd_prefix="advixe-cl --collect=roofline          --project-dir=$advdir  --no-auto-finalize                  --search-dir all:=./src -- "
        #cmd=$PRE$cmd_prefix$CMD
        #echo $cmd; echo
        #$cmd || exit 1

        echo "advixe-gui $advdir/tmp_advis_profile.advixeproj &"

    else
        cmd=$PRE$CMD
        echo $cmd; echo
        $cmd || exit 1
    fi
fi

if [ $run_mixed_sync_bed == 1 ]; then
    echo; echo; echo "@@@ MIXED + BED @@@"; echo
    #cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  cpuinfo"
    #echo $cmd
    #$cmd || exit 1

    #cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  ${HOME}/DCSR/Affinity/xthi_mpi"
    #echo $cmd
    #$cmd || exit 1

    sol2=mixed_sync_bed
    cmd="srun -N $N --ntasks-per-node=$TPN --cpus-per-task=$CPT  $EXE --number-individuals $NUMINDS --number-markers $NUMSNPS --mpibayes bayesMPI --pheno $sparsedir/${phen}.phen --chain-length $CL --thin $THIN --save $SAVE  --mcmc-out-dir $outdir --mcmc-out-name $sol2 --seed $SEED --shuf-mark $SM --sync-rate $SR --S $S --bfile $datadir/$dataset --sparse-dir $sparsedir  --sparse-basename $sparsebsn --threshold-fnz $FNZ --bed-sync $COV $BLK"
    echo $cmd; echo
    $cmd || exit 1
fi
