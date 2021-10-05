#!/bin/bash

BAYES=""
COMPILER=""

while [[ $# -gt 0 ]]
do
    key="${1}"
    #echo key = $key
    case ${key} in
    -b|--bayes)
        BAYES="${2}";
        shift # past argument
        shift # past value
        ;;
    -c|--compiler)
        COMPILER="${2}";
        shift # past argument
        shift # past value
        ;;
    -h|--help)
        echo "Show help"
        shift # past argument
        ;;
    *)    # unknown option
        shift # past argument
        ;;
    esac
done

[ -z $BAYES ]    && echo "Fatal: mandatory option -b|--bayes is missing"    && exit 1
[ -z $COMPILER ] && echo "Fatal: mandatory option -c|--compiler is missing" && exit 1

echo "Bayes type = $BAYES"
echo "Compiler   = $COMPILER"
echo "SRUN_OPTS  = $SRUN_OPTS"

[[ "$COMPILER" != "GCC" && "$COMPILER" != "Intel" ]] && echo "FATAL: Only GCC and Intel accepted. >>$COMPILER<< is not a valid option." && exit 1


# Load the modules and set the executable
module purge
if [[ "$COMPILER" = "GCC" ]]; then
    module load gcc mvapich2 boost eigen
    SUFF=_g
else
    module load intel intel-mpi boost eigen
    SUFF=_i
fi
module list

EXE=../../bin/hydra$SUFF
[ ! -f $EXE ] && echo "FATAL: hydra executable ($EXE) not found! " && exit 1
echo "EXE        = $EXE"


mpi_name=refactor
mpi_dir=/scratch/orliac/ojavee/sim_UK22_matt

sparsedir=/work/ext-unil-ctgg/marion/benchmark_simulation/data
sparsebsn=ukb_chr2_N_QC_sparse
phen=/work/ext-unil-ctgg/marion/benchmark_simulation/phen/sim_1/data.noHEAD.phen 
SEED=123
NUMINDS=20000
NUMSNPS=328383
NUMSNPS=11111


EXE_OPTS="\
    --mpibayes           $BAYES \
    --sparse-dir         $sparsedir \
    --sparse-basename    $sparsebsn \
    --pheno              $phen \
    --burn-in            0 \
    --thin               2 \
    --save               4 \
    --mcmc-out-dir       $mpi_dir \
    --mcmc-out-name      $mpi_name \
    --seed               $SEED \
    --shuf-mark          1 \
    --number-markers     $NUMSNPS \
    --number-individuals $NUMINDS \
    --S                  0.0001,0.001,0.01 \
    --sync-rate          1 \
    --ignore-xfiles      \
    --bfile              $sparsedir/ukb_chr2_N_QC.bed"
#    --verbosity          3"


OUT1=out.1$SUFF
OUT2=out.2$SUFF

FAIL=7
CL=15

test_restart=1

if [ 1 == 1 ]; then
    
    [ -f $OUT1 ] && rm $OUT1
    [ -f $OUT2 ] && rm $OUT2
    
    srun $SRUN_OPTS $EXE $EXE_OPTS --chain-length $CL              > $OUT1 #|  grep RESULT  |  tail -n 1
    srun $SRUN_OPTS $EXE $EXE_OPTS --chain-length $FAIL            >  /dev/null 2>&1
    srun $SRUN_OPTS $EXE $EXE_OPTS --chain-length $CL   --restart  | tee $OUT2 #|  grep RESULT  |  tail -n 1
    
    #tkdiff $OUT1 $OUT2 &
    
else
    
    srun $SRUN_OPTS $EXE $EXE_OPTS --chain-length $CL             |  grep RESULT  |  tail -n 1
    if [ $test_restart == 1 ]; then
        srun $SRUN_OPTS $EXE $EXE_OPTS --chain-length $FAIL           >  /dev/null 2>&1
        srun $SRUN_OPTS $EXE $EXE_OPTS --chain-length $CL   --restart |  grep RESULT  |  tail -n 1
    fi
    #exit 0;
    
    echo "@@@ Switching to REF @@@"
    
    #EXE=./src/hydra_ref
    #srun $SRUN_OPTS $EXE $EXE_OPTS --chain-length $CL              |  grep RESULT  |  tail -n 1
    #srun $SRUN_OPTS $EXE $EXE_OPTS --chain-length $FAIL            >  /dev/null 2>&1
    #srun $SRUN_OPTS $EXE $EXE_OPTS --chain-length $CL   --restart  |  grep RESULT  |  tail -n 1
    
    EXE=./src/hydra_master
    srun $SRUN_OPTS $EXE $EXE_OPTS --chain-length $CL              |  grep RESULT  |  tail -n 1
    if [ $test_restart == 1 ]; then
        srun $SRUN_OPTS $EXE $EXE_OPTS --chain-length $FAIL            >  /dev/null 2>&1
        srun $SRUN_OPTS $EXE $EXE_OPTS --chain-length $CL   --restart  |  grep RESULT  |  tail -n 1
    fi
fi

