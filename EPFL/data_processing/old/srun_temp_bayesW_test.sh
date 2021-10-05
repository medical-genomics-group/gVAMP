#!/bin/bash

BAYES=""

while [[ $# -gt 0 ]]
do
    key="${1}"
    case ${key} in
    -b|--bayes)
        BAYES="${2}";
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
    shift
done

[ -z $BAYES ] && echo "Fatal: mandatory option -b|--bayes is missing" && exit 1

echo "requested bayes type = $BAYES"

if [ $BAYES != "bayesWMPI" ]; then
    echo "Not a valid option"
    exit 1
fi

echo "SRUN_OPTS = $SRUN_OPTS"
echo "EXE       = $EXE"


out_name=refactor
out_dir=/scratch/orliac/ojavee/sim_UK22_matt

dir=/work/ext-unil-ctgg/etienne/test_bw
bsn=t_M50K_N5K
phen=$dir/$bsn.phen
fail=$dir/$bsn.fail

SEED=1
NUMINDS=5000
NUMSNPS=50348
#NUMSNPS=1000
#NUMSNPS=14



EXE_OPTS="\
    --mpibayes           $BAYES \
    --sparse-dir         $dir \
    --sparse-basename    $bsn \
    --pheno              $phen \
    --failure            $fail \
    --burn-in            0 \
    --thin               1 \
    --mcmc-out-dir       $out_dir \
    --mcmc-out-name      $out_name \
    --seed               $SEED \
    --shuf-mark          0 \
    --number-markers     $NUMSNPS \
    --number-individuals $NUMINDS \
    --S                  0.001,0.01,0.1 \
    --quad_points        25 \
    --sync-rate          10 \
    --save               2"

CL=5
FA=3

if [ 1 == 0 ]; then
    export OMP_NUM_THREADS=1
    echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
    srun $SRUN_OPTS -c $OMP_NUM_THREADS $EXE $EXE_OPTS --chain-length $CL | grep RESULT | tail -n 1

    export OMP_NUM_THREADS=4
    echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
    srun $SRUN_OPTS -c $OMP_NUM_THREADS $EXE $EXE_OPTS --chain-length $CL | grep RESULT | tail -n 1
    srun $SRUN_OPTS -c $OMP_NUM_THREADS $EXE $EXE_OPTS --chain-length $CL | grep RESULT | tail -n 1

elif [ 1 == 0 ]; then

    export OMP_NUM_THREADS=1
    echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
    srun $SRUN_OPTS -c $OMP_NUM_THREADS $EXE $EXE_OPTS --chain-length $CL > out.1 #| grep RESULT | tail -n 1
    

    export OMP_NUM_THREADS=4
    echo "OMP_NUM_THREADS = $OMP_NUM_THREADS"
    srun $SRUN_OPTS -c $OMP_NUM_THREADS $EXE $EXE_OPTS --chain-length $CL > out.2 #| grep RESULT | tail -n 1
    #srun $SRUN_OPTS -c $OMP_NUM_THREADS $EXE $EXE_OPTS --chain-length $CL > out.3 #| grep RESULT | tail -n 1

    tkdiff out.1 out.2 &
    #tail -n 10 out.1 out.2 #out.3

fi


export OMP_NUM_THREADS=1

echo "@@@@ REF"
srun $SRUN_OPTS -c $OMP_NUM_THREADS $EXE $EXE_OPTS --chain-length $CL   > out.1 #         |  grep RESULT  | tail -n 1
echo "@@@@ FAIL"
srun $SRUN_OPTS -c $OMP_NUM_THREADS $EXE $EXE_OPTS --chain-length $FA   >  /dev/null 2>&1        |  grep RESULT  | tail -n 1 #>  /dev/null 2>&1
echo "@@@@ RESTART"
srun $SRUN_OPTS -c $OMP_NUM_THREADS $EXE $EXE_OPTS --chain-length $CL --restart > out.2  #|  grep RESULT  | tail -n 1

tkdiff out.1 out.2 &
