#!/bin/bash

set -e
#set -x

COMPILER=gcc
B=""
print_help() {
cat <<-HELP

Call the script like this:

sh $0 [options]
  -c|--compiler   Select compiler, GCC or Intel; default is GCC.
  -B              To force recompilation.

HELP
exit 0
}

# Parse Command Line Arguments
while [ "$#" -gt 0 ]; do
    case "$1" in
        --compiler*|-c*)
            if [[ "$1" != *=* ]]; then shift; fi
            COMPILER="${1#*=}"
            ;;
        -B)
            B="-B"
            ;;
        --help|-h)
            print_help;;
        *)
            printf "************************************************************\n"
            printf "* Error: Invalid argument, run --help for valid arguments. *\n"
            printf "************************************************************\n"
            exit 1
            ;;
    esac
    shift
done

if [ $COMPILER == "gcc" ]; then
    echo "GCC"
    export MV2_ENABLE_AFFINITY=0
elif [ $COMPILER == "intel" ]; then
    echo "INTEL"
else 
    echo "fatal: unknown compiler $COMPILER. Check --help."
fi

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
source $DIR/../compile_with_${COMPILER}.sh $B

PGM_ROOT=$ARDYH_ROOT
PGM_EXE=$ARDYH_EXE

BENCH_DIR=/home/orliac/DCSR/CTGG/ardyh/dataset/mt_test
[ -d $BENCH_DIR ] || (echo "fatal: bench directory not found! $BENCH_DIR" && exit)
echo BENCH_DIR = $BENCH_DIR

SOC=1
TPS=2
NTASKS=`echo "$SOC * $TPS" | bc`
echo NTASKS = $NTASKS with $SOC sockets and $TPS tasks per socket.
CPT=8

export OMP_NUM_THREADS=$CPT
#export OMP_DISPLAY_ENV="TRUE" 

CMD_BASE="srun \
--partition build \
--ntasks $NTASKS \
--ntasks-per-socket $TPS \
--cpus-per-task $CPT \
--time 00:10:00 \
--mem 10G \
--cpu-bind=verbose \
$PGM_ROOT/bin/$PGM_EXE \
--bed-file $BENCH_DIR/test2.bed \
--dim-file $BENCH_DIR/test2.dim \
--trunc-markers 2005 \
--verbosity 2 \
--out-dir /scratch/orliac/pred_out \
--predict \
--bim-file $BENCH_DIR/test2.bim \
--ref-bim-file $BENCH_DIR/test1.bim"

PHEN1=$BENCH_DIR/test1.phen
PHEN2=$BENCH_DIR/test2.phen
PHEN3=$BENCH_DIR/test3.phen
PHEN4=$BENCH_DIR/test4.phen
PHEN5=$BENCH_DIR/test5.phen

PHENS1="--phen-files $PHEN1"
PHENS2="--phen-files $PHEN2"
PHENS3="--phen-files $PHEN3"
PHENS4="--phen-files $PHEN4"
PHENS5="--phen-files $PHEN5"
PHENS1_2="--phen-files $PHEN1,$PHEN2"
PHENS1_3="--phen-files $PHEN1,$PHEN2,$PHEN3"
PHENS1_4="--phen-files $PHEN1,$PHEN2,$PHEN3,$PHEN4"
PHENS1_5="--phen-files $PHEN1,$PHEN2,$PHEN3,$PHEN4,$PHEN5"
PHENS15="--phen-files $PHEN1,$PHEN5"

PHENS3_5="--phen-files $PHEN3,$PHEN4,$PHEN5"


CMD=${CMD_BASE}" "${PHENS2};  echo CMD = $CMD; $CMD;  #exit 0
#CMD=${CMD_BASE}" "${PHENS5};  echo CMD = $CMD; $CMD;  #exit 0
#CMD=${CMD_BASE}" "${PHENS15}; echo CMD = $CMD; $CMD;  #exit 0

#echo; echo; CMD=${CMD_BASE}" "${PHENS1};  echo CMD = $CMD; $CMD;  #exit 0
#echo; echo; CMD=${CMD_BASE}" "${PHENS2};  echo CMD = $CMD; $CMD;  #exit 0
#echo; echo; CMD=${CMD_BASE}" "${PHENS3};  echo CMD = $CMD; $CMD;  #exit 0
#echo; echo; CMD=${CMD_BASE}" "${PHENS4};  echo CMD = $CMD; $CMD;  #exit 0
#echo; echo; CMD=${CMD_BASE}" "${PHENS5};  echo CMD = $CMD; $CMD;  #exit 0
#CMD=${CMD_BASE}" "${PHENS1_2}; echo CMD = $CMD; $CMD;
#CMD=${CMD_BASE}" "${PHENS1_3}; echo CMD = $CMD; $CMD;
#CMD=${CMD_BASE}" "${PHENS1_4}; echo CMD = $CMD; $CMD;
#echo; echo; CMD=${CMD_BASE}" "${PHENS1_5}; echo CMD = $CMD; $CMD;

#CMD=${CMD_BASE}" "${PHENS3_5}; echo CMD = $CMD; $CMD;

