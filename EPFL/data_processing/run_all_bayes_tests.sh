#!/bin/bash

cd $(dirname $0)

cd ../..
sh compile_with_intel.sh || exit 1
sh compile_with_gcc.sh   || exit 1
cd -

export OMP_NUM_THREADS=4

NT=3

export MV2_ENABLE_AFFINITY=0


export SRUN_OPTS="-n $NT -c $OMP_NUM_THREADS"

echo; echo "@@@@@@ BAYES R"
sh srun_bayes_test.sh --bayes bayesMPI -c Intel || exit 1
sh srun_bayes_test.sh --bayes bayesMPI -c GCC   || exit 1

tkdiff out.2_i out.2_g&

#echo; echo "@@@@@@ BAYES FH"
#sh srun_bayes_test.sh --bayes bayesFHMPI || exit 1

#echo; echo "@@@@@@ BAYES W"
#sh srun_temp_bayesW_test.sh --bayes bayesWMPI
