#!/bin/bash

# SBATCH --account=ext-unil-ctgg
#SBATCH --mem=0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-socket=4
#SBATCH --cpus-per-task=4
#SBATCH --time 0-00:05:00
#SBATCH --partition debug
#SBATCH --output BINDTEST

module purge
module load intel intel-mpi eigen boost
module list

export OMP_NUM_THREADS=4

export SLURM_CPU_BIND=verbose
export SLURM_MEM_BIND=verbose

env | grep SLURM


srun sleep 1
echo "@@@@@"
srun --mem-bind=verbose --cpu-bind=cores sleep 1
echo "@@@@@"
srun --mem-bind=verbose --cpu-bind=cores --distribution=block:block sleep 1
