#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
# SBATCH --mem 6gb
#SBATCH --mem 0
#SBATCH --time 0-00:15:00
#SBATCH --output=main_1012022_SK_noSIMD.log
#SBATCH --exclude=bjoern55,bjoern52,delta206
# SBATCH --constraint=gamma|delta

module purge 

ml gcc openmpi boost

module list 

export OMP_NUM_THREADS=2

mpic++ main.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main.exe

time mpirun -np 4 ./main.exe

