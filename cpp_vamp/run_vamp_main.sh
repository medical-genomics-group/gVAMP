#!/bin/bash

# SBATCH --reservation=robingrp_106
#SBATCH --ntasks 5
#SBATCH --cpus-per-task 6
#SBATCH --mem 0
#SBATCH --time 00:15:00
#SBATCH --output=emvamp_height_rho098_gamwL_PL_truesignalOn_testset_it10_CG10_31082022_MPI_5_main.log
#SBATCH --exclude=bjoern55,bjoern52

module load gcc openmpi boost

module list 

export OMP_NUM_THREADS=1

mpic++ main.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main.exe

time mpirun -np 5 ./main.exe
