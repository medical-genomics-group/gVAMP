#!/bin/bash

# SBATCH --reservation=robingrp_106
#SBATCH --ntasks 5
#SBATCH --cpus-per-task 2
#SBATCH --mem 0
#SBATCH --time 2-00:15:00
#SBATCH --output=main.log
#SBATCH --exclude=bjoern55,bjoern52,delta206
# SBATCH --constraint=gamma

module load gcc openmpi boost

module list 

export OMP_NUM_THREADS=2

mpic++ main.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main.exe

# mpic++ main_corr.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main_corr.exe

# mpic++ main_corr_big.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main_corr_big.exe

time mpirun -np 5 ./main.exe

# time mpirun -np 5 ./main_corr.exe

#time mpirun -np 40 ./main_corr_big.exe
