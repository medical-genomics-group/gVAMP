#!/bin/bash

# SBATCH --reservation=robingrp_106
#SBATCH --nodes 6
#SBATCH --ntasks-per-node 2
#SBATCH --cpus-per-task 2
# SBATCH --mem 6gb
#SBATCH --mem 0
#SBATCH --time 1-20:15:00
#SBATCH --output=main_0612022_SK_gamma_noSIMD_corr_fd_maf_ver4.log
#SBATCH --exclude=bjoern55,bjoern52,delta206
# SBATCH --constraint=gamma

module purge 

ml gcc openmpi boost

module list 

export OMP_NUM_THREADS=2

# -DMANVECT
# mpic++ main.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main.exe

# -DMANVECT
# -Duse_shmem
 mpic++ main_corr.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main_corr.exe

# mpic++ main_corr_big.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main_corr_big.exe

# time mpirun -np 1 ./main.exe

 time mpirun -np 12 ./main_corr.exe

# time mpirun -np 40 ./main_corr_big.exe
