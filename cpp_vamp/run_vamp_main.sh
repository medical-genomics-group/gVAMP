#!/bin/bash

# SBATCH --reservation=robingrp_106
#SBATCH --nodes 8
# SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 1
#SBATCH --mem 7gb
# SBATCH --mem 0
#SBATCH --time 1-20:15:00
#SBATCH --output=main_gam11_normal2_3_length_CG60_precondCG_1112022_SK_gamma_corr_fd_maf_500k_400k.log
#SBATCH --exclude=bjoern55,bjoern52,delta206
# SBATCH --constraint=gamma|delta

module purge 

ml gcc openmpi boost

module list 

export OMP_NUM_THREADS=1

# -DMANVECT
# mpic++ main.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main.exe

# -DMANVECT
# -Duse_shmem
 mpic++ main_corr.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main_corr.exe

# mpic++ main_corr_big.cpp vamp.cpp utilities.cpp data.cpp options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  main_corr_big.exe

# time mpirun -np 1 ./main.exe

 time mpirun -np 8 ./main_corr.exe

# time mpirun -np 40 ./main_corr_big.exe
