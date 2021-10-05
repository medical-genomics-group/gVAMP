#!/bin/bash

module purge

module load intel intel-mpi boost eigen

make $1
