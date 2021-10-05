#!/bin/bash

module purge

module load gcc mvapich2 boost eigen

make $1
