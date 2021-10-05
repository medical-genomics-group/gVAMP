

#!/bin/bash

module purge

# module load openmpi/3.1.2-intel2018 eigen/3.3.7 boost/1.75.0 intel2018/compiler2018 
module load openmpi/4.1.1 eigen/3.3.9 boost/1.77.0
# module load openmpi/4.0.1-cuda10.0 eigen/3.3.9 boost/1.77.0 intel2018/compiler2018 cuda

export EIGEN_ROOT="/mnt/nfs/clustersw/Debian/buster/eigen/3.3.7"


module list

make -f Makefile_G clean
make -f Makefile_G

