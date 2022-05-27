
#!/bin/bash

module purge

module load openmpi/3.1.2-intel2018 eigen/3.3.7 boost/1.72.0 intel2018/compiler2018 
module list

cmake --version
make $1

