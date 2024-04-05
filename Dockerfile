# syntax=docker/dockerfile:1

FROM ubuntu
RUN ln -snf /usr/share/zoneinfo/Europe/London /etc/localtime && echo Europe/London > /etc/timezone
RUN apt update --yes && apt install build-essential --yes && apt install mpich --yes && apt install libboost-all-dev --yes && apt install git-all --yes && git clone https://github.com/medical-genomics-group/gVAMP.git
# CMD git clone https://github.com/medical-genomics-group/gVAMP.git 
CMD ls -all ./gVAMP/ &&  mpic++ ./gVAMP/main_real.cpp ./gVAMP/vamp.cpp ./gVAMP/utilities.cpp ./gVAMP/data.cpp ./gVAMP/options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  ./gVAMP/main_real.exe &&  ls -all ./gVAMP/ && mpirun -np 2 --allow-run-as-root ./gVAMP/main_real.exe
