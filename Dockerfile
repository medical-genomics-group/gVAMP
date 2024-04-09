FROM ubuntu
RUN ln -snf /usr/share/zoneinfo/Europe/London /etc/localtime && echo Europe/London > /etc/timezone && \
    apt-get update --yes && \ 
    apt-get install build-essential --yes --no-install-recommends && \
    apt install mpich --yes --no-install-recommends && \
    apt install libboost-all-dev --yes --no-install-recommends && \
    apt install git-all --yes --no-install-recommends
RUN git clone https://github.com/medical-genomics-group/gVAMP.git
RUN mpic++ ./gVAMP/main_real.cpp ./gVAMP/vamp.cpp ./gVAMP/utilities.cpp ./gVAMP/data.cpp ./gVAMP/options.cpp -march=native -DMANVECT -Ofast -g -fopenmp -lstdc++fs -D_GLIBCXX_DEBUG -o  ./gVAMP/main_real.exe
COPY run_gvamp.sh /gVAMP/run_gvamp.sh
WORKDIR /gVAMP
CMD sh run_gvamp.sh