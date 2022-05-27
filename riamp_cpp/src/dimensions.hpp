#pragma once
#include <mpi.h>
#include "options.hpp"



class Dimensions {

public:
    Dimensions(const Options& opt) {
        read_dim_file(opt.get_dim_file());
        const unsigned int truncm = opt.get_truncm();
        if (truncm > 0 && truncm < Mt) 
            Mt = truncm;
        MPI_Comm_size(MPI_COMM_WORLD, &nranks);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    }
    int get_nt() const { return Nt; }
    int get_mt() const { return Mt; }
    int get_rank() const { return rank; }
    int get_nranks() const { return nranks; }
private:
    int Nt = 0;
    int Mt = 0;
    int rank = 0;
    int nranks = 0;
    void read_dim_file(const std::string);
};
