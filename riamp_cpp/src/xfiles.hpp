#pragma once

#include <vector>
#include <mpi.h>
#include "utilities.hpp"
#include "const.hpp"


// Original csv output file
void write_ofile_csv(const MPI_File fh, const uint iteration, const std::vector<double>* sigmaG, const double sigmaE, const int m0_sum,
                     const uint n_thinned_saved, const std::vector<std::vector<double>>* estPi);

// History file with layout: Mtot | [ iteration | rank_0_data ... rank_N_data ] 
template <class T>
void write_ofile_h1 (MPI_File fh, const uint rank, const uint Mtot, const uint iteration, const uint n_thinned_saved, const uint mranks, const uint size, const T* data, const MPI_Datatype mpi_type) {

    MPI_Status status;
    MPI_Offset offset;

    int mpi_type_size = 0;
    MPI_Type_size(mpi_type, &mpi_type_size);

    if (rank == 0) {

        if (n_thinned_saved == 0) {
            check_mpi(MPI_File_write_at(fh, size_t(0), &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        }

        offset = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * mpi_type_size);
        check_mpi(MPI_File_write_at(fh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    }
   
    offset = sizeof(uint) + sizeof(uint)
        + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * mpi_type_size)
        + size_t(mranks) * mpi_type_size;
    
    check_mpi(MPI_File_write_at(fh, offset, data, size, mpi_type, &status), __LINE__, __FILE__);
}
