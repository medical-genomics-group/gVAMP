#pragma once

#include <mpi.h>
#include "utils.hpp"

// Original csv output file
void write_ofile_csv(const MPI_File fh, const uint iteration, const VectorXd sigmaG, const double sigmaE, const VectorXi m0,
                     const uint n_thinned_saved, const MatrixXd estPi);


// Same info as in .csv but in binary format rather than text (for full precision)
void write_ofile_out(const MPI_File fh, const uint iteration, const VectorXd sigmaG, const double sigmaE, const VectorXi m0,
                     const uint n_thinned_saved, const MatrixXd estPi);



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


// file layout: /HISTORY/ iteration | single value
template <class T>
void write_ofile_s1 (MPI_File fh, const uint iteration, const uint n_thinned_saved, const T data, const MPI_Datatype mpi_type) {

    MPI_Status status;

    int mpi_type_size = 0;
    MPI_Type_size(mpi_type, &mpi_type_size);

    MPI_Offset offset = size_t(n_thinned_saved) * ( sizeof(uint) + mpi_type_size );

    check_mpi(MPI_File_write_at(fh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

    offset += sizeof(uint);

    check_mpi(MPI_File_write_at(fh, offset, &data,      1, MPI_DOUBLE,   &status), __LINE__, __FILE__);
}

template <class T>
void read_ofile_s1 (string fp,
                    const uint iteration_to_restart_from,
                    const uint first_thinned_iteration,
                    const int  opt_thin,
                    T data,
                    MPI_Datatype mpi_type) {
    
    MPI_Status status;
    MPI_File   fh;

    int mpi_type_size = 0;
    MPI_Type_size(mpi_type, &mpi_type_size);

    check_mpi(MPI_File_open(MPI_COMM_SELF, fp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), __LINE__, __FILE__);

    int n_skip = iteration_to_restart_from - first_thinned_iteration;
    assert(n_skip >= 0);
    assert(n_skip % opt_thin == 0);
    n_skip /= opt_thin;

    MPI_Offset offset = (size_t)n_skip * (sizeof(uint) + mpi_type_size);

    // 1. Read iteration and check against expected one
    uint iteration_   = UINT_MAX;
    check_mpi(MPI_File_read_at(fh, offset, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_to_restart_from) {
        printf("Mismatch between expected and read eps iteration: %d vs %d\n", iteration_to_restart_from, iteration_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    offset += sizeof(uint);

    // 2. Read the data
    check_mpi(MPI_File_read_at(fh, offset, data, 1, mpi_type, &status), __LINE__, __FILE__);

    check_mpi(MPI_File_close(&fh), __LINE__, __FILE__);
}




// file layout: /HISTORY/  iteration | size | data
//                           UINT      UINT    T
template <class T>
void write_ofile_hsa (MPI_File fh, const uint iteration, const uint n_thinned_saved, const uint size, const T* data, const MPI_Datatype mpi_type) {

    MPI_Status status;

    int mpi_type_size = 0;
    MPI_Type_size(mpi_type, &mpi_type_size);

    MPI_Offset offset = size_t(n_thinned_saved) * ( sizeof(uint) + sizeof(uint) + (size_t)size * mpi_type_size);

    check_mpi(MPI_File_write_at(fh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    offset += sizeof(uint);

    check_mpi(MPI_File_write_at(fh, offset, &size,      1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    offset += sizeof(uint);

    check_mpi(MPI_File_write_at(fh, offset, data,   size, mpi_type,   &status), __LINE__, __FILE__);
}

template <class T>
void read_ofile_hsa (string fp,
                     const uint iteration_to_restart_from,
                     const uint first_thinned_iteration,
                     const int  opt_thin,
                     const uint size,
                     T data,
                     MPI_Datatype mpi_type) {

    MPI_Status status;
    MPI_File   fh;

    int mpi_type_size = 0;
    MPI_Type_size(mpi_type, &mpi_type_size);

    check_mpi(MPI_File_open(MPI_COMM_SELF, fp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), __LINE__, __FILE__);

    int n_skip = iteration_to_restart_from - first_thinned_iteration;
    assert(n_skip >= 0);
    assert(n_skip % opt_thin == 0);
    n_skip /= opt_thin;

    MPI_Offset offset = (size_t)n_skip * (sizeof(uint) + sizeof(uint) + (size_t)size * mpi_type_size);

    // 1. Read iteration and check against expected one
    uint iteration_   = UINT_MAX;
    check_mpi(MPI_File_read_at(fh, offset, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_to_restart_from) {
        printf("Mismatch between expected and read eps iteration: %d vs %d\n", iteration_to_restart_from, iteration_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    offset += sizeof(uint);

    // 2. Get and validate array length
    uint size_ = 0;
    check_mpi(MPI_File_read_at(fh, offset, &size_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (size_ != size) {
        printf("Mismatch between expected and read size of vector: %d vs %d\n", size, size_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    offset += sizeof(uint);

    // 3. Read the data
    check_mpi(MPI_File_read_at(fh, offset, data, size_, mpi_type, &status), __LINE__, __FILE__);

    check_mpi(MPI_File_close(&fh), __LINE__, __FILE__);
}




// file layout:  /SINGLE-LINE/ iter | size | data
//
template <class T>
void write_ofile_t1 (MPI_File fh, uint iteration, uint size, T data, MPI_Datatype mpi_type) {

    MPI_Status status;

    MPI_Offset offset = 0;
   
    check_mpi(MPI_File_write_at(fh, offset, &iteration, 1,    MPI_UNSIGNED, &status), __LINE__, __FILE__);
    offset += sizeof(uint);

    check_mpi(MPI_File_write_at(fh, offset, &size,      1,    MPI_UNSIGNED, &status), __LINE__, __FILE__);
    offset += sizeof(uint);

    check_mpi(MPI_File_write_at(fh, offset, data,       size, mpi_type,     &status), __LINE__, __FILE__);
}

template <class T>
void read_ofile_t1 (string fp, const uint iteration_to_restart_from, const uint length, T* data, MPI_Datatype mpi_type) {

    MPI_Status status;
    
    MPI_File fh;
    check_mpi(MPI_File_open(MPI_COMM_SELF, fp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), __LINE__, __FILE__);

    MPI_Offset offset = size_t(0);
    
    // 1. Read iteration and check against expected one
    uint iteration_   = UINT_MAX;
    check_mpi(MPI_File_read_at(fh, offset, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_to_restart_from) {
        printf("Mismatch between expected and read eps iteration: %d vs %d\n", iteration_to_restart_from, iteration_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    offset += sizeof(uint);

    // 2. Get and validate array length
    uint length_ = 0;
    check_mpi(MPI_File_read_at(fh, offset, &length_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (length_ != length) {
        printf("Mismatch between expected and read length of gamma vector: %d vs %d\n", length, length_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    offset += sizeof(uint);

    // 3. Read the data
    check_mpi(MPI_File_read_at(fh, offset, data, length_, mpi_type, &status), __LINE__, __FILE__);

    check_mpi(MPI_File_close(&fh), __LINE__, __FILE__);
}



template <class T>
void write_ofile_t2 (MPI_File fh, const int rank, const int mranks, const uint Mtot, const uint iteration, const uint size, const T* data, const MPI_Datatype mpi_type) {

    MPI_Status status;

    MPI_Offset offset = 0;
   
    if (rank == 0)
        check_mpi(MPI_File_write_at(fh, offset, &Mtot,      1,    MPI_UNSIGNED, &status), __LINE__, __FILE__);

    
    offset += sizeof(uint);

    if (rank == 0)
        check_mpi(MPI_File_write_at(fh, offset, &iteration, 1,    MPI_UNSIGNED, &status), __LINE__, __FILE__);

   
    offset += sizeof(uint);

    int mpi_type_size = 0;
    MPI_Type_size(mpi_type, &mpi_type_size);

    offset += size_t(mranks) * mpi_type_size;

    check_mpi(MPI_File_write_at(fh, offset, data,       size, mpi_type,     &status), __LINE__, __FILE__);
}

