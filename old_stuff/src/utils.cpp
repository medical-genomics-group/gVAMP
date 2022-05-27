#include <vector>
#include <mpi.h>
#include <climits>
#include <cassert>
#include <sys/stat.h>
#include "hydra.h"
#include "options.hpp"
#include "data.hpp"
#include "utils.hpp"


// Check malloc in MPI context
// ---------------------------
//inline
void check_malloc(const void* ptr, const int linenumber, const char* filename) {
    if (ptr == NULL) {
        fprintf(stderr, "#FATAL#: malloc failed on line %d of %s\n", linenumber, filename);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


// Check MPI call returned value. If error print message and call MPI_Abort()
// --------------------------------------------------------------------------
//inline
void check_mpi(const int error, const int linenumber, const char* filename) {
    if (error != MPI_SUCCESS) {
        fprintf(stderr, "*FATAL*: MPI error %d at line %d of file %s\n", error, linenumber, filename);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


// Check if size_t can be casted to int or would overflow
// ------------------------------------------------------
//inline 
int check_int_overflow(const size_t n, const int linenumber, const char* filename) {

    if (n > INT_MAX) {
        fprintf(stderr, "FATAL  : integer overflow detected on line %d of %s. %lu does not fit in type int.\n", linenumber, filename, n);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<int>(n);
}


MPI_File get_fh(fh_map file_handlers, std::string fp) {

    fh_it fit = file_handlers.find(fp);
    
    if (fit == file_handlers.end()) {
        printf("*FATAL*: file %s not found in file_handlers map", fp.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
     
    return fit->second;
}


// Check file size is the expected one
// -----------------------------------
void check_file_size(const MPI_File fh, const size_t N, const size_t DTSIZE, const int linenumber, const char* filename) {

    //printf("DEBUG  : expected size = %lu x %lu bytes\n", N, DTSIZE);
    size_t     expected_file_size = N * DTSIZE;

    MPI_Offset actual_file_size = 0;

    check_mpi(MPI_File_get_size(fh, &actual_file_size), __LINE__, __FILE__);

    if (actual_file_size != expected_file_size) {
        fprintf(stderr, "FATAL  : expected file size in bytes: %lu; actual file size: %lu from call on line %d of %s.\n", expected_file_size, actual_file_size, linenumber, filename);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Define blocks of markers to be processed by each task
// By default processes all markers
// -----------------------------------------------------
void define_blocks_of_markers(const int Mtot, int* MrankS, int* MrankL, const uint nblocks) {

    const uint modu   = Mtot % nblocks;
    const uint Mrank  = int(Mtot / nblocks);
    uint checkM = 0;
    uint start  = 0;

    for (int i=0; i<nblocks; ++i) {
        MrankL[i] = int(Mtot / nblocks);
        if (modu != 0 && i < modu)
            MrankL[i] += 1;
        MrankS[i] = start;
        //printf("start %d, len %d\n", MrankS[i], MrankL[i]);
        start += MrankL[i];
        checkM += MrankL[i];
    }
    assert(checkM == Mtot);
}


// Sanity check: make sure all elements were set (requires init at UINT_MAX)
// -------------------------------------------------------------------------
void check_whole_array_was_set(const uint* array, const size_t size, const int linenumber, const char* filename) {

    for (size_t i=0; i<size; i++) {
        if (array[i] == UINT_MAX) {
            printf("FATAL  : array[%lu] = %d not set at %d of %s!\n", i, array[i], linenumber, filename);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}


size_t get_file_size(const std::string& filename) {
    struct stat st;
    if(stat(filename.c_str(), &st) != 0) { return 0; }
    return st.st_size;
}



void assign_blocks_to_tasks(const uint numBlocks, const std::vector<int> blocksStarts, const std::vector<int> blocksEnds, const uint Mtot, const int nranks, const int rank, int* MrankS, int* MrankL, int& lmin, int& lmax) {

    if (numBlocks > 0) {

        if (nranks != numBlocks) {
            if (rank == 0) {
                printf("FATAL  : block definition does not match number of tasks (%d versus %d).\n", numBlocks, nranks);
                printf("        => Provide each task with a block definition\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // First and last markers (continuity is garanteed from reading)
        if (blocksStarts[0] != 1) {
            if (rank == 0) {
                printf("FATAL  : first marker in block definition file should be 1 but is %d\n", blocksStarts[0]);
                printf("        => Adjust block definition file\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        if (blocksEnds[numBlocks-1] != Mtot) {
            if (rank == 0) {
                printf("FATAL  : last marker in block definition file should be Mtot = %d whereas is %d\n", Mtot, blocksEnds[numBlocks-1]+1);
                printf("        => Adjust block definition file\n");
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Assign to MrankS and MrankL to catch up on logic
        for (int i=0; i<numBlocks; ++i) {
            MrankS[i] = blocksStarts[i] - 1;                  // starts from 0, not 1
            MrankL[i] = blocksEnds[i] - blocksStarts[i] + 1;  // compute length
        }

    } else {
        //if (rank == 0)
        //    printf("INFO   : no marker block definition file used. Will go for even distribution over tasks.\n");
        define_blocks_of_markers(Mtot, MrankS, MrankL, nranks);
    }

    lmax = 0, lmin = 1E9;
    for (int i=0; i<nranks; ++i) {
        if (MrankL[i]>lmax) lmax = MrankL[i];
        if (MrankL[i]<lmin) lmin = MrankL[i];
    }
}
