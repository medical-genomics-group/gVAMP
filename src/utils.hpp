#ifndef HYDRA_UTILS_H_
#define HYDRA_UTILS_H_

#include <vector>
#include <string>
#include <mpi.h>
#include <ctime>
#include <sys/time.h>
#include "hydra.h"
#include "options.hpp"

inline
double mysecond() {
    struct timeval  tp;
    struct timezone tzp;
    int i;
    i = gettimeofday(&tp, &tzp);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.e-6 );
}


inline
void errorCheck(int err){
	if(err>0){
		cout << "Error code = " << err << endl;
		exit(1);
	}
}



void check_malloc(const void* ptr, const int linenumber, const char* filename);


void check_mpi(const int error, const int linenumber, const char* filename);


int check_int_overflow(const size_t n, const int linenumber, const char* filename);


MPI_File get_fh(fh_map file_handlers, std::string fp);


void check_file_size(const MPI_File fh, const size_t N, const size_t DTSIZE, const int linenumber, const char* filename);


size_t get_file_size(const std::string& filename);


void define_blocks_of_markers(const int  Mtot,
                              int*       MrankS,
                              int*       MrankL,
                              const uint nblocks);


void check_whole_array_was_set(const uint*  array,
                               const size_t size,
                               const int    linenumber,
                               const char*  filename);


void assign_blocks_to_tasks(const uint             numBlocks,
                            const std::vector<int> blocksStarts,
                            const std::vector<int> blocksEnds,
                            const uint Mtot,
                            const int  nranks,
                            const int  rank,
                            int* MrankS,
                            int* MrankL,
                            int& lmin,
                            int& lmax);

#endif 
