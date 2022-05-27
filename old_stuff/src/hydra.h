#ifndef HYDRA_H
#define HYDRA_H

#include <cstddef>
#include <mpi.h>
#include <map>
#include <vector>
#include <string>

#ifdef __GNUG__
/* Old compatibility names for C types.  */
typedef unsigned long int ulong;
typedef unsigned short int ushort;
typedef unsigned int uint;
#endif


const size_t LENBUF = 50000;


using fp_it = std::vector<std::string>::iterator;

typedef std::map<std::string, MPI_File>            fh_map;
typedef std::map<std::string, MPI_File>::iterator  fh_it;

struct sparse_info_t {
    size_t *N1S, *N1L;
    size_t *N2S, *N2L;
    size_t *NMS, *NML;
    uint *I1, *I2, *IM;
};


#endif
