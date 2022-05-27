#include <cstring>
#include <iostream>
#include "xfiles.hpp"

// CSV out file
void write_ofile_csv(const MPI_File fh, const uint iteration, const std::vector<double>* sigmaG, const double sigmaE, const int m0_sum,
                     const uint n_thinned_saved, const std::vector<std::vector<double>>* estPi) {
    
    int R = estPi->size();
    int C = estPi->at(0).size();
    //printf("R x C = %d x %d\n", R, C);

    MPI_Status status;
    
    char buff[LENBUF];

    int cx = snprintf(buff, LENBUF, "%5d, %4d", iteration, (int) sigmaG->size());
    assert(cx >= 0 && cx < LENBUF);
        
    for(int i=0; i<sigmaG->size(); i++){
        cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f", sigmaG->at(i));
        assert(cx >= 0 && cx < LENBUF - strlen(buff));
    }

    assert(sigmaG->size() == estPi->size());
    double sigmag_sum = 0.0;
    for (int i = 0; i < sigmaG->size(); i++)
        sigmag_sum += sigmaG->at(i);

    cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f, %20.15f, %7d, %4d, %2d",  sigmaE, sigmag_sum / (sigmaE + sigmag_sum), m0_sum, int(estPi->size()), int(estPi->at(0).size()));
    assert(cx >= 0 && cx < LENBUF - strlen(buff));

    for (int i=0; i<R; i++) {
        for(int j=0; j<C; j++) {
            //printf("%d %d %20.15f\n", i, j, estPi->at(i).at(j));
            cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f", estPi->at(i).at(j));
            assert(cx >= 0 && cx < LENBUF - strlen(buff));
        }
    }
        
    
    cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), "\n");
    assert(cx >= 0 && cx < LENBUF - strlen(buff));
        
    MPI_Offset offset = size_t(n_thinned_saved) * strlen(buff);
    check_mpi(MPI_File_write_at(fh, offset, &buff, strlen(buff), MPI_CHAR, &status), __LINE__, __FILE__);
}
