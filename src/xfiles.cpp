#include "xfiles.h"


// CSV out file
void write_ofile_csv(const MPI_File fh, const uint iteration, const VectorXd sigmaG, const double sigmaE, const VectorXi m0,
                     const uint n_thinned_saved, const MatrixXd estPi) {
    
    MPI_Status status;
    
    char buff[LENBUF];

    int cx = snprintf(buff, LENBUF, "%5d, %4d", iteration, (int) sigmaG.size());
    assert(cx >= 0 && cx < LENBUF);
        
    for(int jj = 0; jj < sigmaG.size(); ++jj){
        cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f", sigmaG(jj));
        assert(cx >= 0 && cx < LENBUF - strlen(buff));
    }
        
    cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f, %20.15f, %7d, %4d, %2d",  sigmaE, sigmaG.sum()/(sigmaE+sigmaG.sum()), m0.sum(), int(estPi.rows()), int(estPi.cols()));
    assert(cx >= 0 && cx < LENBUF - strlen(buff));
        
    for (int ii=0; ii<estPi.rows(); ++ii) {
        for(int kk = 0; kk < estPi.cols(); ++kk) {
            cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), ", %20.15f", estPi(ii,kk));
            assert(cx >= 0 && cx < LENBUF - strlen(buff));
        }
    }
        
    cx = snprintf(&buff[strlen(buff)], LENBUF - strlen(buff), "\n");
    assert(cx >= 0 && cx < LENBUF - strlen(buff));
        
    MPI_Offset offset = size_t(n_thinned_saved) * strlen(buff);
    check_mpi(MPI_File_write_at(fh, offset, &buff, strlen(buff), MPI_CHAR, &status), __LINE__, __FILE__);
}


// BIN out file
// Same info as in .csv but in binary format rather than text (for full precision)
// Consider DDT
void write_ofile_out(const MPI_File fh, const uint iteration, const VectorXd sigmaG, const double sigmaE, const VectorXi m0,
                     const uint n_thinned_saved, const MatrixXd estPi) {
    
    MPI_Status status;
    MPI_Offset offset = size_t(n_thinned_saved) * ((size_t)1                                  * sizeof(uint)   +
                                                   (size_t)(1 + 3)                            * sizeof(int)    +
                                                   (size_t)(sigmaG.size() + 2 + estPi.size()) * sizeof(double));

    check_mpi(MPI_File_write_at(fh, offset, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    offset += sizeof(uint);

    int sigg_size = sigmaG.size();
    check_mpi(MPI_File_write_at(fh, offset, &sigg_size, 1, MPI_INTEGER, &status), __LINE__, __FILE__);
    offset += sizeof(int);

    check_mpi(MPI_File_write_at(fh, offset, sigmaG.data(), sigg_size, MPI_DOUBLE, &status), __LINE__, __FILE__);
    offset += sigg_size * sizeof(double);

    double dtmp[2] = {sigmaE, sigmaG.sum()/(sigmaE+sigmaG.sum())};
    check_mpi(MPI_File_write_at(fh, offset, dtmp, 2, MPI_DOUBLE, &status), __LINE__, __FILE__);
    offset += 2 * sizeof(double);

    int itmp[3] = {m0.sum(), int(estPi.rows()), int(estPi.cols())};
    check_mpi(MPI_File_write_at(fh, offset, itmp, 3, MPI_INTEGER, &status), __LINE__, __FILE__);
    offset += 3 * sizeof(int);
    
    check_mpi(MPI_File_write_at(fh, offset, estPi.data(), estPi.size(), MPI_DOUBLE, &status), __LINE__, __FILE__);
}
