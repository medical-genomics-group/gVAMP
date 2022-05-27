#include "BayesRRm.h"
#include "dense.hpp"
#include "sparse.hpp"

VectorXd xbeta_mult(VectorXd beta, int Ntot, int M, size_t *N1S, size_t *N1L, size_t *N2S, size_t *N2L, double *mstd, double *mave, uint *I1, uint *I2, double scale);

VectorXd xtr_mult(VectorXd r, int Ntot, int M, size_t *N1S, size_t *N1L, size_t *N2S, size_t *N2L, double *mstd, double *mave, uint *I1, uint *I2, double scale);