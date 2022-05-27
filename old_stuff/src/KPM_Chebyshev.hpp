#include "constants.hpp"
#include "BayesRRm.h"
#include "sparse_mult_tools.hpp"

VectorXd KPM(VectorXd points, int num_points, int M_deg, size_t *N1S, size_t *N1L, size_t *N2S, size_t *N2L, double *mstd, double *mave, uint *I1, uint *I2, int Ntot, int M, double scaling);