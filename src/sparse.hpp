#ifndef HYDRA_SPARSE_H
#define HYDRA_SPARSE_H

#include <cstdlib>

void sparse_set(double*       __restrict__ vec,
                const double               val,
                const uint*   __restrict__ IX, const size_t NXS, const size_t NXL);


void sparse_add(double*       __restrict__ vec,
                const double               val,
                const uint*   __restrict__ IX, const size_t NXS, const size_t NXL);


double sparse_partial_sum(const double* __restrict__ vec,
                          const uint*   __restrict__ IX,
                          const size_t               NXS,
                          const size_t               NXL);
double sparse_partial_sum(const long double* __restrict__ vec,
                          const uint*        __restrict__ IX,
                          const size_t                    NXS,
                          const size_t                    NXL);


double partial_sparse_dotprod(const double* __restrict__ vec,
                              const uint*   __restrict__ IX,
                              const size_t               NXS,
                              const size_t               NXL,
                              const double               fac);


void  sparse_scaadd(double*     __restrict__ vout,
                    const double  dMULT,
                    const uint* __restrict__ I1, const size_t N1S, const size_t N1L,
                    const uint* __restrict__ I2, const size_t N2S, const size_t N2L,
                    const uint* __restrict__ IM, const size_t NMS, const size_t NML,
                    const double  mu,
                    const double  sig_inv,
                    const int     N);


double sparse_dotprod(const double* __restrict__ vin1,
                      const uint*   __restrict__ I1,      const size_t N1S,  const size_t N1L,
                      const uint*   __restrict__ I2,      const size_t N2S,  const size_t N2L,
                      const uint*   __restrict__ IM,      const size_t NMS,  const size_t NML,
                      const double               mu, 
                      const double               sig_inv,
                      const int                  N,
                      const int                  marker);



#endif //#define HYDRA_SPARSE_H
