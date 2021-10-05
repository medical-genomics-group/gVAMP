#include <cstdlib>
#include <stdio.h>
#include <omp.h>
#include "sparse.hpp"
#include "dense.hpp"

void sparse_set(double*       __restrict__ vec,
                const double               val,
                const uint*   __restrict__ IX, const size_t NXS, const size_t NXL) {
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume_aligned(IX,  64);
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i=NXS; i<NXS+NXL; ++i) {
        vec[ IX[i] ] = val;
    }
}


void sparse_add(double*       __restrict__ vec,
                const double               val,
                const uint*   __restrict__ IX, const size_t NXS, const size_t NXL) {
    
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume_aligned(IX,  64);
#endif
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (size_t i=NXS; i<NXS+NXL; ++i) {
        vec[ IX[i] ] += val;
    }
}


// Function that finds the sum across the sum across individuals who have marker with specified value
double sparse_partial_sum(const double* __restrict__ vec,
                          const uint*   __restrict__ IX,
                          const size_t               NXS,
                          const size_t               NXL) {
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume_aligned(IX,  64);
#endif

    double sum = 0.0;

#ifdef _OPENMP

    double partial_sum = 0.0;
    //EO: see https://gcc.gnu.org/gcc-9/porting_to.html#ompdatasharing
    //#pragma omp parallel default(none) firstprivate(partial_sum) shared(sum, vec, IX, NXS, NXL)
#pragma omp parallel default(none) firstprivate(partial_sum, NXS, NXL) shared(sum, vec, IX)
    {
        //int ID = omp_get_thread_num();
        
#pragma omp for schedule(static)
        for (size_t i=NXS; i < NXS + NXL; i++) {
            partial_sum += vec[IX[i]];
        }
        
#pragma omp for ordered schedule(static,1)
        for (int t=0; t<omp_get_num_threads(); ++t) {
            //assert( t==ID );
#pragma omp ordered
            {
                sum += partial_sum;
            }
        }
    }
    
#else

    for (size_t i=NXS; i < NXS + NXL; i++) {
        sum += vec[IX[i]];
    }

#endif

    return sum;
}


// Function that finds the sum across the sum across individuals who have marker with specified value
double sparse_partial_sum(const long double* __restrict__ vec,
                          const uint*        __restrict__ IX,
                          const size_t                    NXS,
                          const size_t                    NXL) {
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume_aligned(IX,  64);
#endif

    long double sum = 0.0;

#ifdef _OPENMP

    long double partial_sum = 0.0;
    
    //#pragma omp parallel default(none) firstprivate(partial_sum) shared(sum, vec, IX, NXS, NXL)
#pragma omp parallel default(none) firstprivate(partial_sum, NXS, NXL) shared(sum, vec, IX)
    {
        //int ID = omp_get_thread_num();
        
#pragma omp for schedule(static)
        for (size_t i=NXS; i < NXS + NXL; i++) {
            partial_sum += vec[IX[i]];
        }
        
#pragma omp for ordered schedule(static,1)
        for (int t=0; t<omp_get_num_threads(); ++t) {
            //assert( t==ID );
#pragma omp ordered
            {
                sum += partial_sum;
            }
        }
    }
    
#else

    for (size_t i=NXS; i < NXS + NXL; i++) {
        sum += vec[IX[i]];
    }

#endif

    return (double) sum;
}


void sparse_scaadd(double*     __restrict__ vout,
                   const double             dMULT,
                   const uint* __restrict__ I1, const size_t N1S, const size_t N1L,
                   const uint* __restrict__ I2, const size_t N2S, const size_t N2L,
                   const uint* __restrict__ IM, const size_t NMS, const size_t NML,
                   const double             mu,
                   const double             sig_inv,
                   const int                N) {
    
    if (dMULT == 0.0) {

        set_array(vout, 0.0, N);

    } else {

        double aux = mu * sig_inv * dMULT;
        //printf("sparse_scaadd aux = %15.10f with mu = %15.10f, dbetsig = %15.10f\n", aux, mu, sig_inv * dMULT);
        set_array(vout, -aux, N);

        //cout << "sparse set on M: " << NMS << ", " << NML << endl;
        sparse_set(vout, 0.0, IM, NMS, NML);

        //cout << "sparse set on 1: " << N1S << ", " << N1L << endl;
        aux = dMULT * (1.0 - mu) * sig_inv;
        //printf("1: aux = %15.10f\n", aux);
        sparse_set(vout, aux, I1, N1S, N1L);

        //cout << "sparse set on 2: " << N2S << ", " << N2L << endl;
        aux = dMULT * (2.0 - mu) * sig_inv;
        sparse_set(vout, aux, I2, N2S, N2L);
    }
}


double partial_sparse_dotprod(const double* __restrict__ vec,
                              const uint*   __restrict__ IX,
                              const size_t               NXS,
                              const size_t               NXL,
                              const double               fac) {
    
    double dp = 0.0;
#ifdef __INTEL_COMPILER
    __assume_aligned(vec, 64);
    __assume_aligned(IX,  64);
#endif
#ifdef _OPENMP
#pragma omp parallel for reduction(+: dp)
#endif
    for (size_t i=NXS; i<NXS+NXL; i++) {
        dp += vec[ IX[i] ];
    }

    dp *= fac;

    //printf("dp = %20.16f\n", dp);

    return dp;
}


double sparse_dotprod(const double* __restrict__ vin1,
                      const uint*   __restrict__ I1,      const size_t N1S,  const size_t N1L,
                      const uint*   __restrict__ I2,      const size_t N2S,  const size_t N2L,
                      const uint*   __restrict__ IM,      const size_t NMS,  const size_t NML,
                      const double               mu,
                      const double               sig_inv,
                      const int                  N,
                      const int                  marker) {
    
    double dp  = 0.0;

    dp += partial_sparse_dotprod(vin1, I1, N1S, N1L, 1.0);

    dp += partial_sparse_dotprod(vin1, I2, N2S, N2L, 2.0);

    double syt = sum_array_elements(vin1, N);

    double dsyt = partial_sparse_dotprod(vin1, IM, NMS, NML, 1.0);

    syt -= dsyt;

    dp  -= (mu * syt);

    dp  *= sig_inv;

    return dp;
}
