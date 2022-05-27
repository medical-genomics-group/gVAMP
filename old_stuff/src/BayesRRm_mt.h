#ifndef SRC_BAYESRRM_MT_H_
#define SRC_BAYESRRM_MT_H_

#include "BayesRRm.h"
#include "data.hpp"
#include "options.hpp"
#include "distributions_boost.hpp"

#include <Eigen/Eigen>

class BayesRRm_mt : public BayesRRm
{

public:

 BayesRRm_mt(Data &data, Options &opt, const long memPageSize)
     : BayesRRm(data, opt, memPageSize) 
        {
        }
    virtual ~BayesRRm_mt();
    
    void set_mt_vector_f64(double* __restrict__ vec,
                           const double*        val,
                           const int            NT,
                           const int            N,
                           const bool           interleave);
    
    void sum_mt_vector_elements_f64(const double* __restrict__ vec,
                                    const int                  NT,
                                    const int                  N,
                                    const bool                 interleave,
                                    double*       __restrict__ syt8);
    
    void sparse_dotprod_mt(const double* __restrict__ vin1, const uint8_t* __restrict__ mask,
                           const uint*   __restrict__ I1,   const size_t __restrict__ N1S,  const size_t __restrict__ N1L,
                           const uint*   __restrict__ I2,   const size_t __restrict__ N2S,  const size_t __restrict__ N2L,
                           const uint*   __restrict__ IM,   const size_t NMS,  const size_t NML,
                           const double  mu, const double sig_inv, const int Ntot, const int marker,
                           double* __restrict__ m8, const int NT, const bool interleave);
    
    void sparse_scaadd_mt(double*       __restrict__ vout,
                          const double* __restrict__ dMULT,
                          const uint*   __restrict__ I1,    const size_t N1S, const size_t N1L,
                          const uint*   __restrict__ I2,    const size_t N2S, const size_t N2L,
                          const uint*   __restrict__ IM,    const size_t NMS, const size_t NML,
                          const double  mu,                 const double sig_inv,
                          const int     Ntot,               const int NT,     const bool interleave);
    
    int    runMpiGibbsMultiTraits();
};

#endif /* SRC_BAYESRRM_MT_H_ */
