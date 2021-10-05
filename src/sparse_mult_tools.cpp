#include "sparse_mult_tools.hpp"

VectorXd xbeta_mult(VectorXd beta, int Ntot, int M, size_t *N1S, size_t *N1L, size_t *N2S, size_t *N2L, double *mstd, double *mave, uint *I1, uint *I2, double scale)
{
    VectorXd xbeta = VectorXd::Zero(Ntot);
    double tmp_sum = 0;
    //double beta_sum = sum_array_elements(beta.data(), M);
    for (int j=0; j < M ; j++)
    {
        tmp_sum -= mave[j]*beta(j)/mstd[j];
    }
    for (int i=0; i<Ntot; i++)
    {
        xbeta(i) = tmp_sum;
    }
    for (int j=0; j < M ; j++)
    {
        double cor_term = beta(j)/mstd[j];
        //cout << "cor_term: "<< cor_term << endl;
        for (size_t ii = N1S[j]; ii < (N1S[j] + N1L[j]); ii++)
        {
            xbeta(I1[ii]) = xbeta(I1[ii]) + cor_term;
        }
        cor_term = 2*cor_term;
        for (size_t ii = N2S[j]; ii < (N2S[j] + N2L[j]); ii++)
        {
            xbeta(I2[ii]) = xbeta(I2[ii]) + cor_term;
        }        
    }
    xbeta = xbeta / scale;
    return xbeta;
}

VectorXd xtr_mult(VectorXd r, int Ntot, int M, size_t *N1S, size_t *N1L, size_t *N2S, size_t *N2L, double *mstd, double *mave, uint *I1, uint *I2, double scale)
{   
    VectorXd c_beta = VectorXd::Zero(M);
    for (int j = 0; j < M; j++)
    {
        double *r_pointer = r.data();
        double r_sum = sum_array_elements(r_pointer, Ntot);
        double r_sum_2 = sparse_partial_sum(r_pointer, I2, N2S[j], N2L[j]);
        double r_sum_1 = sparse_partial_sum(r_pointer, I1, N1S[j], N1L[j]);
        c_beta(j) = r_sum_1/mstd[j] + 2*r_sum_2/mstd[j] -mave[j]/mstd[j]*r_sum;
    }
    return c_beta/ scale;
}