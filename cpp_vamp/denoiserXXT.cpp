#include <vector>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <cmath>
#include <numeric> // contains std::accumulate
#include <random>
#include <omp.h>
#include <fstream>
#include <cfloat>
#include "na_lut.hpp"
#include "vamp.hpp"
#include "utilities.hpp"

std::vector<double> vamp::lmmse_multAAT(std::vector<double> u, double tau, data* dataset){ // multiplying with (tau*AA^Tu + gam2*u)

    size_t phen_size = 4 * (*dataset).get_mbytes();
    if (u == std::vector<double>(N, 0.0) || u == std::vector<double>(phen_size, 0.0))
        return std::vector<double>(phen_size, 0.0);
    std::vector<double> res(phen_size, 0.0);
    std::vector<double> res_temp(M, 0.0);
    res_temp = (*dataset).ATx(u.data(), normal);
    res = (*dataset).Ax(res_temp.data(), normal);
    for (int i = 0; i < N; i++){
        res[i] *= tau;
        res[i] += gam2 * u[i];
    }
    return res;
}

std::vector<double> vamp::lmmse_denoiserAAT(std::vector<double> r2, std::vector<double> mu_CG_AAT_last, data* dataset){
    double norm_r2 = sqrt(l2_norm2(r2,1));
    if (rank == 0)
        std::cout << "||r2|| = " << norm_r2 << std::endl;
    std::vector<double> z2 = (*dataset).Ax(r2.data(), normal);
    if (rank == 0)
        std::cout << "||z2|| = " << sqrt(l2_norm2(z2,0)) << std::endl;
    if (rank == 0)
        std::cout << "||y|| = " << sqrt(l2_norm2(y,0)) << std::endl;
    std::vector<double> v = std::vector<double> (N, 0.0);
    for (int i=0; i<N; i++)
        v[i] = y[i] - z2[i];
    if (rank == 0)
        std::cout << "||v|| = " << sqrt(l2_norm2(v,0)) << std::endl;
    std::vector<double> u = CG_solverAAT(v, mu_CG_AAT_last, gamw, 1, dataset);
    std::vector<double> res = (*dataset).ATx(u.data(), normal);
    for (int i=0; i<M; i++)
        res[i] = gamw * res[i] + r2[i];
    return res;
}

std::vector<double> vamp::CG_solverAAT(std::vector<double> v, std::vector<double> mu_start, double tau, int save, data* dataset){
    
    int mbytes4 = 4*(*dataset).get_mbytes();
    std::vector<double> mave_people = (*dataset).get_mave_people();
    std::vector<double> msig_people = (*dataset).get_msig_people();
    std::vector<double> numb_people = (*dataset).get_numb_people();

    // constructing preconditioner
    std::vector<double> diag(N, 1.0);
    for (int i=0; i<N; i++){
        diag[i] = tau * ((numb_people[i]-1) / msig_people[i] / msig_people[i] + mave_people[i] * mave_people[i] * numb_people[i]) / N + gam2;
        if (diag[i] == 0 || msig_people[i] < 0){
            std::cout << "diag[i] == 0, i = " << i << ", numb_people[i] = " << numb_people[i] << ", msig_people[i] = " << msig_people[i] << std::endl;
        }
    }

    if (rank == 0){
        std::cout << "diag[0] = " << diag[0] << std::endl;
        std::cout << "diag[1] = " << diag[1] << std::endl;
        std::cout << "diag[2] = " << diag[2] << std::endl;
    }
    
    std::vector<double> mu = mu_start;
    std::vector<double> d; // d = Ap
    std::vector<double> r = lmmse_multAAT(mu, tau, dataset); // r = residual
    for (int i0 = 0; i0 < N; i0++)
        r[i0] = v[i0] - r[i0];
    std::vector<double> z(mbytes4, 0.0);
    // preconditioning step
    for (int i0=0; i0<N; i0++)
        z[i0] = r[i0] / diag[i0];

    std::vector<double> p = z;
    std::vector<double> Apalpha(mbytes4, 0.0);
    std::vector<double> palpha(mbytes4, 0.0);
    double alpha, beta;

    for (int i = 0; i < CG_max_iter; i++){

        d = lmmse_multAAT(p, tau, dataset);

        alpha = inner_prod(r, z, 0) / inner_prod(d, p, 0);
        
        for (int j = 0; j < N; j++)
            palpha[j] = alpha * p[j];

        std::transform (mu.begin(), mu.end(), palpha.begin(), mu.begin(), std::plus<double>());

        for (int j = 0; j < N; j++)
            Apalpha[j] = d[j] * alpha;

        beta = pow( inner_prod(r, z, 0), -1 ); // optimize further!

        std::transform (r.begin(), r.end(), Apalpha.begin(), r.begin(), std::minus<double>());

        for (int j=0; j<N; j++)
            z[j] = r[j] / diag[j];

        double inner_prod_rz = inner_prod(r, z, 0);
        beta *= inner_prod_rz;

        for (int j = 0; j < N; j++)
            p[j] = z[j] + beta * p[j];

        // stopping criteria
        double l2_norm2_r = l2_norm2(r, 0);
        double rel_err = sqrt( l2_norm2_r / l2_norm2(v, 0) );
        double norm_mu = sqrt( l2_norm2(mu, 0) );
        double norm_z = sqrt( l2_norm2(z,0) );
        double err_tol = 1e-4;
        if (rank == 0)
            std::cout << "[CG] it = " << i << ": ||r_it|| / ||RHS|| = " << rel_err << ", ||x_it|| = " << norm_mu << ", ||z|| / ||RHS|| = " << norm_z /  l2_norm2(v, 0) << std::endl;
        if (rel_err < err_tol) 
            break;
    }
    if (save == 1)
        mu_CG_last = mu;
    return mu;
 }

 
double vamp::g2d_onsagerAAT(double gam2, double tau, data* dataset) { // shared between linear and binary classification model
    
    std::random_device rd;
    std::bernoulli_distribution bern(0.5);

    bern_vec = std::vector<double> (M, 0.0);
    for (int i = 0; i < M; i++)
        bern_vec[i] = (2*bern(rd) - 1) / sqrt(Mt); // Bernoulli variables are sampled independently
    std::vector<double> res = (*dataset).Ax(bern_vec.data(), normal);
    invQ_bern_vec = CG_solverAAT(res, std::vector<double> (4*(*dataset).get_mbytes(), 0.0), tau, 0, dataset); // precond_change
    res = (*dataset).ATx(invQ_bern_vec.data(), normal);
    double onsager = inner_prod(bern_vec, res, 1) * gamw; 
    return gam2 * (1 + onsager);    
}


void vamp::updateNoisePrecAAT(data* dataset){

    std::vector<double> temp = (*dataset).Ax(x2_hat.data(), normal);
    // std::vector<double> y = (*dataset).get_phen();

    for (int i = 0; i < N; i++)  // because length(y) = N
        temp[i] -= y[i];
    
    double temp_norm2 = l2_norm2(temp, 0); 
    double trace_corr = Mt * (alpha2-1) / gamw;
    if (rank == 0){
        std::cout << "l2_norm2(temp) / N = " << temp_norm2 / N << std::endl;
        std::cout << "trace_correction / N = " << trace_corr / N << std::endl;
    }
    gamw = (double) N / (temp_norm2 + trace_corr);
     
}