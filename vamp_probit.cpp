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


std::vector<double> vamp::infere_bin_class( data* dataset ){


    double tol = 1e-11;

    std::vector<double> x1_hat_d(M, 0.0);
    std::vector<double> x1_hat_d_prev(M, 0.0);
    std::vector<double> r1_prev(M, 0.0);
    std::vector<double> x1_hat_prev(M, 0.0);
    tau1 = gam1; // hardcoding initial variability

    // Gaussian noise start
    r1 = simulate(M, std::vector<double> {1.0/gam1}, std::vector<double> {1});
    p1 = simulate(N, std::vector<double> {1.0/gam1}, std::vector<double> {1});

    for (int it = 1; it <= max_iter; it++)
    {    

        //************************
        //************************
        //      DENOISING X
        //************************
        //************************

        if (rank == 0)
                std::cout << std::endl << "********************" << std::endl << "iteration = "<< it << std::endl << "********************" << std::endl << "->DENOISING" << std::endl;

        x1_hat_prev = x1_hat;

        for (int i = 0; i < M; i++ )
            x1_hat[i] = rho * g1(r1[i], gam1) + (1 - rho) * x1_hat_prev[i];


        // storing current estimate of the signal
        std::string filepath_out = out_dir + out_name + "_bin_class_" + "_it_" + std::to_string(it) + "_rank_" + std::to_string(rank) + ".txt"; 
        if (rank == 0)
            std::cout << "filepath_out is " << filepath_out << std::endl;

        double scale = sqrt(N);
        int S = (*dataset).get_S();
        std::vector<double> x1_hat_stored = x1_hat;

        for (int i0=0; i0<x1_hat_stored.size(); i0++)
            x1_hat_stored[i0] =  x1_hat[i0] / scale;

        mpi_store_vec_to_file(filepath_out, x1_hat_stored, S, M);


        x1_hat_d_prev = x1_hat_d;
        double sum_d = 0;
        for (int i = 0; i < M; i++)
        {
            // we have to keep the entire derivative vector so that we could have its previous version in the damping step 
            x1_hat_d[i] = rho * g1d(r1[i], gam1) + (1 - rho) * x1_hat_d_prev[i]; 
            sum_d += x1_hat_d[i];
        }

        double alpha1;
        MPI_Allreduce(&sum_d, &alpha1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        alpha1 /= Mt;

        if (rank == 0)
            std::cout << "alpha1 = " << alpha1 << std::endl;

        eta1 = gam1 / alpha1;

        gam_before = gam2;

        gam2 = std::min( std::max( eta1 - gam1, gamma_min ), gamma_max );

        if (rank == 0){
            std::cout << "eta1 = " << eta1 << std::endl;
            std::cout << "gam2 = " << gam2 << std::endl;
        }

        for (int i = 0; i < M; i++)
            r2[i] = (eta1 * x1_hat[i] - gam1 * r1[i]) / gam2;


        // updating parameters of prior distribution
        probs_before = probs;
        vars_before = vars;
        updatePrior();



        //************************
        //************************
        //      DENOISING Z
        //************************
        //************************
                
        std::vector<double> y = (*dataset).filter_pheno();

        // denoising part
        for (int i=0; i<N; i++)
            z1_hat[i] = g1_bin_class(p1[i], tau1, y[i]);


        double beta1 = 0;

        for (int i=0; i<N; i++)
            beta1 += g1d_bin_class(p1[i], tau1, y[i]);

        beta1 /= N;
        
        for (int i=0; i<N; i++)
            p2[i] = (z1_hat[i] - beta1*p1[i]) / (1-beta1);

        tau2 = tau1 * (1-beta1) / beta1;

        // updating probit_var
        if (rank == 0){

            probit_var = update_probit_var(probit_var, tau1 / beta1, z1_hat, y);

            for (int ran = 1; ran < nranks; ran++)
                MPI_Send(&probit_var, 1, MPI_DOUBLE, ran, 0, MPI_COMM_WORLD);

        }
        else
            MPI_Recv(&probit_var, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);




        //************************
        //************************
        // LMMSE estimation of x
        //************************
        //************************

        std::vector<double> v = (*dataset).ATx(p2.data());

        for (int i = 0; i < M; i++)
            v[i] = tau2 * v[i] + gam2 * r2[i];

        // running conjugate gradient solver to compute LMMSE
        x2_hat = precondCG_solver(v, tau2, 1, dataset); // precond_change!


        double alpha2 = g2d_onsager(gam2, tau2, dataset);

        for (int i=0; i<M; i++)
            r1[i] = (x2_hat[i] - alpha2*r2[i]) / (1-alpha2);

        gam1 = gam2 * (1-alpha2) / alpha2;



        
        //************************
        //************************
        // LMMSE estimation of z
        //************************
        //************************
        
        z2_hat = (*dataset).Ax(x2_hat.data());
        
        double beta2 = N / M * (1-alpha2);

        for (int i=0; i<M; i++)
            p1[i] = (z2_hat[i] - beta2 * p2[i]) / (1-beta2);

        tau1 = tau2 * (1 - beta2) / beta2;
       

        // stopping criteria
        std::vector<double> x1_hat_diff(M, 0.0);

        for (int i0 = 0; i0 < x1_hat_diff.size(); i0++)
            x1_hat_diff[i0] = x1_hat_prev[i0] - x1_hat_diff[i0];

    }
    
    return x1_hat;

}


double vamp::g1_bin_class(double p, double tau1, double y){

    double c = p / sqrt(probit_var + 1/tau1);
    double temp = p + (2*y-1) * exp(-0.5 * c * c) / sqrt(2*M_PI) / tau1 / sqrt(probit_var + 1/tau1) / normal_cdf((2*y - 1)*c);

    return temp;

}


double vamp::g1d_bin_class(double p, double tau1, double y){
    
    double c = p / sqrt(probit_var + 1/tau1);

    double Nc = exp(-0.5 * c * c) /  sqrt(2*M_PI);

    double phic = normal_cdf((2*y-1)*c);

    double temp = 1 -  Nc / (1 + tau1 * probit_var) / phic * ((2*y-1)*c + Nc / phic); // because der = tau * Var

    return temp;

}

double vamp::probit_var_EM_deriv(double v, std::vector<double> z, std::vector<double> y){

    double der = 0.0;

    for (int i=0; i<N; i++){

        double c = (2*y[i] - 1) * z[i] / v;
        double phic = normal_cdf(c);

        der += c * exp(-c*c/2) / sqrt(2*M_PI) * z[i] / v / phic;    
    }

    double der_MPIsync;

    MPI_Allreduce(&der, &der_MPIsync, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    return der_MPIsync / N;

}

double vamp::expe_probit_var_EM_deriv(double v, double eta, std::vector<double> z_hat, std::vector<double> y){

    // z ~ N(z_hat, I/eta)

    int miter_probit_expe_der = 1e3;

    int it = 1;

    double sum = 0;

    std::vector<double> z = simulate(M, std::vector<double> {1.0/eta}, std::vector<double> {1});

    std::transform (z.begin(), z.end(), z_hat.begin(), z.begin(), std::plus<double>());

    for (; it < miter_probit_expe_der; it++){

        sum +=  probit_var_EM_deriv(v, z, y);

    }

    return  sum / miter_probit_expe_der;

}

double vamp::update_probit_var(double v, double eta, std::vector<double> z_hat, std::vector<double> y){

    double new_v=v, var_min=1e-10, var_max=1e10; // z ~ N(z_hat, I/eta)

    int max_iter_bisec = 50;

    int it = 1;

    assert(sgn(var_min)==-sgn(var_max));

    for (; it<max_iter_bisec; it++){

        double fv = expe_probit_var_EM_deriv(new_v, eta, z_hat, y);

        if (abs(fv) < 1e-6)
            break;

        if (sgn(fv) == sgn(var_min))
            var_min = v;
        else if (sgn(fv) == sgn(var_max))
            var_max = v;
        
        // refine midpoint on a log scale
        new_v = exp(0.5*(log(var_min) + log(var_max)));

    }

    if (rank == 0)
        std::cout << "bisection method finished after " << it << " / " << max_iter_bisec << "iterations" << std::endl;
    
    return new_v;

}