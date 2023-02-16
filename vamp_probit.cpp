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
            probit_var = update_probit_var(probit_var, y);
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
        double start_CG = MPI_Wtime();
        x2_hat = precondCG_solver(v, tau2, 1, dataset); // precond_change!
        double end_CG = MPI_Wtime();
        
        if (rank == 0)
            std::cout << "CG took "  << end_CG - start_CG << " seconds." << std::endl;

        double alpha2 = g2d_onsager(gam2, tau2, dataset);

        for (int i=0; i<M; i++)
            r1[i] = (x2_hat[i] - alpha2*r2[i]) / (1-alpha2);

        gam1 = gam2 * (1-alpha2) / alpha2;


        

        //************************
        // LMMSE estimation of x
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
    double temp = p + (2*y-1) * exp(-0.5 * c * c) / sqrt(2*M_PI) / tau1 / sqrt(probit_var + 1/tau1) / normalCDF((2*y - 1)*c);

    return temp;

}


double vamp::g1d_bin_class(double p, double tau1, double y){
    
    double c = p / sqrt(probit_var + 1/tau1);

    double Nc = exp(-0.5 * c * c) /  sqrt(2*M_PI);

    double phic = normalCDF((2*y-1)*c);

    double temp = 1 -  Nc / (1 + tau1 * probit_var * probit_var) / phic * ((2*y-1)*c + c / phic); // because der = tau * Var

    return temp;

}

double vamp::probit_var_EM_deriv(double v, std::vector<double> y){ // we'll do the update just before z1

    double sqrt_v = sqrt(v);
    double new_v;
    for (int i=0; i<N; i++){
        double c = z1_hat[i] / sqrt_v;
        double phic = 0.5 * erfc(-c * M_SQRT1_2);
        if (y[i] == 1)
            new_v += exp(-c*c/2) / sqrt(2*M_PI) / phic * z1_hat[i];
        else if (y[i] == 0)
            new_v += - exp(-c*c/2) / sqrt(2*M_PI) / (1-phic) * z1_hat[i];  
    }
    double new_v_MPIsync;
    MPI_Allreduce(&new_v, &new_v_MPIsync, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return new_v_MPIsync / Mt;
}

double vamp::update_probit_var(double v, std::vector<double> y){

    double new_v=v, var_min=1/tau2, var_max=1e8; //z ~ N(p2, I/tau2)

    // bisection method for finding the root of 'probit_var_EM_deriv' function
    if (probit_var_EM_deriv(var_min,y)<0){
        // solution must be <= var_min
        new_v = var_min;
    }
    else if (probit_var_EM_deriv(var_max,y)>0){
        // solution must be >= var_max
        new_v = var_max;
    }
    else{
        // refine midpoint on a log scale
        new_v = exp(0.5*(log(var_min) + log(var_max)));
        for (int it=0; it<20; it++){   
            if (probit_var_EM_deriv(new_v,y)>0)
                var_min = new_v;
            else
                var_max = new_v;
        new_v = exp(0.5*(log(var_min) + log(var_max)));
        }
    }

    return new_v;
}