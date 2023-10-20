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
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/lu.hpp>
#include <boost/numeric/ublas/io.hpp>


// this C++ code implements Bayesian linear regression with Huber loss as loss function via 
// Vector Approximate Message Passing Framework


std::vector<double> vamp::infere_robust( data* dataset ){

    double total_time = 0;
    double tol = 1e-11;

    std::vector<double> x1_hat_d(M, 0.0);
    std::vector<double> x1_hat_d_prev(M, 0.0);
    std::vector<double> r1_prev(M, 0.0);
    std::vector<double> p1_prev(N, 0.0);
    std::vector<double> x1_hat_prev(M, 0.0);

    // tau1 = gam1; // hardcoding initial variability
    tau1 = gam1;
    double gam1_prev = gam1;
    double tau1_prev = tau1;

    double sqrtN = sqrt(N);
    // loading the true signal
    std::vector<double> true_signal_s = true_signal;
    for (int i=0; i<true_signal_s.size(); i++)
        true_signal_s[i] = true_signal[i] * sqrtN;
    std::vector<double> true_g = (*dataset).Ax(true_signal_s.data());

    r1 = std::vector<double> (M, 0.0);
    p1 = std::vector<double> (N, 0.0);
    r2 = r1;
    alpha1 = 0;

    // initializing z1_hat and p2
    z1_hat = std::vector<double> (N, 0.0);
    p2 = std::vector<double> (N, 0.0);

    // initializing Huber loss parameter
    deltaH = 1e-3;


    // VAMP iterations
    for (int it = 1; it <= max_iter; it++)
    {    

        double rho_it;
        if (it > 2)
                rho_it = rho;
            else
                rho_it = 1;

        rho_it = 1;

        double start_denoising = 0, stop_denoising = 0;

        if (rank == 0)
            std::cout << std::endl << "********************" << std::endl << "iteration = "<< it << std::endl << "********************" << std::endl;

        //************************
        //************************
        //      DENOISING X
        //************************
        //************************

        start_denoising = MPI_Wtime();

        if (rank == 0)
                std::cout << "->DENOISING" << std::endl;

        x1_hat_prev = x1_hat;
        double alpha1_prev = alpha1;
        double gam1_reEst_prev;
        int it_revar = 1;
        auto_var_max_iter = 50;

        for (; it_revar <= auto_var_max_iter; it_revar++){

            // new signal estimate
            for (int i = 0; i < M; i++)
                x1_hat[i] = g1(r1[i], gam1);

            std::vector<double> x1_hat_m_r1 = x1_hat;
            for (int i0 = 0; i0 < x1_hat_m_r1.size(); i0++)
                x1_hat_m_r1[i0] = x1_hat_m_r1[i0] - r1[i0];

            // new MMSE estimate
            double sum_d = 0;
            for (int i=0; i<M; i++)
            {
                x1_hat_d[i] = g1d(r1[i], gam1);
                sum_d += x1_hat_d[i];
            }

            alpha1 = 0;
            MPI_Allreduce(&sum_d, &alpha1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            alpha1 /= Mt;
            eta1 = gam1 / alpha1;

            if (it <= 1)
                break;

            gam1_reEst_prev = gam1;
            gam1 = std::min( std::max(  1 / (1/eta1 + l2_norm2(x1_hat_m_r1, 1)/Mt), gamma_min ), gamma_max );

            if (rank == 0 && it_revar % 1 == 0)
                std::cout << "it_revar = " << it_revar << ": gam1 = " << gam1 << std::endl;

            updatePrior(0);

            if ( abs(gam1 - gam1_reEst_prev) < 1e-3 )
                break;
            
        }

        if (it > 1){ // damping on the level of x1
            for (int i = 0; i < M; i++)
                x1_hat[i] = rho * x1_hat[i] + (1-rho) * x1_hat_prev[i];
            
            alpha1 = rho * alpha1 + (1-rho) * alpha1_prev;
        }

        
        // saving x1_hat
        double start_saving = MPI_Wtime();
        std::vector<double> x1_hat_stored = x1_hat;
        double scale = 1.0 / sqrt(N);
        std::string filepath_out = out_dir + out_name + "_robust_it_" + std::to_string(it) + ".bin";
        int S = (*dataset).get_S();
        for (int i0=0; i0<x1_hat_stored.size(); i0++)
            x1_hat_stored[i0] =  x1_hat[i0] * scale;
        mpi_store_vec_to_file(filepath_out, x1_hat_stored, S, M);

        if (rank == 0)
        std::cout << "filepath_out is " << filepath_out << std::endl;

        std::string filepath_out_r1 = out_dir + out_name + "_robust_r1_it_" + std::to_string(it) + ".bin";
        std::vector<double> r1_stored = r1;
        for (int i0=0; i0<r1_stored.size(); i0++)
            r1_stored[i0] =  r1[i0] * scale;
        mpi_store_vec_to_file(filepath_out_r1, r1_stored, S, M);
        double end_saving = MPI_Wtime();

        if (rank == 0)
            std::cout << "time needed to save beta1 to an external file = " << end_saving - start_saving << " seconds" <<  std::endl;


        // printing out error measures
        err_measures(dataset, 1);

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
            std::cout << "alpha1 = " << alpha1 << std::endl;

        // true precision calculation
        std::vector<double> x1hatmtrue = x1_hat;
        for (int i=0; i<x1hatmtrue.size(); i++)
            x1hatmtrue[i] -= true_signal_s[i];
        double x1hatmtrue_l2norm2 = l2_norm2(x1hatmtrue,1) / Mt;
        if (rank == 0)
            std::cout << "true eta1 = " << 1.0 / x1hatmtrue_l2norm2 << std::endl;

        gam_before = gam2;

        gam2 = std::min( std::max( eta1 - gam1, gamma_min ), gamma_max );

        if (rank == 0){
            std::cout << "eta1 = " << eta1 << std::endl;
            std::cout << "gam2 = " << gam2 << std::endl;
        }

        std::vector<double> r2_prev = r2;
        for (int i = 0; i < M; i++)
            r2[i] = (eta1 * x1_hat[i] - gam1 * r1[i]) / gam2;

        // true precision calcultion
        std::vector<double> r2mtrue = r2;
        for (int i=0; i<r2mtrue.size(); i++)
            r2mtrue[i] -= true_signal_s[i];

        double r2mtrue_std = calc_stdev(r2mtrue, 1);
        if (rank == 0)
            std::cout << "true gam2 = " << 1.0 / r2mtrue_std / r2mtrue_std << std::endl;

        // updating parameters of prior distribution
        //probs_before = probs;
        //vars_before = vars;
        //updatePrior(1);

        if (rank == 0)
            std::cout << "_____________________________________________" << std::endl;

        //************************
        //************************
        //      DENOISING Z
        //************************
        //************************
    
        std::vector<double> y = (*dataset).filter_pheno();

        std::vector<double> z1_hat_prev = z1_hat;


        double beta1;
        // new signal estimate
        for (int i=0; i<N; i++){    

            z1_hat[i] = g1_Huber(p1[i], tau1, deltaH, y[i]);
        }

        std::vector<double> z1_hat_m_y = z1_hat;
        for (int i0 = 0; i0 < N; i0++)
            z1_hat_m_y[i0] = z1_hat_m_y[i0] - y[i0];

        double z1_hat_m_y_std = calc_stdev(z1_hat_m_y, 0);
        if (rank == 0)
            std::cout << "R2 = " << 1 - z1_hat_m_y_std * z1_hat_m_y_std / calc_stdev(y,0) / calc_stdev(y,0) << std::endl; 

        std::vector<double> z1_hat_m_p1 = z1_hat;
        for (int i0 = 0; i0 < N; i0++)
            z1_hat_m_p1[i0] = z1_hat_m_p1[i0] - p1[i0];

        // new MMSE estimate
        beta1 = 0;

        for (int i=0; i<N; i++){

            beta1 += g1d_Huber_der(p1[i], tau1, deltaH, y[i]);        
        }

        beta1 /= N;

        if (rank == 0)
            std::cout << "beta1 = " << beta1 << std::endl;

        double zeta1 = tau1 / beta1;

        if (it >= 2)
            tau1 = std::min( std::max(  1 / (1/zeta1 + l2_norm2(z1_hat_m_p1, 0)/N), gamma_min ), gamma_max );
            
        std::vector<double> grid = { 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3 };
        deltaH = EM_deltaH(p1, tau1, deltaH, y, 100, grid, 100);

        if (rank == 0)
            std::cout << "deltaH = " << deltaH << std::endl;

        //***********************


        // true precision calculation
        std::vector<double> z1hatmtrue = z1_hat;
        for (int i=0; i<z1hatmtrue.size(); i++)
            z1hatmtrue[i] -= true_g[i];
        double z1hatmtrue_l2norm2 = l2_norm2(z1hatmtrue,0) / N;
        if (rank == 0)
            std::cout << "true eta1_z = " << 1.0 / z1hatmtrue_l2norm2 << std::endl;
        

        for (int i=0; i<N; i++)
            p2[i] = (z1_hat[i] - beta1*p1[i]) / (1-beta1);

        std::vector<double> p2mtrue = p2;
        for (int i=0; i<p2mtrue.size(); i++)
            p2mtrue[i] -= true_g[i];
        double p2mtrue_std = calc_stdev(p2mtrue, 0);
        if (rank == 0)
            std::cout << "true tau2 = " << 1.0 / p2mtrue_std / p2mtrue_std << std::endl;

        tau2 = std::min( std::max( tau1 * (1-beta1) / beta1, gamma_min ), gamma_max );

        if (rank == 0)
            std::cout << "tau2 = " << tau2 << std::endl;

        stop_denoising = MPI_Wtime();


        //************************
        //************************
        // LMMSE estimation of x
        //************************
        //************************

        double start_LMMSE = MPI_Wtime();

        if (rank == 0)
                std::cout << std::endl << "->LMMMSE" << std::endl;

        std::vector<double> v = (*dataset).ATx(p2.data());

        for (int i = 0; i < M; i++)
            v[i] = tau2 * v[i] + gam2 * r2[i];

        // running conjugate gradient solver to compute LMMSE
        x2_hat = precondCG_solver(v, std::vector<double> (M, 0.0), tau2, 1, dataset); // precond_change!

        std::vector<double> x2_hat_s = x2_hat;
        for (int i=0; i<x2_hat_s.size(); i++)
            x2_hat_s[i] = x2_hat[i] / sqrt(N);

        // printing out error measures
        err_measures(dataset, 2);

        double alpha2 = g2d_onsager(gam2, tau2, dataset);
        if (rank == 0)
            std::cout << "alpha2 = " << alpha2 << std::endl;

        eta2 = gam2 / alpha2;

        // re-estimation of gam2
        std::vector<double> x2_hat_m_r2 = x2_hat;
        for (int i0 = 0; i0 < x2_hat_m_r2.size(); i0++)
            x2_hat_m_r2[i0] = x2_hat_m_r2[i0] - r2[i0];

        if (it > 1)
            gam2 = std::min( std::max(  1 / (1/eta2 + l2_norm2(x2_hat_m_r2, 1)/Mt), gamma_min ), gamma_max );

        if (rank == 0)
            std::cout << "gam2 after reest = " << gam2 << std::endl;

        for (int i=0; i<M; i++)
            r1[i] = (x2_hat[i] - alpha2*r2[i]) / (1-alpha2);

        //r1_prev = r1;
        //for (int i=0; i<M; i++)
        //    r1[i] = rho_it * (x2_hat[i] - alpha2*r2[i]) / (1-alpha2) + (1-rho_it) * r1_prev[i];

        // true precision calculation
        std::vector<double> r1mtrue = r1;
        for (int i=0; i<r1mtrue.size(); i++)
            r1mtrue[i] -= true_signal_s[i];
        double r1mtrue_std = calc_stdev(r1mtrue, 1);
        if (rank == 0)
            std::cout << "true gam1 = " << 1.0 / r1mtrue_std / r1mtrue_std << std::endl;


        gam1_prev = gam1;
        gam1 = gam2 * (1-alpha2) / alpha2;

        if (rank == 0)
            std::cout << "gam1 = " << gam1 << std::endl;



        
        //************************
        //************************
        // LMMSE estimation of z
        //************************
        //************************
        
        z2_hat = (*dataset).Ax(x2_hat.data());
        
        double beta2 = (double) N / Mt * (1-alpha2);
        beta2 = (double) Mt / N * (1 - alpha2);

        if (rank == 0)
            std::cout << "beta2 = " << beta2 << std::endl;

        // re-estimation of tau2
        std::vector<double> z2_hat_m_p2 = z2_hat;
        for (int i0 = 0; i0 < N; i0++)
            z2_hat_m_p2[i0] = z2_hat_m_p2[i0] - p2[i0];

        double zeta2 = tau2 / beta2;

        if (it > 1)
            tau2 = 1.0 / (1.0/zeta2 + l2_norm2(z2_hat_m_p2, 0)/N);

        if (rank == 0)
            std::cout << "tau2 after reest = " << tau2 << std::endl;


        for (int i=0; i<N; i++)
            p1[i] = (z2_hat[i] - beta2 * p2[i]) / (1-beta2);

        //p1_prev = p1;
        //for (int i=0; i<N; i++)
        //    p1[i] = rho_it * (z2_hat[i] - beta2 * p2[i]) / (1-beta2) + (1-rho_it) * p1_prev[i];

        
        // true precision calculation
        std::vector<double> p1mtrue = p1;
        for (int i=0; i<p1mtrue.size(); i++)
            p1mtrue[i] -= true_g[i];
        double p1mtrue_std = calc_stdev(p1mtrue, 0);
        if (rank == 0)
            std::cout << "true tau1 = " << 1.0 / p1mtrue_std / p1mtrue_std << std::endl;

        tau1_prev = tau1;
        tau1 = std::min( std::max( tau2 * (1 - beta2) / beta2, gamma_min ), gamma_max );


        if (rank == 0)
            std::cout << "tau1 = " << tau1 << std::endl;

        // stopping criteria
        std::vector<double> x1_hat_diff(M, 0.0);

        for (int i0 = 0; i0 < x1_hat_diff.size(); i0++)
            x1_hat_diff[i0] = x1_hat_prev[i0] - x1_hat[i0];
            
        double rel_err_x1 = sqrt( l2_norm2(x1_hat_diff, 1) / l2_norm2(x1_hat_prev, 1) );

        MPI_Barrier(MPI_COMM_WORLD);

        double stop_LMMSE = MPI_Wtime();

        total_time += (stop_LMMSE - start_LMMSE) + (stop_denoising - start_denoising);

        if (rank == 0)
            std::cout << "total time so far = " << total_time << " seconds." << std::endl;
        
        if (it > 1 && rel_err_x1 < stop_criteria_thr){
            if (rank == 0)
                std::cout << "robustVAMP stopping criteria fulfilled with threshold = " << stop_criteria_thr << "." << std::endl;
            break;
        }

    }
    
    return x1_hat;

}

double vamp::g1_Huber(double p1, double tau1, double deltaH, double y){

    // from Bradic, Chen: Robustness in sparse linear models: relative efficiency based on robust approximate message passing, Ex. 2.
    // b = 1/tau1, z = y - p1, deltaH = Huber loss regularization parameter

    double var = 1.0/tau1;
    double thr = (1+var) * deltaH;
    double w = y - p1;
    double est;

    if (abs(w) <= thr)
        est = w/(1+var);
    else if (w > thr)
        est = w - var * deltaH;
    else if (w < -thr)
        est = w + var * deltaH;
    return (y-est);

}

double vamp::g1d_Huber(double p1, double tau1, double deltaH, double y){

    // from Bradic, Chen: Robustness in sparse linear models: relative efficiency based on robust approximate message passing, Ex. 2.
    // b = 1/tau1, z = p1, deltaH = Huber loss regularization parameter

    double var = 1.0/tau1;
    double thr = (1+var) * deltaH;
    double w = y - p1;
    double est = g1_Huber(p1, tau1, deltaH, y);
    double der;

    if (abs(p1) <= thr)
        der = -var * w/(1+var);
    else if (w > thr)
        der = -var * deltaH;
    else if (w < -thr)
        der = var * deltaH;
    return der;

}


double vamp::g1d_Huber_der(double p1, double tau1, double deltaH, double y){

    // from Bradic, Chen: Robustness in sparse linear models: relative efficiency based on robust approximate message passing, Ex. 2.
    // b = 1/tau1, z = p1, deltaH = Huber loss regularization parameter

    double var = 1.0/tau1;
    double thr = (1+var) * deltaH;
    double w = y - p1;
    double der;

    if (abs(p1) <= thr)
        der = 1.0/(1+var);
    else if (w > thr)
        der = 1;
    else if (w < -thr)
        der = -1;
    return der;

}

double vamp::Huber_loss(double z, double deltaH, double y){

    // evaluating a Huber loss function
    double w = y - z;
    double out;

    if (abs(w) <= deltaH)
        out = w*w/2;
    else if (w > deltaH)
        out = deltaH * (w - deltaH/2);
    else if (w < - deltaH)
        out = deltaH * (-w - deltaH/2);

    return out;
}


double vamp::E_MC_eval_ind(double p1, double tau1, double deltaH, double y, int num_MC_steps){

    // a function evaluation for an individual in E-step of EM algorithm

    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());      

    std::normal_distribution<double> gauss_distr( p1, sqrt( 1.0/tau1 ) ); //2nd parameter is stddev

    double est = 0;

    for (int it=0; it<num_MC_steps; it++){
        double z = gauss_distr(generator);
        est += Huber_loss(z, deltaH, y);
    }
    
    est /= num_MC_steps;
    return est;
}


double vamp::E_MC_eval(std::vector<double> p1, double tau1, double deltaH, std::vector<double> y, int num_MC_steps){

    // a function evaluation in E-step of EM algorithm
    double out = 0;
    for (int i=0; i<N; i++)
        out += E_MC_eval_ind(p1[i], tau1, deltaH, y[i], num_MC_steps);
    return out / N;

}


double vamp::M_deltaH_update(std::vector<double> p1, double tau1, double deltaH, std::vector<double> y, int num_MC_steps, std::vector<double> grid){

    // a function evaluation in E-step of EM algorithm

    int grid_len = grid.size();
    int min_ind = 0;
    double min_value = std::numeric_limits<double>::max();

    for (int i=0; i<grid_len; i++){
        double deltaH_val = grid[i];
        double eval = E_MC_eval(p1, tau1, deltaH_val, y, num_MC_steps);
        if ( eval < min_value ){
            min_ind = i;
            min_value = eval;
        }
    }

    return grid[min_ind];
    
}


double vamp::EM_deltaH(std::vector<double> p1, double tau1, double deltaH, std::vector<double> y, int num_MC_steps, std::vector<double> grid, int num_EM_steps){

    // a function evaluation in E-step of EM algorithm
    for (int i=0; i<num_EM_steps; i++){
        double deltaH_prev = deltaH;
        deltaH = M_deltaH_update(p1, tau1, deltaH, y, num_MC_steps, grid);
        if (abs(deltaH_prev - deltaH) / deltaH < 1e-3)
            break;
    }   
    return deltaH;    
}


void vamp::robust_err_measures(data *dataset, int sync, std::vector<double> true_signal, std::vector<double> est, std::string var_name){
    
    // correlation
    double corr = inner_prod(est, true_signal, sync) / sqrt( l2_norm2(est, sync) * l2_norm2(true_signal, sync) );

    if ( rank == 0 )
        std::cout << "correlation " + var_name + " = " << corr << std::endl;  

    
    // l2 signal error
    int len = (int) ( N + (M-N) * sync );
    std::vector<double> temp(len, 0.0);

    for (int i = 0; i<len; i++)
        temp[i] = est[i] - true_signal[i];

    double l2_signal_err = sqrt( l2_norm2(temp, sync) / l2_norm2(true_signal, sync) );
    if (rank == 0)
        std::cout << "l2 signal error for " + var_name + " = " << l2_signal_err << std::endl;


    // prior distribution parameters
    if (sync == 1){

        if (rank == 0)
        std::cout << "prior variances = ";

        for (int i = 0; i < vars.size(); i++)
            if (rank == 0)
                std::cout << vars[i] << ' ';

        if (rank == 0) {
            std::cout << std::endl;
            std::cout << "prior probabilities = ";
        }

        for (int i = 0; i < probs.size(); i++)
            if (rank == 0)
                std::cout << probs[i] << ' ';

        if (rank == 0)
            std::cout << std::endl;
    }
    
}