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
    // tau1 = gam1; // hardcoding initial variability
    //tau1 = 1e-1; // ideally sqrt( tau_10 / v ) approx 1, since it is to be composed with a gaussian CDF
    tau1 = gam1;
    double tau2_prev = 0;

    double sqrtN = sqrt(N);
    std::vector<double> true_signal_s = true_signal;
    for (int i=0; i<true_signal_s.size(); i++)
        true_signal_s[i] = true_signal[i] * sqrtN;
    std::vector<double> true_g = (*dataset).Ax(true_signal_s.data());


    // Gaussian noise start
    //r1 = simulate(M, std::vector<double> {1.0/gam1}, std::vector<double> {1});
    //p1 = simulate(N, std::vector<double> {1.0/tau1}, std::vector<double> {1});

    r1 = std::vector<double> (M, 0.0);
    p1 = std::vector<double> (N, 0.0);


    // initializing under VAMP assumption
    //for (int i=0; i<r1.size(); i++)
    //    r1[i] += true_signal_s[i];

    //for (int i=0; i<p1.size(); i++)
    //    p1[i] += true_g[i];

    // initializing z1_hat and p2
    z1_hat = std::vector<double> (N, 0.0);
    p2 = std::vector<double> (N, 0.0);

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

        double rho_it;
        if (it > 1)
            rho_it = rho;
        else
            rho_it = 1;

        for (int i = 0; i < M; i++ )
            x1_hat[i] = rho_it * g1(r1[i], gam1) + (1 - rho_it) * x1_hat_prev[i];

        //std::vector<double> x1_hat_s = x1_hat;
        //for (int i=0; i<x1_hat_s.size(); i++)
        //    x1_hat_s[i] = x1_hat[i] / sqrt(N);

        probit_err_measures(dataset, 1, true_signal_s, x1_hat, "x1_hat");


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

        // true precision calcultion
        std::vector<double> r2mtrue = r2;
        for (int i=0; i<r2mtrue.size(); i++)
            r2mtrue[i] -= true_signal_s[i];

        //if (rank == 0)
        //    std::cout << "r2mtrue[0] = " << r2mtrue[0] << std::endl;
        double r2mtrue_std = calc_stdev(r2mtrue, 1);
        if (rank == 0)
            std::cout << "true gam2 = " << 1.0 / r2mtrue_std / r2mtrue_std << std::endl;

        // updating parameters of prior distribution
        probs_before = probs;
        vars_before = vars;
        // updatePrior(1);


        //************************
        //************************
        //      DENOISING Z
        //************************
        //************************
      
        std::vector<double> y = (*dataset).filter_pheno();

        probit_err_measures(dataset, 0, true_g, p1, "p1");

        std::vector<double> z1_hat_prev = z1_hat;

        // denoising part
        for (int i=0; i<N; i++)
            z1_hat[i] = rho_it * g1_bin_class(p1[i], tau1, y[i]) + (1-rho_it) * z1_hat_prev[i];

        probit_err_measures(dataset, 0, true_g, z1_hat, "z1_hat");

        double beta1 = 0;

        for (int i=0; i<N; i++){
            beta1 += g1d_bin_class(p1[i], tau1, y[i]);
            //if (rank == 0 && i <= 10){
            //    std::cout << " p1[i] = " << p1[i] << ", tau1 = " << tau1 << ", y[i] = " << y[i] << ", g1d_bin_class(p1[i], tau1, y[i]) = " << g1d_bin_class(p1[i], tau1, y[i]) << std::endl;
            //}
        }

        beta1 /= N;

        if (rank == 0)
            std::cout << "beta1 = " << beta1 << std::endl;
        

        for (int i=0; i<N; i++)
            p2[i] = (z1_hat[i] - beta1*p1[i]) / (1-beta1);

        std::vector<double> p2mtrue = p2;
        for (int i=0; i<p2mtrue.size(); i++)
            p2mtrue[i] -= true_g[i];
        double p2mtrue_std = calc_stdev(p2mtrue, 0);
        if (rank == 0)
            std::cout << "true tau2 = " << 1.0 / p2mtrue_std / p2mtrue_std << std::endl;

        tau2_prev = tau2;

        tau2 = rho_it * tau1 * (1-beta1) / beta1 + (1 - rho_it) * tau2_prev;

        if (rank == 0)
            std::cout << "tau2 = " << tau2 << std::endl;

        // updating probit_var
        //probit_var = update_probit_var(probit_var, tau1 / beta1, z1_hat, y);

        /*
        
        if (rank == 0){

            probit_var = update_probit_var(probit_var, tau1 / beta1, z1_hat, y);

            for (int ran = 1; ran < nranks; ran++)
                MPI_Send(&probit_var, 1, MPI_DOUBLE, ran, 0, MPI_COMM_WORLD);

        }
        else
            MPI_Recv(&probit_var, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        */

        if (rank == 0)
            std::cout << "probit_var = " << probit_var << std::endl;



        //************************
        //************************
        // LMMSE estimation of x
        //************************
        //************************

        
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

        probit_err_measures(dataset, 1, true_signal, x2_hat_s, "x2_hat");

        double alpha2 = g2d_onsager(gam2, tau2, dataset);
        if (rank == 0)
            std::cout << "alpha2 = " << alpha2 << std::endl;

        for (int i=0; i<M; i++)
            r1[i] = (x2_hat[i] - alpha2*r2[i]) / (1-alpha2);

        // true precision calculation
        std::vector<double> r1mtrue = r1;
        for (int i=0; i<r1mtrue.size(); i++)
            r1mtrue[i] -= true_signal_s[i];
        double r1mtrue_std = calc_stdev(r1mtrue, 1);
        if (rank == 0)
            std::cout << "true gam1 = " << 1.0 / r1mtrue_std / r1mtrue_std << std::endl;

        gam1 = gam2 * (1-alpha2) / alpha2;

        if (rank == 0)
            std::cout << "gam1 = " << gam1 << std::endl;



        
        //************************
        //************************
        // LMMSE estimation of z
        //************************
        //************************
        
        z2_hat = (*dataset).Ax(x2_hat.data());
        
        probit_err_measures(dataset, 0, true_g, z2_hat, "z2_hat");
        
        double beta2 = (double) N / Mt * (1-alpha2);
        beta2 = (double) Mt / N * (1 - alpha2);

        if (rank == 0)
            std::cout << "beta2 = " << beta2 << std::endl;

        for (int i=0; i<N; i++)
            p1[i] = (z2_hat[i] - beta2 * p2[i]) / (1-beta2);

        
        // true precision calculation
        std::vector<double> p1mtrue = p1;
        for (int i=0; i<p1mtrue.size(); i++)
            p1mtrue[i] -= true_g[i];
        double p1mtrue_std = calc_stdev(p1mtrue, 0);
        if (rank == 0)
            std::cout << "true tau1 = " << 1.0 / p1mtrue_std / p1mtrue_std << std::endl;

        tau1 = tau2 * (1 - beta2) / beta2;

        if (rank == 0)
            std::cout << "tau1 = " << tau1 << std::endl;
       

        // stopping criteria
        std::vector<double> x1_hat_diff(M, 0.0);

        for (int i0 = 0; i0 < x1_hat_diff.size(); i0++)
            x1_hat_diff[i0] = x1_hat_prev[i0] - x1_hat[i0];
        
        if (it > 1 && sqrt( l2_norm2(x1_hat_diff, 1) / l2_norm2(x1_hat_prev, 1) ) < stop_criteria_thr){
            if (rank == 0)
                std::cout << "probitVAMP stopping criteria fulfilled with threshold = " << stop_criteria_thr << "." << std::endl;
            break;
        }

    }
    
    return x1_hat;

}


double vamp::g1_bin_class(double p, double tau1, double y){

    double c = p / sqrt(probit_var + 1.0/tau1);

    //double normalCdf = normal_cdf( (2*y - 1)*c );
    //double normalPdf = exp(-0.5 * c * c) / sqrt(2*M_PI);

    /*
    if (rank == 0)
        if (abs(c) > 100)
            std::cout << "c = " << c << ", exp(-0.5 * c * c) = " << exp(-0.5 * c * c) << ", normal_cdf((2*y - 1)*c) = " << normal_cdf((2*y - 1)*c) << std::endl;
    */

    double temp;
    
    double normalPdf_normalCdf = 2.0 /  sqrt(2*M_PI) / erfcx( - (2*y-1) * c / sqrt(2) ); 

    /*
    if ( abs(normalCdf) < 1e-50 && abs(normalPdf) < 1e-50 )
        normalPdf_normalCdf = abs(c);
    else
        normalPdf_normalCdf = normalPdf / normalCdf;   
    */     

    temp = p + (2*y-1) * normalPdf_normalCdf / tau1 / sqrt(probit_var + 1.0/tau1);

    return temp;

}


double vamp::g1d_bin_class(double p, double tau1, double y){
    
    double c = p / sqrt(probit_var + 1.0/tau1);

    double Nc = exp(-0.5 * c * c) /  sqrt(2*M_PI);

    //double phic = normal_cdf(c);

    double phiyc = normal_cdf( (2*y-1)*c );

    //double Nc_phic;

    //if ( abs(phic) < 1e-10 && abs(Nc) < 1e-10 )
    //    Nc_phic = abs(c);
    //else 
    //    Nc_phic = Nc / phic;

    double Nc_phiyc = 2.0 /  sqrt(2*M_PI) / erfcx( - (2*y-1) * c / sqrt(2) );

    /*
    if ( abs(phiyc) < 1e-50 && abs(Nc) < 1e-50 )
        Nc_phiyc = abs(c);
    else 
        Nc_phiyc = Nc / phiyc;
    */

    double temp;

    temp = 1 -  Nc_phiyc / (1 + tau1 * probit_var) * ( (2*y-1)*c + Nc_phiyc ); // because der = tau * Var

    return temp;

}

double vamp::probit_var_EM_deriv(double v, std::vector<double> z, std::vector<double> y){

    double der = 0.0;

    for (int i=0; i<N; i++){

        double c = (2*y[i] - 1) * z[i] / v;
        double phic = normal_cdf(c);

        der += c * exp(-c*c/2) / sqrt(2*M_PI) * z[i] / v / phic;    
    }

    // double der_MPIsync;

    // MPI_Allreduce(&der, &der_MPIsync, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    // return der / N;

    return der;

}

double vamp::expe_probit_var_EM_deriv(double v, double eta, std::vector<double> z_hat, std::vector<double> y){

    // z ~ N(z_hat, I/eta)

    int miter_probit_expe_der = 1e3;

    int it = 1;

    double sum = 0;

    std::vector<double> z = simulate(N, std::vector<double> {1.0/eta}, std::vector<double> {1});

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

    // assert(sgn(var_min)==-sgn(var_max));

    for (; it<max_iter_bisec; it++){

        double fv = expe_probit_var_EM_deriv(new_v, eta, z_hat, y);

         if (rank == 0)
            std::cout << "bisec iter = " << it << ", fv = " << fv << ", v = " << new_v <<  std::endl;


        if (abs(fv) < 1e-6)
            break;

        if (sgn(fv) == sgn(var_min))
            var_min = v;
        else if (sgn(fv) == sgn(var_max))
            var_max = v;
        
        // refine midpoint on a log scale
        // new_v = exp(0.5*(log(var_min) + log(var_max)));

        v = sqrt(var_min * var_max);

    }

    if (rank == 0)
        std::cout << "bisection method finished after " << it << " / " << max_iter_bisec << "iterations" << std::endl;
    
    return new_v;

}

std::vector<double> vamp::grad_cov(std::vector<double> y, std::vector<double> gg, double probit_var, std::vector< std::vector<double> > Z, std::vector<double> eta){ 
    // gg = g_genetics, gc = g_covariates, eta = vector of covariate's effect sizes

    std::vector<double> grad(C, 0.0);

    for (int j=0; j<C; j++){

        for (int i=0; i<N; i++){

            double g_i = gg[i] + inner_prod(Z[i], eta, 0);

            double arg = (2*y[i] - 1) / sqrt(probit_var) * g_i;

            double ratio = 2.0 /  sqrt(2*M_PI) / erfcx( - arg / sqrt(2) );

            grad[j] += ratio * (2*y[i]-1) / sqrt(probit_var) * Z[i][j];

        }

    }

    return grad;
}

double vamp::mlogL_probit(std::vector<double> y, std::vector<double> gg, double probit_var, std::vector< std::vector<double> > Z, std::vector<double> eta){

    double mlogL = 0;

    for (int i=0; i<N; i++){

        double g_i = gg[i] + inner_prod(Z[i], eta, 0);

        double arg = (2*y[i] - 1) / sqrt(probit_var) * g_i;

        double phi_arg = normal_cdf(arg);

        mlogL -= log(phi_arg);
    }

    return mlogL;
}

std::vector<double> vamp::grad_desc_step_cov(std::vector<double> y, std::vector<double> gg, double probit_var, std::vector< std::vector<double> > Z, std::vector<double> eta, double step_size){

    std::vector<double> new_eta(C, 0.0);

    std::vector<double> grad = grad_cov(y, gg, probit_var, Z, eta);
    
    double scale = 1;

    double best_val = mlogL_probit(y, gg, probit_var, Z, eta);

    for (int i = 1; i<20; i++){ // 0.8^20 = 0.01152922

        for (int j=0; j<C; j++)
            grad[j] *= scale;
        
        scale = 0.8;

        std::transform (eta.begin(), eta.end(), grad.begin(), new_eta.begin(), std::minus<double>());

        double curr_val = mlogL_probit(y, gg, probit_var, Z, eta);

        if (best_val < curr_val)
            best_val = curr_val;
        else
            break;

        eta = new_eta;

    }

    return new_eta;

}

void vamp::probit_err_measures(data *dataset, int sync, std::vector<double> true_signal, std::vector<double> est, std::string var_name){
    
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


    // precision calculation
    //double std = calc_stdev(temp, sync);

    //if (rank == 0)
    //    std::cout << "true precision for " + var_name + " = " << 1.0 / N / std / std << std::endl;


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