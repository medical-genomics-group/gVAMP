#include <vector>
#include <algorithm>
#include <iostream>
#include <mpi.h>
#include <cmath>
#include <numeric> // contains std::accumulate
#include <random>
#include <omp.h>
#include <fstream>
#include "vamp.hpp"
#include "utilities.hpp"

//constructor for class data
vamp::vamp(int N, int M,  int Mt, double gam1, double gamw, int max_iter, double rho, std::vector<double> vars,  std::vector<double> probs, std::vector<double> true_signal, int rank, std::string out_dir, std::string out_name, std::string model) :
    N(N),
    M(M),
    Mt(Mt),
    gam1(gam1),
    gamw(gamw),
    gam2(0),
    eta1(0),
    eta2(0),
    max_iter(max_iter),
    rho(rho),
    vars(vars),
    probs(probs),
    out_dir(out_dir),
    out_name(out_name),
    true_signal(true_signal),
    model(model),
    rank(rank)  {
    x1_hat = std::vector<double> (M, 0.0);
    x2_hat = std::vector<double> (M, 0.0);
    r1 = std::vector<double> (M, 0.0);
    r2 = std::vector<double> (M, 0.0);
    p1 = std::vector<double> (N, 0.0);

    Options opt;
    EM_max_iter = opt.get_EM_max_iter();
    EM_err_thr = opt.get_EM_err_thr();
    CG_max_iter = opt.get_CG_max_iter();
    stop_criteria_thr = opt.get_stop_criteria_thr();
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
}

//constructor for class data
vamp::vamp(int M, double gam1, double gamw, std::vector<double> true_signal, int rank, Options opt):
    M(M),
    gam1(gam1),
    gamw(gamw),
    gam2(0),
    eta1(0),
    eta2(0),
    rho(opt.get_rho()),
    probs(opt.get_probs()),
    out_dir(opt.get_out_dir()),
    out_name(opt.get_out_name()),
    true_signal(true_signal),
    model(opt.get_model()),
    rank(rank)  {
    N = opt.get_N();
    Mt = opt.get_Mt();
    max_iter = opt.get_iterations();
    x1_hat = std::vector<double> (M, 0.0);
    x2_hat = std::vector<double> (M, 0.0);
    p1 = std::vector<double> (N, 0.0);
    r1 = std::vector<double> (M, 0.0);
    r2 = std::vector<double> (M, 0.0);
    EM_max_iter = opt.get_EM_max_iter();
    EM_err_thr = opt.get_EM_err_thr();
    CG_max_iter = opt.get_CG_max_iter();
    stop_criteria_thr = opt.get_stop_criteria_thr();
    vars = opt.get_vars();
    // we scale the signal prior with N since X -> X / sqrt(N)
    for (int i=0; i<vars.size(); i++)
        vars[i] *= N;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
}

//std::vector<double> predict(std::vector<double> est, data* dataset){
//    return (*dataset).Ax(est.data());
//}

std::vector<double> vamp::infere( data* dataset ){

    if (!strcmp(model.c_str(), "linear"))
        return infere_linear(dataset);
    else if (!strcmp(model.c_str(), "bin_class"))
        return infere_bin_class(dataset);
    else
        throw "invalid model specification!";

    return std::vector<double> (M, 0.0);
}

std::vector<double> vamp::infere_bin_class( data* dataset ){


    double tol = 1e-11;

    std::vector<double> x1_hat_d(M, 0.0);
    std::vector<double> x1_hat_d_prev(M, 0.0);
    std::vector<double> r1_prev(M, 0.0);
    std::vector<double> x1_hat_prev(M, 0.0);
    tau1 = gam1; // hardcoding initial variability

    for (int it = 1; it <= max_iter; it++)
    {    

        //************************
        //      Denoising x
        //************************

        if (rank == 0)
                std::cout << std::endl << "********************" << std::endl << "iteration = "<< it << std::endl << "********************" << std::endl << "->DENOISING" << std::endl;

        x1_hat_prev = x1_hat;

        // updating parameters of prior distribution
        probs_before = probs;
        vars_before = vars;
        updatePrior();

        for (int i = 0; i < M; i++ )
            x1_hat[i] = rho * g1(r1[i], gam1) + (1 - rho) * x1_hat_prev[i];

        std::string filepath_out = out_dir + out_name + "_bin_class_" + "_it_" + std::to_string(it) + "_rank_" + std::to_string(rank) + ".txt"; 
        if (rank == 0)
            std::cout << "filepath_out is " << filepath_out << std::endl;

        store_vec_to_file(filepath_out, x1_hat);

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
        //std::cout << "alpha1 after MPI_Allreduce() = "<< alpha1 << std::endl;
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



        //************************
        //      Denoising z
        //************************
                
        std::vector<double> y = (*dataset).get_phen();
        // updating probit_var
        if (rank == 0){
            probit_var = update_probit_var(probit_var, y);
            for (int ran = 1; ran < nranks; ran++)
                MPI_Send(&probit_var, 1, MPI_DOUBLE, ran, 0, MPI_COMM_WORLD);
        }
        else
            MPI_Recv(&probit_var, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);


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



        //************************
        // LMMSE estimation of x
        //************************

        std::vector<double> v = (*dataset).ATx(p2.data());

        for (int i = 0; i < M; i++)
            v[i] = tau2 * v[i] + gam2 * r2[i];

        // running conjugate gradient solver to compute LMMSE
        double start_CG = MPI_Wtime();
        x2_hat = CG_solver(v, tau2, dataset);
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

        if (it > 1 && sqrt( l2_norm2(x1_hat_diff, 1) / l2_norm2(x1_hat_prev, 1) ) < stop_criteria_thr){
            if (rank == 0)
                std::cout << "stopping criteria fulfilled" << std::endl;
            break;
        }
    }
    
    return x1_hat;

}


std::vector<double> vamp::infere_linear( data* dataset ){

    double tol = 1e-11;

    std::vector<double> x1_hat_d(M, 0.0);
    std::vector<double> x1_hat_d_prev(M, 0.0);
    std::vector<double> r1_prev(M, 0.0);
    std::vector<double> x1_hat_prev(M, 0.0);

    for (int it = 1; it <= max_iter; it++)
    {    

        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // %%%%%%%%%%% Denoising step %%%%%%%%%%%%%%
        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        double start_denoising = MPI_Wtime();

        if (rank == 0)
            std::cout << std::endl << "********************" << std::endl << "iteration = "<< it << std::endl << "********************" << std::endl << "->DENOISING" << std::endl;

        x1_hat_prev = x1_hat;

        // updating parameters of prior distribution
        probs_before = probs;
        vars_before = vars;
        updatePrior();

        for (int i = 0; i < M; i++ )
            x1_hat[i] = rho * g1(r1[i], gam1) + (1 - rho) * x1_hat_prev[i];

        z1 = (*dataset).Ax(x1_hat.data());
        // we start adaptive damping from iteration numb_adap_damp_hist
        while (use_adap_damp == 1 && it>numb_adap_damp_hist){
            double obj_fun_val = vamp_obj_func(eta1, gam1, invQ_bern_vec, bern_vec, vars, probs, dataset);
            double min_obj_fun_val = *std::min_element(obj_fun_vals.begin(), obj_fun_vals.end()); 
            if (obj_fun_val >= min_obj_fun_val){
                obj_fun_vals.erase(obj_fun_vals.begin());
                obj_fun_vals.push_back(obj_fun_val);
                rho = std::min(rho*damp_inc_fact, damp_max);
                break;
            }
            else{
                rho *= damp_dec_fact;
                for (int i = 0; i < M; i++)
                    x1_hat[i] = rho * g1(r1[i], gam1) + (1 - rho) * x1_hat_prev[i];
                z1 = (*dataset).Ax(x1_hat.data());
                assert(rho > 1e-8);                  
            }   
        }

        if (rank == 0)
            std::cout << "rho = " << rho << std::endl;
        //std::cout << "x1_hat_d[0] = " << x1_hat_d[0] << std::endl;
        //std::cout << "g1d(0) = " << g1d(0, gam1) << "r1[0] = " << r1[0] << std::endl;
        //std::cout << "sum_d = " << sum_d << std::endl;
        //std::string filepath_out = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/sig_estimates/x1_hat_it_" + std::to_string(it) + "_rank_" + std::to_string(rank) + ".txt";
        
        // saving x1_hat
        //MPI_File outfh;
        //MPI_Status status;
        std::string filepath_out = out_dir + out_name + "_it_" + std::to_string(it) + "_rank_" + std::to_string(rank) + ".txt";
        //std::string filepath_out = out_dir + out_name + "_it_" + std::to_string(it) + ".bin";
        if (rank == 0)
            std::cout << "filepath_out is " << filepath_out << std::endl;
        /*
        MPI_File_open(MPI_COMM_WORLD, filepath_out.c_str(), MPI_MODE_WRONLY|MPI_MODE_CREATE, MPI_INFO_NULL, &outfh);
        MPI_Offset offset = size_t((*dataset).get_S());
        std::cout << "rank = " << rank << ", S = " << offset << ", x1_hat[0] = " << x1_hat[0] << std::endl;
        check_mpi(MPI_File_write_at_all(outfh, offset, x1_hat.data(), x1_hat.size(), MPI_DOUBLE, &status), __LINE__, __FILE__);
        MPI_File_close(&outfh);
        */
        store_vec_to_file(filepath_out, x1_hat);

        x1_hat_d_prev = x1_hat_d;
        double sum_d = 0;
        for (int i = 0; i < M; i++)
        {
            // we have to keep the entire derivative vector so that we could have its previous version in the damping step 
            x1_hat_d[i] = rho * g1d(r1[i], gam1) + (1 - rho) * x1_hat_d_prev[i]; 
            sum_d += x1_hat_d[i];
        }

        if (calc_state_evo == 1){
            std::tuple<double, double, double> state_evo_par2 = state_evo(1, gam1, gam_before, probs_before, vars_before, dataset);
            if (rank == 0)
                std::cout << "gam2_bar = " << std::get<2>(state_evo_par2) << std::endl; 
        }

        //std::cout << "sum_d = " << sum_d << std::endl;
        double alpha1;
        MPI_Allreduce(&sum_d, &alpha1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        //std::cout << "alpha1 after MPI_Allreduce() = "<< alpha1 << std::endl;
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
        //std::cout << "r2[1] = "<< r2[1] << std::endl;
        //std::cout << "norm(r2)^2 = " << l2_norm2(r2, 1) << std::endl;
    
        err_measures(dataset, 1);

        double end_denoising = MPI_Wtime();

        if (rank == 0)
            std::cout << "denoising step took " << end_denoising - start_denoising << " seconds." << std::endl;


        //x1_hat = true_signal; // just for noise precision learning tests
        //for (int i0 = 0; i0 < M; i0++)
        //    x1_hat[i0] = sqrt(N) * x1_hat[i0];


        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // %%%%%%%%%% LMMSE step %%%%%%%%%%%%%%
        // %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        double start_lmmse_step = MPI_Wtime();

        if (rank == 0)
            std::cout << "______________________" << std::endl<< "->LMMSE" << std::endl;

        //gamw = 1 / ( 1 - inner_prod(z1, z1, 0) );

        std::vector<double> y = (*dataset).get_phen();
        std::vector<double> v = (*dataset).ATx(y.data());

        for (int i = 0; i < M; i++)
            v[i] = gamw * v[i] + gam2 * r2[i];

        // running conjugate gradient solver to compute LMMSE
        double start_CG = MPI_Wtime();
        x2_hat = CG_solver(v, gamw, dataset);
        double end_CG = MPI_Wtime();

        if (rank == 0)
            std::cout << "CG took "  << end_CG - start_CG << " seconds." << std::endl;

        /*
        if (calc_state_evo == 1){
            std::tuple<double, double, double> state_evo_par1 = state_evo(2, gam2, gam_before, probs_before, vars_before, dataset);
            std::cout << "gam1_bar = " << get<2>(state_evo_par1) << std::endl; 
        }
        */

        double start_onsager = MPI_Wtime();
        double alpha2 = g2d_onsager(gam2, gamw, dataset);
        double end_onsager = MPI_Wtime();

        if (rank == 0)
            std::cout << "onsager took "  << end_onsager - start_onsager << " seconds." << std::endl;
        
        if (rank == 0)
            std::cout << "alpha2 = " << alpha2 << std::endl;
        eta2 = gam2 / alpha2;
        gam_before = gam1;
        gam1 = rho * std::min( std::max( eta2 - gam2, gamma_min ), gamma_max ) + (1 - rho) * gam1; 
        if (rank == 0)
            std::cout << "gam1 = " << gam1 << std::endl;

        r1_prev = r1;
        for (int i = 0; i < M; i++)
            r1[i] = (eta2 * x2_hat[i] - gam2 * r2[i]) / gam1;

        // learning a noise precision parameter
        updateNoisePrec(dataset);
   
        // printing out error measures
        err_measures(dataset, 2);
        
        double end_lmmse_step = MPI_Wtime();

        if (rank == 0)
            std::cout << "lmmse step took "  << end_lmmse_step - start_lmmse_step << " seconds." << std::endl;
        
        // stopping criteria
        std::vector<double> x1_hat_diff(M, 0.0);
        for (int i0 = 0; i0 < x1_hat_diff.size(); i0++)
            x1_hat_diff[i0] = x1_hat_prev[i0] - x1_hat_diff[i0];

        if (it > 1 && sqrt( l2_norm2(x1_hat_diff, 1) / l2_norm2(x1_hat_prev, 1) ) < stop_criteria_thr){
            if (rank == 0)
                std::cout << "stopping criteria fulfilled" << std::endl;
            break;
        }
            

        if (rank == 0)
            std::cout << std::endl << std::endl;
    }
    //if (rank == 0)
    //    std::cout << "inside vamp: calc_stdev(x1_hat) = " << calc_stdev(x1_hat) << std::endl; 
    return x1_hat;          
}

double vamp::g1(double y, double gam1) { 
    
    double sigma = 1 / gam1;
    //double eta_max = vars[0];
    //for (int j = 1; j < probs.size(); j++){
    //    if (vars[j] > eta_max)
    //        eta_max = vars[j];
    //}
    double eta_max = *(std::max_element(vars.begin(), vars.end()));
    //std::cout << "eta_max = "<< eta_max << std::endl;
    double pk = 0, pkd = 0, val;

    if (sigma < 1e-10 && sigma > -1e-10) {
        return y;
    }

    for (int i = 0; i < probs.size(); i++){

        double expe_sum =  - 0.5 * pow(y,2) * ( eta_max - vars[i] ) / ( vars[i] + sigma ) / ( eta_max + sigma );

        double z = probs[i] / sqrt( vars[i] + sigma ) * exp( expe_sum );

        pk = pk + z;

        z = z / ( vars[i] + sigma ) * y;

        pkd = pkd - z; 

    }

    val = (y + sigma * pkd / pk);
    
    return val;
}

double vamp::g1d(double y, double gam1) { 
    
        double sigma = 1 / gam1;
        double eta_max = *std::max_element(vars.begin(), vars.end());
        double pk = 0, pkd = 0, pkdd = 0;
        
        if (sigma < 1e-10 && sigma > -1e-10) {
            return 1;
        }

        for (int i = 0; i < probs.size(); i++){

            double expe_sum = - 0.5 * pow(y,2) * ( eta_max - vars[i] ) / ( vars[i] + sigma ) / ( eta_max + sigma );

            double z = probs[i] / sqrt( vars[i] + sigma ) * exp( expe_sum );

            pk = pk + z;

            z = z / ( vars[i] + sigma ) * y;

            pkd = pkd - z;

            double z2 = z / ( vars[i] + sigma ) * y;

            pkdd = pkdd - probs[i] / pow( vars[i] + sigma, 1.5 ) * exp( expe_sum ) + z2;
            
        } 

        double val = (1 + sigma * ( pkdd / pk - pow( pkd / pk, 2) ) );

        return val;
}

double vamp::g2d_onsager(double gam2, double tau, data* dataset) { // shared between linear and binary classification model
    
    std::random_device rd;
    std::bernoulli_distribution bern(0.5);

    bern_vec = std::vector<double> (M, 0.0);
    for (int i = 0; i < M; i++)
        bern_vec[i] = 2*bern(rd) - 1;
    std::vector<double> invQ_bern_vec = CG_solver(bern_vec, tau, dataset);
    double onsager = gam2 * inner_prod(bern_vec, invQ_bern_vec, 1) / Mt; 
    return onsager;
    
}

void vamp::updateNoisePrec(data* dataset){

    std::random_device rd;
    std::bernoulli_distribution bern(0.5);
    size_t phen_size = 4 * (*dataset).get_mbytes();
    std::vector<double> u = std::vector<double> (phen_size, 0.0);
    if (rank == 0){
        for (int i = 0; i < phen_size; i++)
            u[i] = 2*bern(rd) - 1;
        for (int ran = 1; ran < nranks; ran++)
            MPI_Send(u.data(), phen_size, MPI_DOUBLE, ran, 0, MPI_COMM_WORLD);
            //std::cout << "nranks = " << nranks << std::endl;
    } else if (rank != 0){
        MPI_Recv(u.data(), phen_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    std::vector<double> v = (*dataset).ATx(u.data());
    std::vector<double> temp = (*dataset).Ax(x2_hat.data());
    std::vector<double> y = (*dataset).get_phen();
    for (int i = 0; i < N; i++)  // because length(y) = N
        temp[i] -= y[i];
    
    double temp_norm2 = l2_norm2(temp, 0); 
    double trace_corr = inner_prod(CG_solver(v, gamw, dataset), v, 1); // check this! ad
    //std::vector<double> z2 = (*dataset).Ax(x2_hat.data());
    if (rank == 0){
        std::cout << "l2_norm2(temp) / N = " << temp_norm2 / N << std::endl;
        std::cout << "trace_correction / N = " << trace_corr / N << std::endl;
        //std::cout << "var(y) = " << inner_prod(y, y, 0) / N << std::endl;
        //std::cout << "alternative gamw = " << 1 / ( 1 - inner_prod(z1, z1, 0) ) << std::endl;
    }
    gamw = (double) N / (temp_norm2 + trace_corr);
     
}

void vamp::updatePrior() {
    
        //double max_sigma = *(std::max_element( vars ) ); 
        double max_sigma = *std::max_element(vars.begin(), vars.end()); // std::max_element returns iterators, not values
        double noise_var = 1 / gam1;
        double lambda = 1 - probs[0];
        std::vector<double> omegas = probs;
        for (int j = 1; j < omegas.size(); j++) // omegas is of length L
            omegas[j] /= lambda;
        //for_each(probs.begin() + 1, probs.end(), [lambda](int &c){ c /= lambda; });
        //omegas = pi_prev(2:end) / lambda;

        //double* beta = new double[M * probs.size()];
                       
        // calculating normalized beta and pin
        for (int it = 0; it < EM_max_iter; it++){

            std::vector<double> probs_prev = probs;
            std::vector< std::vector<double> > gammas;
            std::vector< std::vector<double> > beta;
            std::vector<double> pin(M, 0.0);
            std::vector<double> v;

            for (int i = 0; i < M; i++){
                std::vector<double> temp; // of length (L-1)
                std::vector<double> temp_gammas;
                for (int j = 1; j < probs.size(); j++ ){
                    double num = lambda * exp( - pow(r1[i], 2) / 2 * (max_sigma - vars[j]) / (vars[j] + noise_var) / (max_sigma + noise_var) ) / sqrt(vars[j] + noise_var) / sqrt(2 * M_PI) * omegas[j];
                    double num_gammas = gam1 * r1[i] / ( 1 / vars[j] + gam1 );
                    //if (rank == 0 && i == 0)
                        //std::cout << "num = " << num << std::endl;
                    temp.push_back(num);
                    temp_gammas.push_back(num_gammas);
                }
                double sum_of_elems = std::accumulate(temp.begin(), temp.end(), decltype(temp)::value_type(0));
                //if (rank == 0 && i == 0)
                    //std::cout << "sum_of_elems = " << sum_of_elems << ", it = " << it << std::endl;
                //for_each(temp.begin(), temp.end(), [sum_of_elems](int &c){ c /= sum_of_elems; });
                for (int j = 0; j < temp.size(); j++ ){
                    temp[j] /= sum_of_elems;
                    //if (rank == 0 && i == 0)
                        //std::cout << "j = " << j << ", temp[j] = " << temp[j] << std::endl;
                }
                beta.push_back(temp);
                gammas.push_back(temp_gammas);
                pin[i] = 1 / ( 1 + (1-lambda) / sqrt(2 * M_PI * noise_var) * exp( - pow(r1[i], 2) / 2 * max_sigma / noise_var / (noise_var + max_sigma) ) / sum_of_elems  );
            } 
            //if (rank == 0)
                //std::cout << "pin[0] = " << pin[0] << std::endl;
            for (int j = 1; j < probs.size(); j++){
                v.push_back( 1 / ( 1 / vars[j] + gam1 ) ); // v is of size (L-1) in the end
            }
            lambda = accumulate(pin.begin(), pin.end(), 0.0); // / pin.size();

            double lambda_total = 0;
            MPI_Allreduce(&lambda, &lambda_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            lambda = lambda_total / Mt;

            //if (rank == 0)
                //std::cout << "lambda = " << lambda << std::endl;
        
            for (int i = 0; i < M; i++){
                for (int j = 0; j < (beta[1]).size(); j++ ){
                    gammas[i][j] = beta[i][j] * ( pow(gammas[i][j], 2) + v[j] );
                }
            }

            double sum_of_pin = std::accumulate(pin.begin(), pin.end(), decltype(pin)::value_type(0));

            double sum_of_pin_total = 0;
            MPI_Allreduce(&sum_of_pin, &sum_of_pin_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            sum_of_pin = sum_of_pin_total;

            for (int j = 0; j < (beta[0]).size(); j++){ // of length (L-1)
                double res = 0, res_gammas = 0;
                for (int i = 0; i < M; i++){
                    res += beta[i][j] * pin[i];
                    res_gammas += gammas[i][j] * pin[i];
                }

                double res_gammas_total = 0;
                double res_total = 0;
                MPI_Allreduce(&res_gammas, &res_gammas_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                MPI_Allreduce(&res, &res_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

                vars[j+1] = res_gammas_total / res_total;
                omegas[j+1] = res_total / sum_of_pin;
                probs[j+1] = lambda * omegas[j+1];

                //if (rank == 0)
                    //std::cout << " j = " << j << ", vars[j+1] = " << vars[j+1] << ", probs[j+1] = " << probs[j+1] << std::endl;
            }
            probs[0] = 1 - lambda;
        
        double distance = 0, norm_probs = 0;
        for (int j = 0; j < probs.size(); j++){
            distance += ( probs[j] - probs_prev[j] ) * ( probs[j] - probs_prev[j] );
            norm_probs += probs[j] * probs[j];
        }
        
        if ( distance / norm_probs < EM_err_thr ){
            if (rank == 0)
                std::cout << "final number of prior EM iterations = " << it + 1 << std::endl;
            break;   
        }

        }
        //else
            //if (rank ==0)
                //std::cout << "distance / norm_probs = " << distance / norm_probs << std::endl;
        //}   
        
        //if (rank == 0)
        //        std::cout << "final number of prior EM iterations = " << EM_max_iter << std::endl;
}

std::vector<double> vamp::lmmse_mult(std::vector<double> v, double tau, data* dataset){ // multiplying with (tau*A^TAv + gam2*v)

    std::vector<double> res(M, 0.0);
    size_t phen_size = 4 * (*dataset).get_mbytes();
    std::vector<double> res_temp(phen_size, 0.0);;
    //std::cout << "before res_temp!" << std::endl;
    res_temp = (*dataset).Ax( v.data() );
    //std::cout << "before res!" << std::endl;
    res = (*dataset).ATx( res_temp.data() );
    //std::cout << "before loop!" << std::endl;
    for (int i = 0; i < M; i++){
        //res[i] *= gamw;
        res[i] *= tau;
        res[i] += gam2 * v[i];
        if (std::isnan(res[i] && i < 5000)){
        std::cout << "index at position "<< i << " is nan." << std::endl; 
        }
    }
    //std::cout << "[lmmse_mult] res[1] = " << res[1] << std::endl;

    return res;
}

std::vector<double> vamp::CG_solver(std::vector<double> v, double tau, data* dataset){

    // we start with approximation x0 = 0
    std::vector<double> mu(M, 0.0);
    std::vector<double> p = v, r = v, d;
    //std::vector<double> r = v;
    //std::vector<double> d;
    std::vector<double> p_temp(M, 0.0);
    std::vector<double> palpha(M, 0.0);
    double alpha, beta;
    for (int i = 0; i < CG_max_iter; i++){
        //std::cout << "[CG_solver] i = " << i << ", rank = " << rank << std::endl;
        double start_lmmse = MPI_Wtime();
        d = lmmse_mult(p, tau, dataset);
        double end_lmmse = MPI_Wtime();
        //if (rank == 0 && (i % 10 == 0) )
            //std::cout << "[CG_solver] time for lmmse = " << end_lmmse - start_lmmse << std::endl;
        //std::vector<double> p_temp(M, 0.0);
        alpha = l2_norm2(r, 1) / inner_prod(d, p, 1);
        
        //std::cout << "[CG_solver] l2_norm2(r, 1) = " << l2_norm2(r, 1) << std::endl;
        //std::cout << "[CG_solver] alpha = " << alpha << ", rank = " << rank << std::endl;
        /*std::cout << "[CG_solver] mu[1] = " << mu[1] << std::endl;
        std::cout << "[CG_solver] mu[2] = " << mu[2] << std::endl;
        std::cout << "[CG_solver] mu[3] = " << mu[3] << std::endl;
        std::cout << "[CG_solver] p[1] = " << p[1] << std::endl;
        std::cout << "[CG_solver] p[2] = " << p[2] << std::endl;
        std::cout << "[CG_solver] p[3] = " << p[3] << std::endl;
        */
        for (int j = 0; j < M; j++)
            palpha[j] = alpha * p[j];
        std::transform (mu.begin(), mu.end(), palpha.begin(), mu.begin(), std::plus<double>());
        for (int j = 0; j < p.size(); j++)
            p_temp[j] = d[j] * alpha;
        beta = pow( l2_norm2(r, 1), -1 );
        //std::cout << "(r-p_temp)[1] = " << r[1] - p_temp[1] << std::endl;
        std::transform (r.begin(), r.end(), p_temp.begin(), r.begin(), std::minus<double>());
        //std::cout << "r_new[1] = " << r[1] << std::endl;
        beta *= l2_norm2(r, 1);
        //std::cout << "[CG_solver] beta = " << beta << ", rank = " << rank << std::endl;
        //for (int j = 0; j < p.size(); j++)
        //    p_temp[j] = p[j] * beta;
        for (int j = 0; j < p.size(); j++)
            p[j] = r[j] + beta * p[j];
    }

    return mu;
 }

void vamp::err_measures(data *dataset, int ind){

    //double scale = 4.81e-5 / vars[vars.size()-1];
    double scale = 1.0 / (double) N;
    //scale = 1.0;
    
    // correlation
    if (ind == 1){
        double corr = inner_prod(x1_hat, true_signal, 1) / sqrt( l2_norm2(x1_hat, 1) * l2_norm2(true_signal, 1) );
        if ( rank == 0 )
            std::cout << "correlation x1_hat = " << corr << std::endl;  
        double l2_norm2_x1_hat = l2_norm2(x1_hat, 1);
        double l2_norm2_true_signal = l2_norm2(true_signal, 1);
        //if (rank == 0)
        //    std::cout << "l2_norm2(x1_hat) / N = " << l2_norm2_x1_hat / N << ", l2_norm2(true_signal) = " << l2_norm2_true_signal << std::endl;
    }
    else if (ind == 2){
        double corr_2 = inner_prod(x2_hat, true_signal, 1) / sqrt( l2_norm2(x2_hat, 1) * l2_norm2(true_signal, 1) );
        if (rank == 0)
            std::cout << "correlation x2_hat = " << corr_2 << std::endl;
        double l2_norm2_x2_hat = l2_norm2(x2_hat, 1);
        double l2_norm2_true_signal = l2_norm2(true_signal, 1);
        //if (rank == 0)
        //    std::cout << "l2_norm2(x2_hat) / N = " << l2_norm2_x2_hat / N << ", l2_norm2(true_signal) = " << l2_norm2_true_signal << std::endl;
    }
    

    // l2 signal error
    std::vector<double> temp(M, 0.0);
    if (ind == 1){
        for (int i = 0; i< M; i++)
            temp[i] = sqrt(scale) * x1_hat[i] - true_signal[i];
    }
    else if (ind == 2){
        for (int i = 0; i< M; i++)
            temp[i] = sqrt(scale) * x2_hat[i] - true_signal[i];
    }
    
    //std::cout << "[err_measures] temp[1] = " << temp[1] << std::endl; 
    double l2_signal_err = sqrt( l2_norm2(temp, 1) / l2_norm2(true_signal, 1) );
    if (rank == 0)
        std::cout << "l2 signal error = " << l2_signal_err << std::endl;
    
    // l2 prediction error
    size_t phen_size = 4 * (*dataset).get_mbytes();
    std::vector<double> tempNest(phen_size, 0.0);
    std::vector<double> tempNtrue(phen_size, 0.0);
    std::vector<double> y = (*dataset).get_phen();
    std::vector<double> Axest;
    if (ind == 1)
        //Axest = (*dataset).Ax( x1_hat.data() );
        Axest = z1;
    else if (ind == 2)
       Axest = (*dataset).Ax( x2_hat.data() );
    //std::vector<double> Axtrue = (*dataset).Ax( true_signal.data() );

    for (int i = 0; i < N; i++){ // N because length(y) = N even though length(Axest) = 4*mbytes
        tempNest[i] = -Axest[i] + y[i]; 
    }

    double l2_pred_err = sqrt(l2_norm2(tempNest, 0) / l2_norm2(y, 0));
    if (rank == 0)
        std::cout << "l2 prediction error = " << l2_pred_err << std::endl;

    double R2 = 1 - l2_pred_err * l2_pred_err;
    if (rank == 0)
        std::cout << "R2 = " << R2 << std::endl;

    // prior distribution parameters
    if (rank == 0)
        std::cout << "prior variances = ";
    for (int i = 0; i < vars.size(); i++){
        if (rank == 0)
            std::cout << vars[i] << ' ';
    }
    if (rank == 0) {
    std::cout << std::endl;
    std::cout << "prior probabilities = ";
    }
    for (int i = 0; i < probs.size(); i++){
        if (rank == 0)
            std::cout << probs[i] << ' ';
    }
    if (rank == 0){
    std::cout << std::endl;
    std::cout << "gamw = " << gamw << std::endl;
    }
    
}

double vamp::vamp_obj_func(double eta, double gam, std::vector<double> invQu, std::vector<double> u, std::vector<double> vars, std::vector<double> pi, data* dataset){
  
    std::random_device rd;
    std::bernoulli_distribution bern(0.5);

    // calculating differential entropy of error distribution q
    double Hq = Mt / 2 * (1 + log(2 * M_PI / eta));
    

    // calculating KL divergence between two multivariate gaussians
    double DKL2 = 0;
    // Chebishev approximation of a log(det) component
    //DKL2 += ( 2*eta2 - (2 + log(2)) ) * Mt;
    // Taylor approximation of a log(det) component
    DKL2 += - Mt*log(gamw * (*dataset).get_sigma_max() * (*dataset).get_sigma_max() + gam2);
    for (int i = 0; i < M; i++)
        u[i] = 2*bern(rd) - 1;
    std::vector<double> temp_poly_apr = u;
    double sign = 1;
    for (int i=1; i<=poly_deg_apr_vamp_obj; i++){
        temp_poly_apr = lmmse_mult(temp_poly_apr, gamw, dataset); // this function (vamp_obj_func) is hardcoded for the linear model
        for (int i1=0; i1<M; i1++)
            temp_poly_apr[i1] -= 1;
        DKL2 += sign * inner_prod(temp_poly_apr, u, 1) / (gamw * (*dataset).get_sigma_max() * (*dataset).get_sigma_max() + gam2) / i;
        sign *= -1;
    } 
    
    // trace component
    //std::vector<double> u = std::vector<double> (M, 0.0);
    //for (int i = 0; i < M; i++)
    //    u[i] = 2*bern(rd) - 1;
    std::vector<double> temp = (*dataset).Ax(invQu.data());
    std::vector<double> temp2 = (*dataset).ATx(temp.data());
    for (int i=0; i<temp2.size(); i++)
        temp2[i] *= gamw;
    DKL2 += inner_prod(u, temp2, 1);

    // quadratic form component
    //std::vector<double> z2 = (*dataset).Ax(x2_hat.data());
    DKL2 += (-2) * gamw * inner_prod( (*dataset).get_phen(), z1, 0);
    DKL2 += gamw * inner_prod(x1_hat, (*dataset).ATx(z1.data()), 1);
    DKL2 += (-1) * Mt / 2 * log(gamw);

    //for (int i = 0; i < M; i++)
        //u[i] = 2*bern(rd) - 1;
    //DKL2 -= inner_prod(u, CG_solver(u, gamw, dataset), 1);
    //DKL2 -= Mt / eta2;

    // KL divergence between two gaussian mixtures
    double DKL3 = 0;
    int max_iter_obj_MC=1e5;
    std::vector<double> vars_plus_gam = vars;
    for (int i=0; i<vars_plus_gam.size(); i++){
        vars_plus_gam[i] = vars[i] + 1/gam;
    }

    for (int it_MC=0; it_MC<max_iter_obj_MC; it_MC++){
        double x = generate_mixture_gaussians(vars_plus_gam.size(), vars_plus_gam, pi);
        DKL3 += log( mix_gauss_pdf_ratio(x, vars_plus_gam, vars, pi) );
    }    
    DKL3 /= max_iter_obj_MC;

    return Hq + DKL2 + DKL3;
}

std::tuple<double, double, double> vamp::state_evo(int ind, double gam_prev, double gam_before, std::vector<double> probs_before, std::vector<double> vars_before, data* dataset){

    //double gam_before, std::vector<double> probs_before, std::vector<double> vars_before
    // damping is not taken into account
    double alpha_bar = 0, eta_bar = 0, gam_bar = 0;

    if (ind == 1){

        std::vector<double> sim_beta = simulate(M, vars, probs);
        std::vector<double> sim_gam_noise = simulate(M, std::vector<double> {1/gam_prev}, std::vector<double> {1});
        std::vector<double> sim_beta_before = simulate(M, vars_before, probs_before);
        std::vector<double> sim_gam_noise_before = simulate(M, std::vector<double> {1/gam_before}, std::vector<double> {1});

        std::vector<double> r1_den(M, 0.0);
        for (int i=0; i<M; i++)
            r1_den[i] = rho * g1d(sim_beta[i] + sim_gam_noise[i], gam_prev) + (1-rho) * g1d(sim_beta_before[i] + sim_gam_noise_before[i], gam_before);

        double alpha_bar_temp = std::accumulate(r1_den.begin(), r1_den.end(), 0.0);
        MPI_Allreduce(&alpha_bar_temp, &alpha_bar, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        alpha_bar /= Mt;

        eta_bar = gam_prev / alpha_bar;

        gam_bar = eta_bar - gam_prev;
    }
    else if (ind == 2){

        alpha_bar = g2d_onsager(gam_prev, gamw, dataset); // * gam_prev ;

        eta_bar = gam_prev / alpha_bar;

        gam_bar = eta_bar - gam_prev;
    }

    return {alpha_bar, eta_bar, gam_bar};
}


double vamp::g1_bin_class(double p, double tau1, double y){

    double c = p / sqrt(probit_var + 1/tau1);
    double temp = p + exp(-0.5 * c * c) / tau1 / sqrt(2*M_PI) / sqrt(probit_var + 1/tau1) / (0.5 * erfc(-c * M_SQRT1_2));
    if (y==1)
        return temp;
    else if (y==0)
        return p-temp;
    return 0;
}


double vamp::g1d_bin_class(double p, double tau1, double y){
    
    double c = p / sqrt(probit_var + 1/tau1);
    double Nc = exp(-0.5 * c * c) /  sqrt(2*M_PI);
    double phic = 0.5 * erfc(-c * M_SQRT1_2);
    double temp = 1 -  Nc / (1 + tau1*probit_var) / phic * (c + Nc / phic);
    if (y==1)
        return temp;
    else if (y==0)
        return (1/tau1 + p*p) - temp;
    return 0;
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