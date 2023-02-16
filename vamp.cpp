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
#include "dotp_lut.hpp"
#include "vamp.hpp"
#include "data.hpp"
#include "vamp_probit.cpp"
#include "denoiserXXT.cpp"
#include "utilities.hpp"
#include <boost/math/distributions/students_t.hpp> // contains Student's t distribution needed for pvals calculation

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
    reverse = opt.get_use_XXT_denoiser();
    use_lmmse_damp = opt.get_use_lmmse_damp();
    //std::cout<< "CG_max_iter = " << CG_max_iter << std::endl;
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
    use_adap_damp(opt.get_use_adap_damp()),
    rho(opt.get_rho()),
    probs(opt.get_probs()),
    out_dir(opt.get_out_dir()),
    out_name(opt.get_out_name()),
    true_signal(true_signal),
    model(opt.get_model()),
    store_pvals(opt.get_store_pvals()),
    rank(rank),
    reverse(opt.get_use_XXT_denoiser()),
    use_lmmse_damp(opt.get_use_lmmse_damp())  {
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
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
}

//std::vector<double> predict(std::vector<double> est, data* dataset){
//    return (*dataset).Ax(est.data());
//}

std::vector<double> vamp::infere( data* dataset ){

    normal = (*dataset).get_normal_data();
    y = (*dataset).get_phen();
    if (normal == 1)
        for (int i=0; i<vars.size(); i++)
            vars[i] *= N;
    if (use_adap_damp == 1)
        largest_sing_val2 = (*dataset).largest_sing_val2();
    if (reverse == 1)
        (*dataset).compute_people_statistics();
    
    if (!strcmp(model.c_str(), "linear"))
        return infere_linear(dataset);
    else if (!strcmp(model.c_str(), "bin_class"))
        return infere_bin_class(dataset);
    else
        throw "invalid model specification!";

    return std::vector<double> (M, 0.0);
}

std::vector<double> vamp::infere_linear(data* dataset){

    if (rank == 0)
        std::cout << "use_lmmse_damp = " << use_lmmse_damp << std::endl;
    std::vector<double> x1_hat_d(M, 0.0);
    std::vector<double> x1_hat_d_prev(M, 0.0);
    std::vector<double> x1_hat_stored(M, 0.0);
    std::vector<double> r1_prev(M, 0.0);
    std::vector<double> x1_hat_prev(M, 0.0);

    // filtering a phenotype for nans
    std::vector<double> y =  (*dataset).filter_pheno();

    // Gaussian noise start
    r1 = simulate(M, std::vector<double> {1.0/gam1}, std::vector<double> {1});
    //r1 = std::vector<double> (M, 0.0);

    // linear estimator
    //r1 = (*dataset).ATx(y.data(), normal);
    //for (int i0=0; i0<M; i0++)
	//  r1[i0] = r1[i0]*M/N;

    //gam1 = 1 / pow(calc_stdev(r1, 1), 2);
    //std::cout << "gam1 init = " << gam1 << std::endl;

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
        //updatePrior(); -> moved to after iteration
        // if (it == 1)
        //    gam1 = pow(calc_stdev(true_signal), -2); // setting the right gam1 at the beginning
        
        double start_cond_expe= MPI_Wtime();
        for (int i = 0; i < M; i++)
            x1_hat[i] = rho * g1(r1[i], gam1) + (1 - rho) * x1_hat_prev[i];
        //for (int i = 0; i < M; i++)
        //    x1_hat[i] = g1(r1[i], gam1);
        double end_cond_expe= MPI_Wtime();
        if (rank == 0)
            std::cout << "time needed to calculate conditional expectation = " << end_cond_expe - start_cond_expe << " seconds" <<  std::endl;

        double start_z1= MPI_Wtime();
        z1 = (*dataset).Ax(x1_hat.data(), normal);
        double end_z1= MPI_Wtime();
        if (rank == 0)
            std::cout << "time needed to calculate z1 = " << end_z1 - start_z1 << " seconds" <<  std::endl;

        // we start adaptive damping
        if (use_adap_damp == 1 && it > 1){
            double obj_fun_val = vamp_obj_func(eta1, gam1, invQ_bern_vec, bern_vec, vars, probs, dataset);
            double min_obj_fun_val = *std::min_element(obj_fun_vals.begin(), obj_fun_vals.end()); 

            if (rank == 0){
                std::cout << "min_obj_fun_val thr = " << min_obj_fun_val << std::endl;
                std::cout << "obj_fun_val = " << obj_fun_val << std::endl;
            }

            if (it>numb_adap_damp_hist){
                if (obj_fun_val >= min_obj_fun_val)
                    rho = std::min(rho*damp_inc_fact, damp_max);
                else
                    rho = std::max(rho*damp_dec_fact, damp_min);               
                for (int i = 0; i < M; i++)
                    x1_hat[i] = rho * g1(r1[i], gam1) + (1 - rho) * x1_hat_prev[i];
                z1 = (*dataset).Ax(x1_hat.data(), normal);
                obj_fun_vals.erase(obj_fun_vals.begin());  
            }
            obj_fun_vals.push_back(obj_fun_val);
        }

        if (rank == 0)
           std::cout << "rho = " << rho << std::endl;

        double scale;
        if (normal == 1)
            scale = sqrt(N);
        else if (normal == 2)
            scale = 1;

        double start_saving = MPI_Wtime();
        // saving x1_hat
        std::string filepath_out = out_dir + out_name + "_it_" + std::to_string(it) + ".bin";
        int S = (*dataset).get_S();
        for (int i0=0; i0<x1_hat_stored.size(); i0++)
            x1_hat_stored[i0] =  x1_hat[i0] / scale;
        mpi_store_vec_to_file(filepath_out, x1_hat_stored, S, M);
        //store_vec_to_file(filepath_out, x1_hat);

        if (rank == 0)
           std::cout << "filepath_out is " << filepath_out << std::endl;

        if (it % 5 == 0){
            std::string filepath_out_r1 = out_dir + out_name + "_r1_it_" + std::to_string(it) + ".bin";
            std::vector<double> r1_stored = r1;
            for (int i0=0; i0<r1_stored.size(); i0++)
                r1_stored[i0] =  r1[i0] / scale;
            mpi_store_vec_to_file(filepath_out_r1, r1_stored, S, M);
        }
        double end_saving = MPI_Wtime();
        if (rank == 0)
            std::cout << "time needed to save beta1 to an external file = " << end_saving - start_saving << " seconds" <<  std::endl;

        double start_onsager1 = MPI_Wtime();
        // Onsager correction calculation 
        x1_hat_d_prev = x1_hat_d;
        double sum_d = 0;
        for (int i=0; i<M; i++)
        {
            // we have to keep the entire derivative vector so that we could have its previous version in the damping step 
            x1_hat_d[i] = rho * g1d(r1[i], gam1) + (1 - rho) * x1_hat_d_prev[i];
            //x1_hat_d[i] = g1d(r1[i], gam1); 
            sum_d += x1_hat_d[i];
        }

        if (calc_state_evo == 1){
            std::tuple<double, double, double> state_evo_par2 = state_evo(1, gam1, gam_before, probs_before, vars_before, dataset);
            if (rank == 0)
                std::cout << "gam2_bar = " << std::get<2>(state_evo_par2) << std::endl; 
        } 

        MPI_Allreduce(&sum_d, &alpha1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        alpha1 /= Mt;
        if (rank == 0)
            std::cout << "alpha1 = " << alpha1 << std::endl;
        eta1 = gam1 / alpha1;
        gam_before = gam2;
        gam2 = std::min(std::max(eta1 - gam1, gamma_min), gamma_max);
        if (rank == 0){
            std::cout << "eta1 = " << eta1 << std::endl;
            std::cout << "gam2 = " << gam2 << std::endl;
        }
        double end_onsager1 = MPI_Wtime();
        if (rank == 0)
            std::cout << "denoising onsager calculation took " << end_onsager1 - start_onsager1 << " seconds" << std::endl;
        if (use_lmmse_damp == 1)
            r2_prev = r2;
        for (int i = 0; i < M; i++)
            r2[i] = (eta1 * x1_hat[i] - gam1 * r1[i]) / gam2;
        if (use_lmmse_damp == 1){
            double xi = std::min(2 * std::min(alpha1, alpha2), 1.0);
            if (rank == 0)
                std::cout << "xi = " << xi << std::endl;
            if (it == 1)
                xi = 1;
            for (int i = 0; i < M; i++)
                r2[i] = xi * r2[i] + (1-xi) * r2_prev[i];
        }

        // if the true value of the signal is known, we print out the true gam2
        double se_dev = 0;
        for (int i0=0; i0<M; i0++){
            se_dev += (r2[i0] - scale*true_signal[i0])*(r2[i0] - scale*true_signal[i0]);
        }
        double se_dev_total = 0;
        MPI_Allreduce(&se_dev, &se_dev_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "true gam2 = " << Mt / se_dev_total << std::endl;

        
        double start_prior_up = MPI_Wtime();
        // new place for prior update
        updatePrior();
        double end_prior_up = MPI_Wtime();
        if (rank == 0)
            std::cout << "time needed to calculate conditional expectation = " << end_prior_up - start_prior_up << " seconds" <<  std::endl;
    
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


        // running conjugate gradient solver to compute LMMSE
        double start_CG = MPI_Wtime();
        if (reverse == 0){
            //gamw = 1 / ( 1 - inner_prod(z1, z1, 0) );
            std::vector<double> v = (*dataset).ATx(y.data(), normal);
            for (int i = 0; i < M; i++)
                v[i] = gamw * v[i] + gam2 * r2[i];

            if (it == 1){
                // x2_hat = CG_solver(v, std::vector<double>(M, 0.0), v, gamw, dataset); // precond_change!
                x2_hat = precondCG_solver(v, std::vector<double>(M, 0.0), gamw, 1, dataset); // precond_change!
            }
            else{
                // x2_hat = CG_solver(v, mu_CG_last, p_CG_last, gamw, dataset); // precond_change!
                x2_hat = precondCG_solver(v, mu_CG_last, gamw, 1, dataset); // precond_change!
            }
        } 
        else if (reverse == 1){
            if (it == 1)
                mu_CG_last = std::vector<double> (4*(*dataset).get_mbytes(), 0.0);
            x2_hat = lmmse_denoiserAAT(r2, mu_CG_last, dataset);
        }
        
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
        //if (reverse == 0)
            alpha2 = g2d_onsager(gam2, gamw, dataset);
        //else if (reverse == 1)
        //    alpha2 = g2d_onsagerAAT(gam2, gamw, dataset);
        double end_onsager = MPI_Wtime();

        if (rank == 0)
            std::cout << "onsager took "  << end_onsager - start_onsager << " seconds." << std::endl;
        
        if (rank == 0)
            std::cout << "alpha2 = " << alpha2 << std::endl;
        eta2 = gam2 / alpha2;
        gam_before = gam1;
        // gam1 = rho * std::min( std::max( eta2 - gam2, gamma_min ), gamma_max ) + (1 - rho) * gam1;
        //if (rank == 0)
        //    std::cout << "old gam1 = " << std::min( std::max( eta2 - gam2, gamma_min ), gamma_max ) << std::endl;

        // adaptive VAMP
        if (it < 0){
            std::vector<double> r1mx1hat = r1;
            for (int i=0; i<M; i++)
                r1mx1hat[i] -= x1_hat[i];

            std::cout << "l2_norm2(r1mx1hat, 1) / Mt = " << l2_norm2(r1mx1hat, 1) / Mt << std::endl;
            std::cout << "1/eta1 = " << 1/eta1 << std::endl;

            gam1 = std::min( std::max( 1 / ( l2_norm2(r1mx1hat, 1) / Mt + 1/eta1 ), gamma_min ), gamma_max );
        }
        else
        {
            std::vector<double> r1mx1hat = r1;
            for (int i=0; i<M; i++)
                r1mx1hat[i] -= x1_hat[i];
            double gam1ML = 1 / ( l2_norm2(r1mx1hat, 1) / Mt + 1/eta1 );
            if (rank == 0)
                std::cout << "gam1ML = " << gam1ML << std::endl;
            gam1 = std::min( std::max( eta2 - gam2, gamma_min ), gamma_max );
            //if (gam1ML >= 1.05 * gam1)
            //    rho = std::min(rho*damp_inc_fact, damp_max);
            //else if (gam1ML <= 0.95 * gam1)
            //    rho = std::max(rho*damp_dec_fact, damp_min);
        }
            
            
        if (rank == 0)
            std::cout << "gam1 = " << gam1 << std::endl;

        r1_prev = r1;
        for (int i = 0; i < M; i++)
            r1[i] = (eta2 * x2_hat[i] - gam2 * r2[i]) / gam1;

        // if the true value of the signal is known, we print out the true gam1
        double se_dev1 = 0;
        for (int i0=0; i0<M; i0++)
            se_dev1 += (r1[i0]- scale*true_signal[i0])*(r1[i0]- scale*true_signal[i0]);
        double se_dev_total1 = 0;
        MPI_Allreduce(&se_dev1, &se_dev_total1, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "true gam1 = " << Mt / se_dev_total1 << std::endl; 

        // learning a noise precision parameter
        //if (reverse == 0)
            updateNoisePrec(dataset);
        //else if (reverse == 1)
        //    updateNoisePrecAAT(dataset);
   
        // printing out error measures
        err_measures(dataset, 2);
        
        double end_lmmse_step = MPI_Wtime();

        if (rank == 0)
            std::cout << "lmmse step took "  << end_lmmse_step - start_lmmse_step << " seconds." << std::endl;
        
        // stopping criteria
        std::vector<double> x1_hat_diff = x1_hat;
        for (int i0 = 0; i0 < x1_hat_diff.size(); i0++)
            x1_hat_diff[i0] = x1_hat_prev[i0] - x1_hat_diff[i0];

        if (it > 1 && sqrt( l2_norm2(x1_hat_diff, 1) / l2_norm2(x1_hat_prev, 1) ) < stop_criteria_thr){
            if (rank == 0)
                std::cout << "VAMP stopping criteria fulfilled with threshold = " << stop_criteria_thr << "." << std::endl;
            break;
        }
        
        if (rank == 0)
            std::cout << "total iteration time = " << end_denoising - start_denoising + end_lmmse_step - start_lmmse_step << std::endl;
        total_comp_time += end_denoising - start_denoising + end_lmmse_step - start_lmmse_step;
        if (rank == 0)
            std::cout << "total computation time so far = " << total_comp_time << std::endl;
        
        if (rank == 0)
            std::cout << std::endl << std::endl;
    }

    if (store_pvals == 1){
        std::string filepath_out_pvals = out_dir + out_name + "_pvals.bin";
        std::vector<double> pvals = (*dataset).pvals_calc(z1, y, x1_hat, filepath_out_pvals);
        if (rank == 0)
            std::cout << "filepath_out_pvals = " << filepath_out_pvals << std::endl;
    }
    
    return x1_hat_stored;          
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
        bern_vec[i] = (2*bern(rd) - 1) / sqrt(Mt); // Bernoulli variables are sampled independently
    invQ_bern_vec = precondCG_solver(bern_vec, tau, 0, dataset); // precond_change
    double onsager = gam2 * inner_prod(bern_vec, invQ_bern_vec, 1); 
    return onsager;    
}


void vamp::updateNoisePrec(data* dataset){

    /*
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

    std::vector<double> v = (*dataset).ATx(u.data(), normal);
    */

    std::vector<double> temp = (*dataset).Ax(x2_hat.data(), normal);
    // std::vector<double> y = (*dataset).get_phen();

    //filtering for NAs
    std::vector<unsigned char> mask4 = (*dataset).get_mask4();
    for (int j=0;j<(*dataset).get_mbytes();j++)
        for(int k=0;k<4;k++)
            if (4*j+k < N && na_lut[mask4[j] * 4 + k]==0)
                y[4*j + k] = 0;

    for (int i = 0; i < N; i++)  // because length(y) = N
        temp[i] -= y[i];
    
    double temp_norm2 = l2_norm2(temp, 0); 
    //double trace_corr = inner_prod(precondCG_solver(v, gamw, 0, dataset), v, 1); // check this! ad // precond_change
    std::vector<double> trace_corr_vec_N = (*dataset).Ax(invQ_bern_vec.data(), normal);
    std::vector<double> trace_corr_vec_M = (*dataset).ATx(trace_corr_vec_N.data(), normal);
    double trace_corr = inner_prod(bern_vec, trace_corr_vec_M, 1) * Mt;
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
        //double max_sigma = *std::max_element(vars.begin(), vars.end()); // std::max_element returns iterators, not values
        double noise_var = 1 / gam1;
        double lambda = 1 - probs[0];
        std::vector<double> omegas = probs;
        for (int j = 1; j < omegas.size(); j++) // omegas is of length L
            omegas[j] /= lambda;
        //for_each(probs.begin() + 1, probs.end(), [lambda](int &c){ c /= lambda; });
        //omegas = pi_prev(2:end) / lambda;

        //double* beta = new double[M * probs.size()];
                       
        // calculating normalized beta and pin
        int it;
        for (it = 0; it < EM_max_iter; it++){

            double max_sigma = *std::max_element(vars.begin(), vars.end()); // std::max_element returns iterators, not values

            std::vector<double> probs_prev = probs;
            std::vector<double> vars_prev = vars;
            std::vector< std::vector<double> > gammas;
            std::vector< std::vector<double> > beta;
            std::vector<double> pin(M, 0.0);
            std::vector<double> v;

            for (int i = 0; i < M; i++){
                std::vector<double> temp; // of length (L-1)
                std::vector<double> temp_gammas;
                for (int j = 1; j < probs.size(); j++ ){
                    double num = lambda * exp( - pow(r1[i], 2) / 2 * (max_sigma - vars[j]) / (vars[j] + noise_var) / (max_sigma + noise_var) ) / sqrt(vars[j] + noise_var) / sqrt(2 * M_PI) * omegas[j];
                    //if (i > 1998){
                    //    std::cout << "(vars[" << j << "] + noise_var) / (max_sigma + noise_var) = " << (vars[j] + noise_var) * (max_sigma + noise_var) << std::endl;
                    //    std::cout << "- pow(r1[i], 2) = " << - pow(r1[i], 2) << std::endl;
                    //    std::cout << "(max_sigma - vars[j]) = " << (max_sigma - vars[j])  << std::endl;
                    //    std::cout << "- pow(r1[i], 2) / 2 * (max_sigma - vars[j]) / (vars[j] + noise_var) / (max_sigma + noise_var)  = " << - pow(r1[i], 2) / 2 * (max_sigma - vars[j]) / (vars[j] + noise_var) / (max_sigma + noise_var)  << std::endl;
                    //    std::cout << "num = " << num << std::endl;
                    //}
                    double num_gammas = gam1 * r1[i] / ( 1 / vars[j] + gam1 );  
                    //std::cout << "num_gammas = " << num_gammas << std::endl;                  
                    temp.push_back(num);
                    temp_gammas.push_back(num_gammas);
                }
                double sum_of_elems = std::accumulate(temp.begin(), temp.end(), decltype(temp)::value_type(0));
                //if (i > 1990)
                //    std::cout << "sum_of_elems = " << sum_of_elems << std::endl;
                //for_each(temp.begin(), temp.end(), [sum_of_elems](int &c){ c /= sum_of_elems; });
                for (int j = 0; j < temp.size(); j++ ){
                    temp[j] /= sum_of_elems;
                }
                beta.push_back(temp);
                gammas.push_back(temp_gammas);
                pin[i] = 1 / ( 1 + (1-lambda) / sqrt(2 * M_PI * noise_var) * exp( - pow(r1[i], 2) / 2 * max_sigma / noise_var / (noise_var + max_sigma) ) / sum_of_elems );
                //std::cout << "pin[" << i << "] = " << pin[i] << std::endl;
            } 
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
                for (int j = 0; j < (beta[0]).size(); j++ ){
                    gammas[i][j] = beta[i][j] * ( pow(gammas[i][j], 2) + v[j] );
                }
            }

            //double sum_of_pin = std::accumulate(pin.begin(), pin.end(), decltype(pin)::value_type(0));
            double sum_of_pin = accumulate(pin.begin(), pin.end(), 0.0);

            double sum_of_pin_total = 0;
            MPI_Allreduce(&sum_of_pin, &sum_of_pin_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            sum_of_pin = sum_of_pin_total;

            //std::cout << "size of beta[0] = " << (beta[0]).size() << std::endl;
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

                //if (rank==0 && it%10 == 0)
                //    std::cout << "j=" << j+1 << ", res_gammas_total = " << res_gammas_total << ", res_total = " << res_total << ", sum_of_pin =" << sum_of_pin << std::endl;
                vars[j+1] = res_gammas_total / res_total;
                //std::cout << "vars[" << j+1 << "] = "<< vars[j+1] << std::endl;
                omegas[j+1] = res_total / sum_of_pin;
                probs[j+1] = lambda * omegas[j+1];

                //if (rank == 0)
                //    std::cout << " j = " << j << ", vars[j+1] = " << vars[j+1] << ", probs[j+1] = " << probs[j+1] << std::endl;
            }
            probs[0] = 1 - lambda;
        
            double distance_probs = 0, norm_probs = 0;
            double distance_vars = 0, norm_vars = 0;
            for (int j = 0; j < probs.size(); j++){
                distance_probs += ( probs[j] - probs_prev[j] ) * ( probs[j] - probs_prev[j] );
                norm_probs += probs[j] * probs[j];
                distance_vars += ( vars[j] - vars_prev[j] ) * ( vars[j] - vars_prev[j] );
                norm_vars += vars[j] * vars[j];
            }
            double dist_probs = sqrt(distance_probs / norm_probs);
            double dist_vars = sqrt(distance_vars / norm_vars);
            if (rank == 0)
                std::cout << "it = " << it << ": dist_probs = " << dist_probs << " & dist_vars = " << dist_vars << std::endl;
            if ( dist_probs < EM_err_thr  && dist_vars < EM_err_thr ){
                if (rank == 0)
                    std::cout << "EM error threshold satisfied." << std::endl;
                break;   
            }
        }
        //else
            //if (rank ==0)
                //std::cout << "distance / norm_probs = " << distance / norm_probs << std::endl;
        //} 
        if (rank == 0)  
            std::cout << "Final number of prior EM iterations = " << it + 1 << std::endl;
}

std::vector<double> vamp::lmmse_mult(std::vector<double> v, double tau, data* dataset){ // multiplying with (tau*A^TAv + gam2*v)

    // std::cout << "normal in lmmse_mult = " << normal << std::endl;
    if (v == std::vector<double>(M, 0.0))
        return std::vector<double>(M, 0.0);
    std::vector<double> res(M, 0.0);
    size_t phen_size = 4 * (*dataset).get_mbytes();
    std::vector<double> res_temp(phen_size, 0.0);
    res_temp = (*dataset).Ax(v.data(), normal);
    res = (*dataset).ATx(res_temp.data(), normal);
    for (int i = 0; i < M; i++){
        res[i] *= tau;
        res[i] += gam2 * v[i];
    }
    return res;
}


std::vector<double> vamp::CG_solver(std::vector<double> v, double tau, data* dataset){
    // we start with approximation x0 = 0
    std::vector<double> mu(M, 0.0);
    return CG_solver(v, mu, v, tau, dataset);
}

std::vector<double> vamp::CG_solver(std::vector<double> v, std::vector<double> mu_start, std::vector<double> p_start, double tau, data* dataset){

    std::vector<double> mu = mu_start;
    std::vector<double> p = p_start, d;
    std::vector<double> r = lmmse_mult(mu, tau, dataset);
    for (int i0=0; i0<r.size(); i0++)
        r[i0] = v[i0] - r[i0];
    std::vector<double> p_temp(M, 0.0);
    std::vector<double> palpha(M, 0.0);
    double alpha, beta;
    for (int i = 0; i < CG_max_iter; i++){
        double start_lmmse = MPI_Wtime();
        // d = A*p
        d = lmmse_mult(p, tau, dataset);
        double end_lmmse = MPI_Wtime();
        alpha = l2_norm2(r, 1) / inner_prod(d, p, 1);
        
        for (int j = 0; j < M; j++)
            palpha[j] = alpha * p[j];
        std::transform (mu.begin(), mu.end(), palpha.begin(), mu.begin(), std::plus<double>());
        for (int j = 0; j < p.size(); j++)
            p_temp[j] = d[j] * alpha;
        beta = pow( l2_norm2(r, 1), -1 );
        std::transform (r.begin(), r.end(), p_temp.begin(), r.begin(), std::minus<double>());
        double l2_norm2_r = l2_norm2(r, 1);
        beta *= l2_norm2_r;
        for (int j = 0; j < p.size(); j++)
            p[j] = r[j] + beta * p[j];

        // stopping criteria
        double rel_err = sqrt( l2_norm2_r / l2_norm2(v, 1) );
        double err_tol = 1e-4;
        if (rank == 0)
            std::cout << "[CG] it = " << i << ": ||r_it|| / ||RHS|| = " << rel_err << std::endl;
        if (rel_err < err_tol) 
            break;
    }

    mu_CG_last = mu;
    p_CG_last = p;
    return mu;
 }

std::vector<double> vamp::precondCG_solver(std::vector<double> v, double tau, int save, data* dataset){
    // we start with approximation x0 = 0
    std::vector<double> mu(M, 0.0);
    return precondCG_solver(v, mu, tau, save, dataset);
}

std::vector<double> vamp::precondCG_solver(std::vector<double> v, std::vector<double> mu_start, double tau, int save, data* dataset){
    // tau = gamw
    // preconditioning part
    std::vector<double> diag(M, 1.0);
    if (normal == 1){
        //for (int j=0; j<M; j++)
        //    diag[j] = tau * ((N-1) / msig[j] / msig[j] + mave[j] * mave[j] * N) + gam2;
        for (int j=0; j<M; j++)
            diag[j] = tau * (N-1)/N + gam2;
    }
    if (normal == 2){
        // not complete!
        std::vector<double> xR, xL;
        xL = (*dataset).get_xL();
        xR = (*dataset).get_xR();
        for (int j=0; j<M; j++)
            diag[j] *= xL[j] * xR[j];
    }
         
    std::vector<double> mu = mu_start;
    std::vector<double> d;
    std::vector<double> r = lmmse_mult(mu, tau, dataset);
    for (int i0=0; i0<r.size(); i0++)
        r[i0] = v[i0] - r[i0];
    std::vector<double> z(M, 0.0);
    for (int j=0; j<M; j++)
        z[j] = r[j] / diag[j];
    std::vector<double> p = z;
    std::vector<double> p_temp(M, 0.0);
    std::vector<double> palpha(M, 0.0);
    double alpha, beta;
    for (int i = 0; i < CG_max_iter; i++){
        double start_lmmse = MPI_Wtime();
        // d = A*p
        d = lmmse_mult(p, tau, dataset);
        double end_lmmse = MPI_Wtime();
        alpha = inner_prod(r, z, 1) / inner_prod(d, p, 1);
        
        for (int j = 0; j < M; j++)
            palpha[j] = alpha * p[j];
        std::transform (mu.begin(), mu.end(), palpha.begin(), mu.begin(), std::plus<double>());
        for (int j = 0; j < p.size(); j++)
            p_temp[j] = d[j] * alpha;
        beta = pow(inner_prod(r, z, 1), -1);
        std::transform (r.begin(), r.end(), p_temp.begin(), r.begin(), std::minus<double>());
        for (int j=0; j<M; j++)
            z[j] = r[j] / diag[j];
        beta *= inner_prod(r, z, 1);
        for (int j = 0; j < p.size(); j++)
            p[j] = z[j] + beta * p[j];

        // stopping criteria
        double rel_err = sqrt( l2_norm2(r, 1) / l2_norm2(v, 1) );
        double norm_mu = sqrt( l2_norm2(mu, 1) );
        double err_tol = 1e-4;
        if (rank == 0)
            std::cout << "[CG] it = " << i << ": ||r_it|| / ||RHS|| = " << rel_err << ", ||x_it|| = " << norm_mu << std::endl;
        if (rel_err < err_tol) 
            break;
    }
    if (save == 1){
        mu_CG_last = mu;
        p_CG_last = p;
    }
    return mu;
 }


void vamp::err_measures(data *dataset, int ind){

    double scale; // = 1.0 / (double) N;

    if (normal == 1)
        scale = 1.0 / (double) N;
    else if (normal == 2)
        scale = 1.0;
    
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
    double l2_norm2_xhat;
    if (ind == 1){
        for (int i = 0; i< M; i++)
            temp[i] = sqrt(scale) * x1_hat[i] - true_signal[i];
        l2_norm2_xhat = l2_norm2(x1_hat, 1) * scale;
    }
    else if (ind == 2){
        for (int i = 0; i< M; i++)
            temp[i] = sqrt(scale) * x2_hat[i] - true_signal[i];
        l2_norm2_xhat = l2_norm2(x2_hat, 1) * scale;
    }

    //double l2_signal_err = sqrt( l2_norm2(temp, 1) / l2_norm2_xhat );
    //double l2_true_signal2 = l2_norm2(true_signal, 1);
    double l2_signal_err = sqrt( l2_norm2(temp, 1) / l2_norm2(true_signal, 1) );
    if (rank == 0){
        std::cout << "l2 signal error = " << l2_signal_err << std::endl;
        //std::cout << sqrt( l2_norm2_xhat ) << std::endl;
        //std::cout << sqrt( l2_true_signal2 ) << std::endl;
    }
    
    // l2 prediction error
    size_t phen_size = 4 * (*dataset).get_mbytes();
    std::vector<double> tempNest(phen_size, 0.0);
    std::vector<double> tempNtrue(phen_size, 0.0);
    // std::vector<double> y = (*dataset).get_phen();

    //filtering pheno vector for NAs
    std::vector<unsigned char> mask4 = (*dataset).get_mask4();
    for (int j=0;j<(*dataset).get_mbytes();j++)
        for(int k=0;k<4;k++)
            if (4*j+k < N && na_lut[mask4[j] * 4 + k]==0)
                y[4*j + k] = 0;

    std::vector<double> Axest;
    if (ind == 1)
        //Axest = (*dataset).Ax( x1_hat.data() );
        Axest = z1;
    else if (ind == 2)
       Axest = (*dataset).Ax(x2_hat.data(), normal);
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

double vamp::vamp_obj_func(double eta, double gam, std::vector<double> invQu, std::vector<double> bernu, std::vector<double> vars, std::vector<double> pi, data* dataset){
  
    std::random_device rd;
    std::bernoulli_distribution bern(0.5);
    double alpha = eta2/eta1;
    double lambda1 = 2 * alpha * (gamw * largest_sing_val2 + gam2); 
    // if (rank == 0)
    //    std::cout << "lambda1 = " << lambda1 << std::endl;
    // int verbose_obj_func = 1;

    // calculating differential entropy of error distribution q
    // double Hq = Mt / 2 * (1 + log(2 * M_PI / eta));
    double Hq = - Mt / 2 * log(eta);
    //if (verbose_obj_func == 1)
    //   std::cout << "[vamp_obj_func] Hq = " << Hq / Mt << std::endl;

    // calculating KL divergence between two multivariate gaussians
    double DKL2 = 0;

    // Taylor approximation of a log(det) component
    // DKL2 += - Mt*log(gamw * (*dataset).get_sigma_max() * (*dataset).get_sigma_max() + gam2);
    std::vector<double> u(M, 0.0);
    for (int i = 0; i < M; i++)
        u[i] = 2*bern(rd) - 1;
    std::vector<double> temp_poly_apr = u;
    double sign = 1;
    for (int i=1; i<=poly_deg_apr_vamp_obj; i++){
        temp_poly_apr = lmmse_mult(temp_poly_apr, gamw, dataset); // this function (vamp_obj_func) is hardcoded for the linear model
        for (int i1=0; i1<M; i1++)
            temp_poly_apr[i1] *= alpha / lambda1;
        for (int i1=0; i1<M; i1++)
            temp_poly_apr[i1] -= 1;
        // DKL2 += sign * inner_prod(temp_poly_apr, u, 1) / (gamw * (*dataset).get_sigma_max() * (*dataset).get_sigma_max() + gam2) / i;
        DKL2 += sign * inner_prod(temp_poly_apr, u, 1) / i;
        sign *= -1;
    }
    DKL2 += Mt * log(lambda1);

    
    // trace component
    //std::vector<double> u = std::vector<double> (M, 0.0);
    //for (int i = 0; i < M; i++)
    //    u[i] = 2*bern(rd) - 1;
    std::vector<double> temp = (*dataset).Ax(invQu.data(), 1);
    std::vector<double> temp2 = (*dataset).ATx(temp.data(), 1);
    // for (int i=0; i<temp2.size(); i++)
    //    temp2[i] *= gamw;
    DKL2 += gamw * inner_prod(bernu, temp2, 1) / 2;

    // quadratic form component
    //std::vector<double> z2 = (*dataset).Ax(x2_hat.data());
    std::vector<double> y = (*dataset).filter_pheno();
    // DKL2 += (-2) * gamw * inner_prod(y, z1, 0);
    DKL2 += (-1) * gamw * alpha * inner_prod(y, z1, 0);
    DKL2 += gamw * alpha / 2 * inner_prod(x1_hat, (*dataset).ATx(z1.data(), 1), 1);
    DKL2 += inner_prod(y,y,0) * gamw * alpha / 2;
    DKL2 += (-1) * Mt / 2 * log(gamw * alpha);

    /*
    if (verbose_obj_func == 1)
       if (rank == 0)
            std::cout << "[vamp_obj_func] DKL2 = " << DKL2 / Mt << std::endl;
    */
    //for (int i = 0; i < M; i++)
        //u[i] = 2*bern(rd) - 1;
    //DKL2 -= inner_prod(u, CG_solver(u, gamw, dataset), 1);
    //DKL2 -= Mt / eta2;

    // KL divergence between two gaussian mixtures
    
    double DKL3 = 0;

    int max_iter_obj_MC=1e4;
    std::vector<double> vars_plus_gam = vars;
    for (int i=0; i<vars_plus_gam.size(); i++)
        vars_plus_gam[i] = vars[i] + 1/gam;

    std::vector<double> vars_wo0 = vars;
    std::vector<double> pi_wo0 = pi;
    vars_wo0.erase(vars_wo0.begin());
    pi_wo0.erase(pi_wo0.begin());
    for (int it_MC=0; it_MC<max_iter_obj_MC; it_MC++){
        double x = generate_mixture_gaussians(vars_plus_gam.size(), vars_plus_gam, pi);
        double log_mix_gauss_ratio = log_mix_gauss_pdf_ratio(x, vars_plus_gam, vars_wo0, pi, pi_wo0);
        DKL3 += log_mix_gauss_ratio;
    }    
    DKL3 /= max_iter_obj_MC;
    /*
    if (verbose_obj_func == 1)
        if (rank == 0)
            std::cout << "[vamp_obj_func] DKL3 = " << DKL3 / Mt << std::endl;
    */

    return (Hq + DKL2 + DKL3) / Mt;

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


// finding p-values from t-test on regression coefficient = 0 
// in leave-one-out setting, i.e. y - A_{-k}x_{-k} = alpha_k * x_k, H0: alpha_k = 0
/*
std::vector<double> vamp::pvals_calc(data* dataset){
    std::vector<double> pvals(M, 0.0);
    std::vector<double> y_mod = y;
    for (int i=0; i<N; i++)
        y_mod[i] -= z1[i];
    double df = N-2;
    boost::math::students_t tdist(df);
    std::string type_data = (*dataset).get_type_data();
    if (type_data == "bed"){
        for (int k=0; k<M; k++){
            std::vector<double> y_mark = y_mod;
            int mbytes = (*dataset).get_mbytes();
            unsigned char* bed_data = (*dataset).get_bed_data();
            unsigned char* bed = &bed_data[k * mbytes];

            double sumx = 0, sumsqx = 0, sumxy = 0;
            for (int i=0; i<mbytes; i++) {
                #ifdef _OPENMP
                #pragma omp simd aligned(dotp_lut_a,dotp_lut_b:32) reduction(+:sumx, sumsqx, sumxy)
                #endif
                for (int j=0; j<4; j++) {
                    y_mark[4*i+j] += dotp_lut_a[bed[i] * 4 + j] * x1_hat[k];
                    sumx += dotp_lut_a[bed[i] * 4 + j];
                    sumsqx += dotp_lut_a[bed[i] * 4 + j]*dotp_lut_a[bed[i] * 4 + j];
                    sumxy += dotp_lut_a[bed[i] * 4 + j]*y_mark[4*i+j];
                }
            }
            pvals[k] = linear_reg1d_pvals(sumx, sumsqx, sumxy, y_mark);
        }  
    }
    else if (type_data == "meth"){
        for (int k=0; k<M; k++){
            std::vector<double> y_mark = y_mod;
            double* meth_data = (*dataset).get_meth_data();
            double* meth = &meth_data[k * N];

            double sumx = 0, sumsqx = 0, sumxy = 0;
            #ifdef _OPENMP
                #pragma omp simd
            #endif
            for (int i=0; i<N; i++) 
                y_mark[i] += meth[i] * x1_hat[k];
            
            #ifdef _OPENMP
                #pragma omp simd reduction(+:sumx, sumsqx, sumxy)
            #endif
            for (int i=0; i<N; i++) {
                sumx += meth[i];
                sumsqx += meth[i]*meth[i];
                sumxy += meth[i]*y_mark[i];
            }

            sumx = 0.0;
            for (int i=0; i<N; i++) {
                sumx += meth[i];
            }
            pvals[k] = linear_reg1d_pvals(sumx, sumsqx, sumxy, y_mark);
        }
    }
    return pvals;
}
*/