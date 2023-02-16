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
        std::vector<unsigned char> mask4 = (*dataset).get_mask4();
        for (int j=0;j<(*dataset).get_mbytes();j++)
            for(int k=0;k<4;k++)
                if (4*j+k < N && na_lut[mask4[j] * 4 + k]==0)
                    y[4*j + k] = 0;

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

        /*
        if (it > 1 && sqrt( l2_norm2(x1_hat_diff, 1) / l2_norm2(x1_hat_prev, 1) ) < stop_criteria_thr){
            if (rank == 0)
                std::cout << "stopping criteria fulfilled" << std::endl;
            //break;
        }
        */
    }
    
    return x1_hat;

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