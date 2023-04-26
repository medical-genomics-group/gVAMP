#include <mpi.h>
#include <iostream>
#include <cmath> // contains definition of ceil
#include <bits/stdc++.h>  // contains definition of INT_MAX
#include <immintrin.h> // contains definition of _mm_malloc
#include <numeric>
#include "utilities.hpp"
#include "data.hpp"
#include "vamp.hpp"
#include "options.hpp"


int main(int argc, char** argv)
{

    // starting parallel processes
    int required_MPI_level = MPI_THREAD_MULTIPLE;
    int provided_MPI_level;
    MPI_Init_thread(NULL, NULL, required_MPI_level, &provided_MPI_level);

    const Options opt(argc, argv);

    // retrieving MPI specific information
    int rank = 0;
    int nranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);


    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // reading genotype data / phenotype file
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    if (opt.get_run_mode() == "infere"){ 


        //%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // setting blocks of markers 
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%

        size_t Mt = opt.get_Mt();
        size_t N = opt.get_N();

        std::vector<double> MS = divide_work(Mt);
        int M = MS[0];
        int S = MS[1];
        int Mm = MS[2];
        
        std::string phenfp = (opt.get_phen_files())[0];
        std::string type_data = "bed";

        data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, rank, type_data);

        dataset.read_covariates(opt.get_cov_file(), opt.get_C());


        

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // running EM-VAMP algorithm on the data
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        double gam1 = 1e-8;
        
        std::vector<double> beta_true = std::vector<double> (M, 0.0);

        vamp emvamp(M, gam1, 1, beta_true, rank, opt);

        std::vector<double> x_est = emvamp.infere(&dataset);

    }
    else if (opt.get_run_mode() == "test") // just analyzing the result on the test data
    {   

        // reading test set
        const std::string bedfp_test = opt.get_bed_file_test();
        const std::string pheno_test = (opt.get_phen_files_test())[0]; // currently it is only supported passing one pheno files as an input argument

        int N_test = opt.get_N_test();
        int Mt_test = opt.get_Mt_test();
        std::vector<double> MS = divide_work(Mt_test);
        int M_test = MS[0];
        int S_test = MS[1];

        data dataset_test(pheno_test, bedfp_test, N_test, M_test, Mt_test, S_test, rank);
        
        std::vector<double> y_test = dataset_test.get_phen();

        std::string est_file_name = opt.get_estimate_file();
        int pos_dot = est_file_name.find(".");
        std::string end_est_file_name = est_file_name.substr(pos_dot + 1);
        if (rank == 0)
            std::cout << "est_file_name = " << est_file_name << std::endl;

        int pos_it = est_file_name.find("it");
        std::vector<int> iter_range = opt.get_test_iter_range();
        int min_it = iter_range[0];
        int max_it = iter_range[1];
        if (rank == 0)
            std::cout << "iter range = [" << min_it << ", " << max_it << "]" << std::endl;

        if (min_it != -1){
            for (int it = min_it; it <= max_it; it++){
                std::vector<double> x_est;
                std::string est_file_name_it = est_file_name.substr(0, pos_it) + "it_" + std::to_string(it) + "." + end_est_file_name;

                if (end_est_file_name == "bin")
                    x_est = mpi_read_vec_from_file(est_file_name_it, M_test, S_test);
                else
                    x_est = read_vec_from_file(est_file_name_it, M_test, S_test);

                for (int i0 = 0; i0 < x_est.size(); i0++)
                    x_est[i0] *= sqrt( (double) N_test );

                std::vector<double> z_test = dataset_test.Ax(x_est.data());
                            
                if (opt.get_cov_file() != ""){

                    int C = opt.get_C();

                    std::vector<double> cov_effect = read_vec_from_file(opt.get_cov_estimate_file(), C, 0);

                    dataset_test.read_covariates(opt.get_cov_file(), C);
            
                    std::vector<double> Zx_temp = dataset_test.Zx(cov_effect);

                    std::transform (z_test.begin(), z_test.end(), Zx_temp.begin(), z_test.begin(), std::plus<double>());

                }

                for (int i=0; i<N_test; i++){

                    double prob = normal_cdf(z_test[i]);

                    if (prob >= 0.5)
                        z_test[i] = 1;
                    else 
                        z_test[i] = 0;
                } 

                int N = 0, P = 0, TP = 0, FP = 0;
                for (int i = 0; i < N_test; i++){
                    if (y_test[i] == 1){
                        P++;
                        if (z_test[i] == 1)
                            TP++;
                    }
                    else{
                        N++;
                        if (z_test[i] == 1)
                            FP++;
                    }
                }  

                if (rank == 0){
                    std::cout <<  "P = " << P << ", " << "N = " << N << ", TPR = " << (double) TP / P << ", FPR = " << (double) FP / N <<  ", ";
                }
            }
        }
        else 
        {
            std::vector<double> x_est;
            if (rank == 0)
                std::cout << "est_file_name = " << est_file_name << std::endl;
            if (end_est_file_name == "bin")
                x_est = mpi_read_vec_from_file(est_file_name, M_test, S_test);
            else
                x_est = read_vec_from_file(est_file_name, M_test, S_test);

            for (int i0 = 0; i0 < x_est.size(); i0++)
                x_est[i0] *= sqrt( (double) N_test );

            std::vector<double> z_test = dataset_test.Ax(x_est.data());

            if (opt.get_cov_file() != ""){

                int C = opt.get_C();

                std::vector<double> cov_effect = read_vec_from_file(opt.get_cov_estimate_file(), C, 0);

                dataset_test.read_covariates(opt.get_cov_file(), C);
        
                std::vector<double> Zx_temp = dataset_test.Zx(cov_effect);

                std::transform (z_test.begin(), z_test.end(), Zx_temp.begin(), z_test.begin(), std::plus<double>());

            }

            for (int i=0; i<N_test; i++){

                double prob = normal_cdf(z_test[i]);

                if (prob >= 0.5)
                    z_test[i] = 1;
                else 
                    z_test[i] = 0;
            } 

            int Ne = 0, P = 0, TP = 0, FP = 0;
            for (int i = 0; i < N_test; i++){
                if (y_test[i] == 1){
                    P++;
                    if (z_test[i] == 1)
                        TP++;
                }
                else{
                    Ne++;
                    if (z_test[i] == 1)
                        FP++;
                }
            }  

            if (rank == 0){
                std::cout <<  "P = " << P << ", " << "N = " << Ne << ", TPR = " << (double) TP / P << ", FPR = " << (double) FP / Ne <<  ", ";
            }

        }
        
    }
    else if (opt.get_run_mode() == "both")
    {

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // setting blocks of markers 
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%

        size_t Mt = opt.get_Mt();
        size_t N = opt.get_N();

        std::vector<double> MS = divide_work(Mt);
        int M = MS[0];
        int S = MS[1];
        int Mm = MS[2];

        //const std::string true_beta_height = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/height_true.txt";
        //std::vector<double> beta_true = read_vec_from_file(true_beta_height, M, S);
  
        std::vector<double> beta_true = std::vector<double> (M, 0.0);
        std::string phenfp = (opt.get_phen_files())[0]; // currently only one phenotype file is supported
        data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, rank);

        dataset.read_covariates(opt.get_cov_file(), opt.get_C());

        double gam1 = 1e-8; 

        vamp emvamp(M, gam1, 1, beta_true, rank, opt);
        std::vector<double> x_est = emvamp.infere(&dataset);
                
        int final_it = opt.get_iterations();

        // reading test set
        const std::string bedfp_test = opt.get_bed_file_test();
        const std::string pheno_test = (opt.get_phen_files_test())[0]; // currently it is only supported passing one pheno files as an input argument

        int N_test = opt.get_N_test();
        int Mt_test = opt.get_Mt_test();
        std::vector<double> MS_test = divide_work(Mt);
        int M_test = MS_test[0];
        int S_test = MS_test[1];

        data dataset_test(pheno_test, bedfp_test, N_test, M_test, Mt_test, S, rank);

        for (int i0 = 0; i0 < x_est.size(); i0++)
            x_est[i0] *= sqrt( (double) N_test );

        std::vector<double> z_test = dataset_test.Ax(x_est.data());

        std::vector<double> y_test = dataset_test.get_phen();

        if (opt.get_cov_file() != ""){

                int C = opt.get_C();

                std::vector<double> cov_effect = emvamp.get_cov_eff();
        
                std::vector<double> Zx_temp = dataset.Zx(cov_effect);

                std::transform (z_test.begin(), z_test.end(), Zx_temp.begin(), z_test.begin(), std::plus<double>());

            }

            for (int i=0; i<N_test; i++){

                double prob = normal_cdf(z_test[i]);

                if (prob >= 0.5)
                    z_test[i] = 1;
                else 
                    z_test[i] = 0;
            } 

            int Ne = 0, P = 0, TP = 0, FP = 0;
            for (int i = 0; i < N_test; i++){
                if (y_test[i] == 1){
                    P++;
                    if (z_test[i] == 1)
                        TP++;
                }
                else{
                    Ne++;
                    if (z_test[i] == 1)
                        FP++;
                }
            }  

            if (rank == 0){
                std::cout <<  "P = " << P << ", " << "N = " << Ne << ", TPR = " << (double) TP / P << ", FPR = " << (double) FP / Ne <<  ", ";
            }
    }

    MPI_Finalize();
    return 0;
}