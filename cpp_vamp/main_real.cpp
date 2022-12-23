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

    int normal = 1;
    std::cout << "normal = " << normal << std::endl;

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

        // reading signal estimate from gmrm software
        //const std::string true_beta_height = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/height_true.txt";
        //std::vector<double> beta_true = read_vec_from_file(true_beta_height, M, S); //for smaller dataset
        
        std::string phenfp = (opt.get_phen_files())[0];
        data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, normal, rank);
        // dataset.read_phen();
        // dataset.read_genotype_data();
        // dataset.compute_markers_statistics();
        

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // running EM-VAMP algorithm on the data
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

        double gam1 = 1e-6;
        double gamw = 2;
        std::vector<double> beta_true = std::vector<double> (M, 0.0);
        vamp emvamp(M, gam1, gamw, beta_true, rank, opt);
        std::vector<double> x_est = emvamp.infere(&dataset);

        std::string xL_file_name = opt.get_bed_file();
        std::string xR_file_name = opt.get_bed_file();
        int pos_dot = xL_file_name.find(".");
        xL_file_name = xL_file_name.substr(0, pos_dot);
        xL_file_name.append("_xL.csv");
        xR_file_name.append("_xR.csv");

        std::vector<double> xL = dataset.get_xL();
        std::vector<double> xR = dataset.get_xR();

        store_vec_to_file(xL_file_name, xL);
        store_vec_to_file(xR_file_name, xR);
    }
    else if (opt.get_run_mode() == "test") // just analyzing the result on the test data
    {   

        normal = 1;
        std::cout << "test normal = " << normal << std::endl;
        // reading test set
        //const std::string bedfp_HTtest = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/testing/bed_files/ukb22828_UKB_EST_v3_all_prunned_080_test.bed";
        //const std::string pheno_HTtest = "/nfs/scistore13/robingrp/human_data/pheno/continuous/ukb_test_HT.phen";
        const std::string bedfp_test = opt.get_bed_file_test();
        const std::string pheno_test = (opt.get_phen_files_test())[0]; // currently it is only supported passing one pheno files as an input argument

        int N_test = opt.get_N_test();
        int Mt_test = opt.get_Mt_test();
        std::vector<double> MS = divide_work(Mt_test);
        int M_test = MS[0];
        int S_test = MS[1];

        data dataset_test(pheno_test, bedfp_test, N_test, M_test, Mt_test, S_test, normal, rank);
        // dataset_test.read_phen();
        // dataset_test.read_genotype_data();
        // dataset_test.compute_markers_statistics();
        
        std::vector<double> y_test = dataset_test.get_phen();

        //std::vector<double> x_est = read_vec_from_file(opt.get_estimate_file() + "_rank_" + std::to_string(rank) + ".bin", M, 0);
        std::string est_file_name = opt.get_estimate_file();
        int pos_dot = est_file_name.find(".");
        std::string end_est_file_name = est_file_name.substr(pos_dot + 1);
        std::cout << "end_est_file_name = " << end_est_file_name << std::endl;

        std::vector<double> x_est;
        if (end_est_file_name == "bin")
            x_est = mpi_read_vec_from_file(est_file_name, M_test, S_test);
        else
            x_est = read_vec_from_file(est_file_name, M_test, S_test);

        if (rank == 0){
            std::cout << "x_est[0] = " << x_est[0] << std::endl;
            std::cout << "x_est[1] = " << x_est[1] << std::endl;
            std::cout << "x_est[2] = " << x_est[2] << std::endl;
            std::cout << "x_est[3] = " << x_est[3] << std::endl;
            std::cout << "x_est[4] = " << x_est[4] << std::endl;
            std::cout << "x_est[5] = " << x_est[5] << std::endl;
        }
        if (normal == 1)
            for (int i0 = 0; i0 < x_est.size(); i0++)
                x_est[i0] *= sqrt( (double) N_test );

        std::vector<double> z_test = dataset_test.Ax(x_est.data(), normal);

        //if (normal == 2){
        //    int N = opt.get_N();
        //    for (int i0 = 0; i0 < z_test.size(); i0++)
        //        z_test[i0] *= sqrt( (double) N_test / (double) N );
        //}

        double l2_pred_err2 = 0;
        for (int i0 = 0; i0 < N_test; i0++){
            // std::cout << "(y_test-z_test)[" << i0 << "] = " << y_test[i0] - z_test[i0] << std::endl;
            l2_pred_err2 += (y_test[i0] - z_test[i0]) * (y_test[i0] - z_test[i0]);
        }  

        double stdev = calc_stdev(y_test);
        if (rank == 0){
            std::cout << "y stdev^2 = " << stdev * stdev << std::endl;  
            std::cout << "test l2 pred err^2 = " << l2_pred_err2 << std::endl;
            std::cout << "test R2 = " << 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() ) << std::endl;
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
        data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, normal, rank);
        // dataset.read_phen();
        // dataset.read_genotype_data();
        // dataset.compute_markers_statistics();

        double gam1 = 1e-6; //, gamw = 2.112692482840060;
        double gamw = 2;
        vamp emvamp(M, gam1, gamw, beta_true, rank, opt);
        std::vector<double> x_est = emvamp.infere(&dataset);

        double intercept = dataset.get_intercept();
        double scale = dataset.get_scale();
        if (rank == 0){
            std::cout << "intercept = " << intercept << std::endl;
            std::cout << "scale = " << scale << std::endl;
        }
                
        int final_it = opt.get_iterations();

        // reading test set
        const std::string bedfp_test = opt.get_bed_file_test();
        const std::string pheno_test = (opt.get_phen_files_test())[0]; // currently it is only supported passing one pheno files as an input argument

        int N_test = opt.get_N_test();
        int Mt_test = opt.get_Mt_test();
        std::vector<double> MS_test = divide_work(Mt);
        int M_test = MS_test[0];
        int S_test = MS_test[1];

        data dataset_test(pheno_test, bedfp_test, N_test, M_test, Mt_test, S, normal, rank);
        // dataset_test.read_phen();
        // dataset_test.read_genotype_data();
        // dataset_test.compute_markers_statistics();

        if (normal == 1)
            for (int i0 = 0; i0 < x_est.size(); i0++)
                x_est[i0] *= sqrt( (double) N_test );

        std::vector<double> z_test = dataset_test.Ax(x_est.data(), normal);

        if (normal == 1)
            for (int i0 = 0; i0 < z_test.size(); i0++)
                z_test[i0] = intercept + scale * z_test[i0];

        std::vector<double> y_test = dataset_test.get_phen();

        double l2_pred_err2 = 0;
        for (int i0 = 0; i0 < N_test; i0++){
            l2_pred_err2 += (y_test[i0] - z_test[i0]) * (y_test[i0] - z_test[i0]);
        }

        double stdev = calc_stdev(y_test);
        if (rank == 0){
            std::cout << std::endl;
            std::cout << "y stdev^2 = " << stdev * stdev << std::endl;  
            std::cout << "test l2 pred err^2 = " << l2_pred_err2 << std::endl;
            std::cout << "test R2 = " << 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() ) << std::endl;
        }
    }

    MPI_Finalize();
    return 0;
}