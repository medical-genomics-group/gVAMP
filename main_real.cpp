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

        // reading signal estimate from gmrm software
        //const std::string true_beta_height = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/height_true.txt";
        //std::vector<double> beta_true = read_vec_from_file(true_beta_height, M, S); //for smaller dataset
        
        std::string phenfp = (opt.get_phen_files())[0];
        std::string type_data = "bed";
        double alpha_scale = opt.get_alpha_scale();
        std::string bimfp = opt.get_bim_file();
        data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, rank, type_data, alpha_scale, bimfp);
        // dataset.read_phen();
        // dataset.read_genotype_data();
        // dataset.compute_markers_statistics();
        

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // running EM-VAMP algorithm on the data
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    

        double gam1 = 1e-6;
        double gamw;
        if (opt.get_h2() == -1)
            gamw = 2;
        else 
            gamw = 1.0 / (1.0 - opt.get_h2());
        // gamw = 1.0 / (1.0 - 0.57);
        
        std::vector<double> beta_true = std::vector<double> (M, 0.0);
        vamp emvamp(M, gam1, gamw, beta_true, rank, opt);

        /*
        std::vector<double> vec(M, 1.0);
        emvamp.set_LBglob((N/4)*4/5);
        emvamp.set_LBglob((N/4)/10);
        emvamp.set_SBglob(0);

        std::vector<double> lm1 =  emvamp.lmmse_mult(vec, 1, &dataset, 0);
        std::vector<double> lm2 =  emvamp.lmmse_mult(vec, 1, &dataset, 1);

        double std1 = calc_stdev(lm1,1);
        double std2 = calc_stdev(lm2,1);

        if (rank == 0)
            std::cout << "std1 = " << std1 << ", std2 = " << std2 << std::endl;
        
        std::vector<double> diff = lm1;
        for (int i=0; i<M; i++)
            diff[i] -= lm2[i];
        
        double l2norm2_diff = l2_norm2(diff, 1);
        double l2norm2_true = l2_norm2(lm1, 1);

        if (rank == 0)
            std::cout << "sqrt( l2norm2_diff / l2norm2_true ) = " << sqrt( l2norm2_diff / l2norm2_true ) << std::endl; 

        std::vector<double> xhat = emvamp.precondCG_solver(vec, std::vector<double>(M, 0.0), 2, 1, &dataset, 0);

        std::vector<double> xhat_red = emvamp.precondCG_solver(vec, std::vector<double>(M, 0.0), 2, 1, &dataset, 1);

        double stdx = calc_stdev(xhat,1);
        double stdx_red = calc_stdev(xhat_red,1);

        if (rank == 0)
            std::cout << "stdx = " << stdx << ", stdx_red = " << stdx_red << std::endl;

        
        std::vector<double> diff_x = xhat;
        for (int i=0; i<M; i++)
            diff_x[i] -= xhat_red[i];

        double l2norm2_diff_x = l2_norm2(diff_x, 1);
        double l2norm2_true_x = l2_norm2(xhat, 1);

        if (rank == 0)
            std::cout << "sqrt( l2norm2_diff_x / l2norm2_true_x ) = " << sqrt( l2norm2_diff_x / l2norm2_true_x ) << std::endl; 

        */

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
        std::string type_data = "bed";
        double alpha_scale = opt.get_alpha_scale();
        std::string bimfp = opt.get_bim_file();

        data dataset_test(pheno_test, bedfp_test, N_test, M_test, Mt_test, S_test, rank, type_data, alpha_scale, bimfp);
        // dataset_test.read_phen();
        // dataset_test.read_genotype_data();
        // dataset_test.compute_markers_statistics();
        
        std::vector<double> y_test = dataset_test.get_phen();

        //std::vector<double> x_est = read_vec_from_file(opt.get_estimate_file() + "_rank_" + std::to_string(rank) + ".bin", M, 0);
        std::string est_file_name = opt.get_estimate_file();
        int pos_dot = est_file_name.find(".");
        std::string end_est_file_name = est_file_name.substr(pos_dot + 1);
        //if (rank == 0)
        //    std::cout << "est_file_name = " << est_file_name << std::endl;

        int pos_it = est_file_name.rfind("it");
        std::vector<int> iter_range = opt.get_test_iter_range();
        int min_it = iter_range[0];
        int max_it = iter_range[1];
        if (rank == 0)
            std::cout << "iter range = [" << min_it << ", " << max_it << "]" << std::endl;

        double maxR2 = -1;
        int maxind = -1;

        if (min_it != -1){
            for (int it = min_it; it <= max_it; it++){
                std::vector<double> x_est;
                std::string est_file_name_it = est_file_name.substr(0, pos_it) + "it_" + std::to_string(it) + "." + end_est_file_name;
                //if (rank == 0)
                    //std::cout << "est_file_name_it = " << est_file_name_it << std::endl;
                if (end_est_file_name == "bin")
                    x_est = mpi_read_vec_from_file(est_file_name_it, M_test, S_test);
                else
                    x_est = read_vec_from_file(est_file_name_it, M_test, S_test);

                for (int i0 = 0; i0 < x_est.size(); i0++)
                    x_est[i0] *= sqrt( (double) N_test );
                
                std::vector<double> z_test = dataset_test.Ax(x_est.data());

                double l2_pred_err2 = 0;
                for (int i0 = 0; i0 < N_test; i0++){
                    // std::cout << "(y_test-z_test)[" << i0 << "] = " << y_test[i0] - z_test[i0] << std::endl;
                    l2_pred_err2 += (y_test[i0] - z_test[i0]) * (y_test[i0] - z_test[i0]);
                }  

                double stdev = calc_stdev(y_test);
                double R2 = 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() );
                if (rank == 0){
                    //std::cout << "y stdev^2 = " << stdev * stdev << std::endl;  
                    //std::cout << "test l2 pred err^2 = " << l2_pred_err2 << std::endl;
                    // std::cout << "test R2 = " << 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() ) << std::endl;
                    std::cout <<  R2 << ", ";
                }

                if (R2 > maxR2){
                    maxR2 = R2;
                    maxind = it;
                }
            }

            if (rank == 0){
                std::cout << std::endl << "max R2 = " << maxR2 << std::endl;
                std::cout << std::endl << "max ind = " << maxind << std::endl;
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

            double l2_pred_err2 = 0;
            for (int i0 = 0; i0 < N_test; i0++){
                l2_pred_err2 += (y_test[i0] - z_test[i0]) * (y_test[i0] - z_test[i0]);
                //if ( std::isinf(l2_pred_err2) == 1 || i0 <= 4)
                //    std::cout << "y_test[" << i0 << "] = " << y_test[i0] << ", z_test[i0] = " << z_test[i0] << std::endl;
                //if (i0 % 1000 == 0)
                //    std::cout << "l2_pred_err2 = " << l2_pred_err2 << std::endl;
            }  

            double stdev = calc_stdev(y_test);
            if (rank == 0){
                std::cout << "y stdev^2 = " << stdev * stdev << std::endl;  
                std::cout << "test l2 pred err^2 = " << l2_pred_err2 << std::endl;
                std::cout << "test R2 = " << 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() ) << std::endl;
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
        std::string type_data = "bed";
        double alpha_scale = opt.get_alpha_scale();
        std::string bimfp = opt.get_bim_file();
        data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, rank, type_data, alpha_scale, bimfp);
        // dataset.read_phen();
        // dataset.read_genotype_data();
        // dataset.compute_markers_statistics();

        double gam1 = 1e-6; 
        double gamw;
        
        if (opt.get_h2() == -1)
            gamw = 2;
        else 
            gamw = 1.0 / (1.0 - opt.get_h2());

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

        data dataset_test(pheno_test, bedfp_test, N_test, M_test, Mt_test, S, rank, type_data, alpha_scale, bimfp);
        // dataset_test.read_phen();
        // dataset_test.read_genotype_data();
        // dataset_test.compute_markers_statistics();

        for (int i0 = 0; i0 < x_est.size(); i0++)
            x_est[i0] *= sqrt( (double) N_test );

        std::vector<double> z_test = dataset_test.Ax(x_est.data());

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
    else if (opt.get_run_mode() == "pvals-calc")
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
        
        std::string phenfp = (opt.get_phen_files())[0];
        std::string type_data = "bed";
        double alpha_scale = opt.get_alpha_scale();
        std::string bimfp = opt.get_bim_file();
        data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, rank, type_data, alpha_scale, bimfp);

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // reading an estimate file 
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%

        std::string est_file_name = opt.get_estimate_file();
        int pos_dot = est_file_name.find(".");
        std::string end_est_file_name = est_file_name.substr(pos_dot + 1);
        if (rank == 0)
            std::cout << "end_est_file_name = " << end_est_file_name << std::endl;

        std::vector<double> x_est;
        if (end_est_file_name == "bin")
            x_est = mpi_read_vec_from_file(est_file_name, M, S);
        else
            x_est = read_vec_from_file(est_file_name, M, S);

        for (int i0 = 0; i0 < x_est.size(); i0++)
            x_est[i0] *= sqrt( (double) N );


        //%%%%%%%%%%%%%%%%%%%%%%%%%%%
        //   obtaining p-values 
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        std::vector<double> z1 = dataset.Ax(x_est.data());
        std::vector<double> y =  dataset.filter_pheno();
        // saving pvals vector
        std::string filepath_out_pvals = opt.get_out_dir() + opt.get_out_name() + "_pvals.bin";
        if (rank == 0)
            std::cout << "filepath_out_pvals = " << filepath_out_pvals << std::endl;
        std::vector<double> pvals = dataset.pvals_calc(z1, y, x_est, filepath_out_pvals);

        // calculating p-values using LOCO method, if .bim file is specified
        if (dataset.get_bimfp() != ""){
            std::string filepath_out_pvals_LOCO = opt.get_out_dir() + opt.get_out_name();
            std::vector<double> pvals_LOCO = dataset.pvals_calc_LOCO(z1, y, x_est, filepath_out_pvals_LOCO);
            if (rank == 0)
                std::cout << "filepath_out_pvals_LOCO = " << filepath_out_pvals_LOCO << std::endl;
        }
    }
    else if (opt.get_run_mode() == "restart"){ 

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
        double alpha_scale = opt.get_alpha_scale();
        std::string bimfp = opt.get_bim_file();
        data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, rank, type_data, alpha_scale, bimfp);
        

        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // running EM-VAMP algorithm on the data
        //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        double gam1 = opt.get_gam1_init();
        double gamw = opt.get_gamw_init();
        
        std::vector<double> beta_true = std::vector<double> (M, 0.0);
        vamp emvamp(M, gam1, gamw, beta_true, rank, opt);

        std::vector<double> x_est = emvamp.infere(&dataset);

    }
    else if (opt.get_run_mode() == "predict"){ 

    // reading test set
    const std::string bedfp_test = opt.get_bed_file_test();
    const std::string pheno_test = (opt.get_phen_files_test())[0]; // currently it is only supported passing one pheno files as an input argument

    int N_test = opt.get_N_test();
    int Mt_test = opt.get_Mt_test();
    std::vector<double> MS = divide_work(Mt_test);
    int M_test = MS[0];
    int S_test = MS[1];
    std::string type_data = "bed";
    double alpha_scale = opt.get_alpha_scale();
    std::string bimfp = opt.get_bim_file();

    data dataset_test(pheno_test, bedfp_test, N_test, M_test, Mt_test, S_test, rank, type_data, alpha_scale, bimfp);
    
    std::vector<double> y_test = dataset_test.get_phen();

    //std::vector<double> x_est = read_vec_from_file(opt.get_estimate_file() + "_rank_" + std::to_string(rank) + ".bin", M, 0);
    std::string est_file_name = opt.get_estimate_file();
    int pos_dot = est_file_name.find(".");
    std::string end_est_file_name = est_file_name.substr(pos_dot + 1);
    if (rank == 0)
        std::cout << "est_file_name = " << est_file_name << std::endl;

    int pos_it = est_file_name.rfind("temp");
    std::vector<int> iter_range = opt.get_test_iter_range();
    int min_it = iter_range[0];
    int max_it = iter_range[1];
    if (rank == 0)
        std::cout << "iter range = [" << min_it << ", " << max_it << "]" << std::endl;

    std::vector< std::vector<double> > z_test_mat;

    if (min_it != -1){
        for (int it = min_it; it <= max_it; it++){
            std::vector<double> x_est;
            std::string est_file_name_it = est_file_name.substr(0, pos_it) + "temp_" + std::to_string(it) + "_" + std::to_string(it) + "_gibbs_est." + end_est_file_name;
            if (rank == 0)
                std::cout << "est_file_name_it = " << est_file_name_it << std::endl;
            if (end_est_file_name == "bin")
                x_est = mpi_read_vec_from_file(est_file_name_it, M_test, S_test);
            else
                x_est = read_vec_from_file(est_file_name_it, M_test, S_test);

            for (int i0 = 0; i0 < x_est.size(); i0++)
                x_est[i0] *= sqrt( (double) N_test );
            
            std::vector<double> z_test = dataset_test.Ax(x_est.data());
            z_test_mat.push_back(z_test);            
        }

        for (int i=0; i<N_test; i++){
            std::vector<double> row(max_it-min_it+1, 0.0);
            for (int it = min_it; it <= max_it; it++){ // min_it >= 1
                row[it-min_it] = z_test_mat[it-min_it][i];
            }
            std::string filepath_out = opt.get_out_dir() + opt.get_out_name() + "_predict_" + std::to_string(i) + ".csv";
            if (rank == 0){
                std::cout << "filepath_out = " << filepath_out << std::endl;
                store_vec_to_file(filepath_out, row);
                }
            }
        }
    }
    else if (opt.get_run_mode() == "predict_single"){ 

    // reading test set
    const std::string bedfp_test = opt.get_bed_file_test();
    const std::string pheno_test = (opt.get_phen_files_test())[0]; // currently it is only supported passing one pheno files as an input argument

    int N_test = opt.get_N_test();
    int Mt_test = opt.get_Mt_test();
    std::vector<double> MS = divide_work(Mt_test);
    int M_test = MS[0];
    int S_test = MS[1];
    std::string type_data = "bed";
    double alpha_scale = opt.get_alpha_scale();
    std::string bimfp = opt.get_bim_file();

    data dataset_test(pheno_test, bedfp_test, N_test, M_test, Mt_test, S_test, rank, type_data, alpha_scale, bimfp);
    
    std::vector<double> y_test = dataset_test.get_phen();

    //std::vector<double> x_est = read_vec_from_file(opt.get_estimate_file() + "_rank_" + std::to_string(rank) + ".bin", M, 0);
    std::string est_file_name = opt.get_estimate_file();
    int pos_dot = est_file_name.find(".");
    std::string end_est_file_name = est_file_name.substr(pos_dot + 1);
    //if (rank == 0)
    //    std::cout << "est_file_name = " << est_file_name << std::endl;

    std::vector<double> x_est;
    if (end_est_file_name == "bin")
        x_est = mpi_read_vec_from_file(est_file_name, M_test, S_test);
    else
        x_est = read_vec_from_file(est_file_name, M_test, S_test);

    for (int i0 = 0; i0 < x_est.size(); i0++)
        x_est[i0] *= sqrt( (double) N_test );
        
    std::vector<double> z_test = dataset_test.Ax(x_est.data());           

    std::string filepath_out = opt.get_out_dir() + opt.get_out_name() + "_predict.csv";
    if (rank == 0){
        std::cout << "filepath_out = " << filepath_out << std::endl;
        store_vec_to_file(filepath_out, z_test);
        }

    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}