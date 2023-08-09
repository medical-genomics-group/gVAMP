#include <mpi.h>
#include <iostream>
#include <cmath> // contains definition of ceil
#include <bits/stdc++.h>  // contains definition of INT_MAX
#include <immintrin.h> // contains definition of _mm_malloc
#include <numeric>
#include "utilities.hpp"
#include "data.hpp"
#include "vamp.hpp"

int main(int argc, char** argv)
{

    // starting parallel processes
    int required_MPI_level = MPI_THREAD_MULTIPLE;
    int provided_MPI_level;
    MPI_Init_thread(NULL, NULL, required_MPI_level, &provided_MPI_level);

    const Options opt(argc, argv);

    int rank = 0;
    int nranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (opt.get_run_mode() == "infere"){

    size_t Mt = opt.get_Mt();
    size_t N = opt.get_N();

    std::vector<double> MS = divide_work(Mt);
    int M = MS[0];
    int S = MS[1];
    int Mm = MS[2];
 
    std::string phenfp = (opt.get_phen_files())[0];
    std::string type_data = "meth";
    double alpha_scale = opt.get_alpha_scale();
    std::string bimfp = opt.get_bim_file();
    data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, rank, type_data, alpha_scale, bimfp);

    //dataset.read_covariates(opt.get_cov_file(), opt.get_C());

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // running EM-VAMP algorithm on the data
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    double gamw;
    if (opt.get_h2() == -1)
        gamw = 2;
    else 
        gamw = 1.0 / (1.0 - opt.get_h2());
    double gamw_init = 0.9 * gamw;
    double gam1 = 1e-6;

    std::vector<double> beta_true;
    if(opt.get_true_signal_files().size() > 0)
        beta_true = read_vec_from_file(opt.get_true_signal_files()[0], M, S);
    else 
        beta_true = std::vector<double> (M, 0.0);

    vamp emvamp(M, gam1, gamw_init, beta_true, rank, opt);
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
        std::string type_data = "meth";
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
        if (rank == 0)
            std::cout << "est_file_name = " << est_file_name << std::endl;

        int pos_it = est_file_name.rfind("it");
        std::vector<int> iter_range = opt.get_test_iter_range();
        int min_it = iter_range[0];
        int max_it = iter_range[1];
        if (rank == 0)
            std::cout << "iter range = [" << min_it << ", " << max_it << "]" << std::endl;

        if (min_it != -1){
            for (int it = min_it; it <= max_it; it++){
                std::vector<double> x_est;
                std::string est_file_name_it = est_file_name.substr(0, pos_it) + "it_" + std::to_string(it) + "." + end_est_file_name;
                //if (rank == 0)
                //    std::cout << "end_est_file_name = " << end_est_file_name << std::endl;
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
                if (rank == 0){
                    std::cout << "----------- ITERATION " << it << " ----------" << std::endl;
                    std::cout << "y stdev^2 = " << stdev * stdev << std::endl;  
                    std::cout << "test l2 pred err^2 = " << l2_pred_err2 << std::endl;
                    std::cout << "test R2 = " << 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() ) << std::endl;
                    //std::cout << 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() ) << ", ";
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

            double l2_pred_err2 = 0;
            for (int i0 = 0; i0 < N_test; i0++){
                l2_pred_err2 += (y_test[i0] - z_test[i0]) * (y_test[i0] - z_test[i0]);
            }  

            double stdev = calc_stdev(y_test);
            if (rank == 0){
                
                std::cout << "y stdev^2 = " << stdev * stdev << std::endl;  
                std::cout << "test l2 pred err^2 = " << l2_pred_err2 << std::endl;
                std::cout << "test R2 = " << 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() ) << std::endl;
            }
        }
        
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}