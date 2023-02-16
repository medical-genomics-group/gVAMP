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
    MPI_Init(NULL, NULL);
    int required_MPI_level = MPI_THREAD_MULTIPLE;
    int provided_MPI_level;
    //MPI_Init_thread(NULL, NULL, required_MPI_level, &provided_MPI_level);

    const Options opt(argc, argv);

    int rank = 0;
    int nranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // setting blocks of markers 
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%


    //size_t Mt = 326165;
    //size_t N = 438361;

    size_t Mt = opt.get_Mt();
    size_t N = opt.get_N();

    const int modu = Mt % nranks;
    const int size = Mt / nranks;

    int Mm = Mt % nranks != 0 ? size + 1 : size;

    int len[nranks], start[nranks];
    int cum = 0;
    for (int i=0; i<nranks; i++) {
        len[i]  = i < modu ? size + 1 : size;
        start[i] = cum;
        cum += len[i];
    }
    assert(cum == Mt);

    int M = len[rank];
    int S = start[rank];  // task marker start

    printf("INFO   : rank %4d has %d markers over tot Mt = %d, max Mm = %d, starting at S = %d\n", rank, M, Mt, Mm, S);


    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // reading genotype data / phenotype file
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    // reading signal estimate from gmrm software
    const std::string true_beta_height = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/height_true.txt";
    std::vector<double> beta_true = read_vec_from_file(true_beta_height, M, S);

    //std::vector<double> probs;
    //std::vector<double> vars;
    //probs = {0.70412, 0.26945, 0.02643};
    //vars = {0, N*0.001251585388785e-5, N*0.606523422454662e-5};
    
    std::string phenfp = (opt.get_phen_files())[0]; // currently only one phenotype file is supported
    data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, rank);
    dataset.read_phen();
    dataset.read_genotype_data();
    dataset.compute_markers_statistics();

    std::vector<double> x(N, 1.0);

    double start_ATx = MPI_Wtime();
    std::vector<double> res = dataset.ATx( x.data() );
    double end_ATx = MPI_Wtime();

    if (rank == 0)
    std::cout << "calculation of ATx required " << end_ATx - start_ATx << " seconds" << std::endl;

    double start_dotp = MPI_Wtime();
    dataset.dot_product(1, x.data(), 0, 1);
    double end_dotp = MPI_Wtime();

    if (rank == 0){
        std::cout << "calculation of dot product required " << end_dotp - start_dotp << " seconds" << std::endl;
        std::cout << "ATx with 1 thread should have taken " << M * ( end_dotp - start_dotp ) << " second " << std::endl;
    }

    double start_Ax = MPI_Wtime();
    std::vector<double> Ax_arg(M, 1.0);
    std::vector<double> Ax_out = dataset.Ax( Ax_arg.data() );
    if (rank == 0){
        std::cout << "Ax_out[0] = " << Ax_out[0] << std::endl;
        std::cout << "Ax_out[1] = " << Ax_out[1] << std::endl;
        std::cout << "Ax_out[2] = " << Ax_out[2] << std::endl;
    }
    double end_Ax = MPI_Wtime();
    
    if (rank == 0)
    std::cout << "calculation of Ax required " << end_Ax - start_Ax << " seconds" << std::endl;
    

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // running EM-VAMP algorithm on the data
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    
    double gam1 = 1e-6; //, gamw = 2.112692482840060;
    //int max_iter = 1;
    //double rho = 0.98;
    //double SNR = 1;
    //double gamw = noise_prec_calc( SNR, vars, probs, Mt, N );
    double gamw = 2;
    //vamp emvamp(N, M, Mt, gam1, gamw, max_iter, rho, vars, probs, beta_true, rank);
    vamp emvamp(M, gam1, gamw, beta_true, rank, opt);
    //std::vector<double> x_est = emvamp.infere(&dataset);
    

    if (0){
        
        // reading test set
        const std::string bedfp_HTtest = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/ukb22828_UKB_EST_v3_ldp08_test_HT.bed";
        const std::string pheno_HTtest = "/nfs/scistore13/robingrp/human_data/pheno/continuous/ukb_test_HT.phen";

        int N_test = 15000;
        int Mt_test = Mt;
        int M_test = M;

        data dataset_test(pheno_HTtest, bedfp_HTtest, N_test, M_test, Mt_test, S, rank);
        dataset_test.read_phen();
        dataset_test.read_genotype_data();
        dataset_test.compute_markers_statistics();

        
        std::vector<double> x_est_scaled = beta_true;
        for (int i0 = 0; i0 < x_est_scaled.size(); i0++)
            x_est_scaled[i0] = beta_true[i0] * sqrt(N_test); // * sqrt(N_test / N);

        std::vector<double> z = dataset_test.Ax(x_est_scaled.data());

        // mean and std of z
        double sum_z = std::accumulate(z.begin(), z.end(), 0.0);
        double mean_z = sum_z / z.size();
        std::cout << "mean_z = " << mean_z << std::endl;

        double sq_sum_z = std::inner_product(z.begin(), z.end(), z.begin(), 0.0);
        double stdev_z = std::sqrt(sq_sum_z / z.size() - mean_z * mean_z);
        if (rank == 0)
            std::cout << "z stdev^2 = " << stdev_z * stdev_z << std::endl;
        


        std::cout << "z[0] = " << z[0] << ", rank = "<< rank << std::endl;
        for (int i0 = 0; i0 < z.size(); i0++ ){
            z[i0] = z[i0];
        }
        std::cout << "after z[0] = " << z[0] << ", rank = " << rank << std::endl;

        std::vector<double> y_test = dataset_test.get_phen();

        double l2_pred_err2 = 0;
        for (int i0 = 0; i0 < N_test; i0++){
            l2_pred_err2 += (y_test[i0] - z[i0]) * (y_test[i0] - z[i0]);
        }

        double sum = std::accumulate(y_test.begin(), y_test.end(), 0.0);
        double mean = sum / y_test.size();
        //std::cout << "mean = " << mean << std::endl;

        double sq_sum = std::inner_product(y_test.begin(), y_test.end(), y_test.begin(), 0.0);
        double stdev = std::sqrt(sq_sum / y_test.size() - mean * mean);
        if (rank == 0)
            std::cout << "y stdev^2 = " << stdev * stdev << std::endl;
        

        if (rank == 0){
            std::cout << "test l2 pred err = " << l2_pred_err2 << std::endl;
            std::cout << "test R2 = " << 1 - l2_pred_err2 / ( stdev * stdev * y_test.size() ) << std::endl;
        }

    }

    MPI_Finalize();
    return 0;
}