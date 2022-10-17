#include <mpi.h>
#include <iostream>
#include <cmath> // contains definition of ceil
#include <bits/stdc++.h>  // contains definition of INT_MAX
#include <immintrin.h> // contains definition of _mm_malloc
#include <numeric>
#include "utilities.hpp"
#include "data.hpp"
#include "vamp.hpp"

int main()
{

    // starting parallel processes
    int required_MPI_level = MPI_THREAD_MULTIPLE;
    int provided_MPI_level;
    MPI_Init_thread(NULL, NULL, required_MPI_level, &provided_MPI_level);

    int rank = 0;
    int nranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // setting blocks of markers 
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    size_t Mt = 15000;
    size_t N = 12000;

    std::vector<double> MS = divide_work(Mt);
    int M = MS[0];
    int S = MS[1];
    int Mm = MS[2];

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // reading genotype data / phenotype file
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    const std::string bedfp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPvsGibbsvsLasso/genomes/12000_1.bed";
    const std::string phenfp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPvsGibbsvsLasso/genomes/12000_1_y.txt";
    const std::string true_betafp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPvsGibbsvsLasso/genomes/12000_1_beta_true.txt";
    const std::string phen_out = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/beta_true_out.txt";


    // simulating data for realistic valus of parameters
    std::vector<double> vars{0, 3.0246351e-07, 1.2863391e-03};
    std::vector<double> vars_init{0, N*3.0246351e-05, N*1.2863391e-04};
    //vars_init = vars;

    std::vector<double> probs{7.1100000e-01, 2.6440000e-01, 2.4600000e-02};
    std::vector<double> probs_init{6.000000e-01, 3.000000e-01, 1.0000000e-01};
    //probs_init = probs;

    std::vector<double> beta_true(M, 0.0); 
    
    if (rank == 0){
        std::vector<double> beta_true_tmp = simulate(Mt, vars, probs);
        store_vec_to_file("/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/M_12000_N_15000_beta_true.txt", beta_true_tmp);
        for (int ran = 1; ran < nranks; ran++)
            MPI_Send(beta_true.data(), Mt, MPI_DOUBLE, ran, 0, MPI_COMM_WORLD);
    }
    else{
        MPI_Status status;
        std::vector<double> beta_true_full(Mt, 0.0);
        MPI_Recv(beta_true_full.data(), Mt, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        for (int i0=S; i0<S+M; i0++)
            beta_true[i0-S] = beta_true_full[i0];
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    //beta_true = read_vec_from_file("/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/M_12000_N_15000_beta_true.txt", M, S);
    
    //printing out true variances
    if (rank == 0)
        std::cout << "true scaled variances = ";
    for (int i = 0; i < vars.size(); i++){
        if (rank == 0)
            std::cout << vars[i] * N << ' ';
    }
    if (rank ==0)
        std::cout << std::endl;

    //if (rank == 0){
    //    std::cout << "beta true stdev = " << calc_stdev(beta_true) << std::endl;
    //}

    double SNR = 1;
    double gamw = noise_prec_calc(SNR, vars, probs, Mt, N);
    if (rank == 0)
        std::cout << "true gamw = " << gamw << std::endl;

    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());  
    std::normal_distribution<double> gauss_beta_gen( 0, 1 / sqrt(gamw) ); //2nd parameter is stddev
    std::vector<double> noise(N, 0.0);
    for (int i = 0; i < N; i++){
        noise[i] = gauss_beta_gen(generator);
    }      
    if (rank == 0){
        for (int ran = 1; ran < nranks; ran++)
            MPI_Send(noise.data(), N, MPI_DOUBLE, ran, 0, MPI_COMM_WORLD);
        std::cout << "noise prec = " << 1.0 / pow(calc_stdev(noise), 2) << std::endl;
    }

    double *noise_val = (double*) _mm_malloc(size_t(N) * sizeof(double), 32);
    if (rank != 0)
        MPI_Recv(noise_val, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    data dataset(phenfp, bedfp, N, M, Mt, S, rank);
    dataset.read_phen();
    dataset.read_genotype_data();
    dataset.compute_markers_statistics();
    std::vector<double> beta_true_scaled = beta_true;
    for (int i0=0; i0<M; i0++)
        beta_true_scaled[i0] *= sqrt(N);
    std::vector<double> y = dataset.Ax( beta_true_scaled.data() );

    if (rank == 0)
        std::cout << "Ax stdev = " << calc_stdev(y) << std::endl;

    //if (rank == 0)
    //    std::cout << "noise_val stdev = " << calc_stdev(noise_val) << std::endl;

    for (int i = 0; i < N; i++)
        if (rank != 0)
            y[i] += noise_val[i];    
        else if (rank == 0)
            y[i] += noise[i];
    
    dataset.set_phen(y);

    if (rank == 0){
        std::cout << "var(y) = " << pow(calc_stdev(y), 2) << std::endl;
        double true_R2_tmp = calc_stdev(noise) / calc_stdev(y);
        std::cout << "true R2 = " << 1 - true_R2_tmp*true_R2_tmp << std::endl;
    }
    

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // running EM-VAMP algorithm on the data
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    double gam1 = 1e-6;
    int max_iter = 10;
    double rho = 0.9;
    std::string out_dir = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/sig_estimates/";
    std::string out_name = "x1_hat_height_main_11_10_2022"; 
    std::string model = "linear";
    gamw = 1;
    
    vamp emvamp(N, M, Mt, gam1, gamw, max_iter, rho, vars_init, probs_init, beta_true, rank, out_dir, out_name, model);
    std::vector<double> x_est = emvamp.infere(&dataset);

    MPI_Finalize();
    return 0;
}