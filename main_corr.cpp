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

    // retrieving MPI specific information
    int rank = 0;
    int nranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    int normal = 1;
    std::cout << "normal = " << normal << std::endl;


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
    data dataset(phenfp, opt.get_bed_file(), opt.get_N(), M, opt.get_Mt(), S, normal, rank);
        

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // running EM-VAMP algorithm on the data
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    // simulating data for realistic values of parameters
    std::vector<double> vars{0, 1e-3, 1.2863391e-02};
    std::vector<double> vars_init{0, 2e-3, 2e-02};
    if (normal == 1){
        for (int i=0; i<vars_init.size(); i++)
            vars_init[i] *= N;
    }
    std::vector<double> probs{7.1100000e-01, 2.6440000e-01, 2.4600000e-02};
    std::vector<double> probs_init{6.000000e-01, 3.000000e-01, 1.0000000e-01};

    // simulating beta
    std::vector<double> beta_true(M, 0.0); 
    
    if (rank == 0){
        std::vector<double> beta_true_tmp = simulate(Mt, vars, probs);
        for (int i0=S; i0<S+M; i0++)
            beta_true[i0-S] = beta_true_tmp[i0];
        // storing true beta
        std::string filepath_out = opt.get_out_dir() + opt.get_out_name() + "_beta_true.bin";
        mpi_store_vec_to_file(filepath_out, beta_true_tmp, S, M);
        //store_vec_to_file("/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/M_115233_N_458747_beta_true.txt", beta_true_tmp);
        
        for (int ran = 1; ran < nranks; ran++)
            MPI_Send(beta_true_tmp.data(), Mt, MPI_DOUBLE, ran, 0, MPI_COMM_WORLD);
    }
    else{
        MPI_Status status;
        std::vector<double> beta_true_full(Mt, 0.0);
        MPI_Recv(beta_true_full.data(), Mt, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        for (int i0=S; i0<S+M; i0++)
            beta_true[i0-S] = beta_true_full[i0];
   }
    
    MPI_Barrier(MPI_COMM_WORLD);
    // beta_true = read_vec_from_file("/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/M_12000_N_15000_beta_true.txt", M, S);
    

    //printing out true variances
    if (rank == 0)
        std::cout << "true scaled variances = ";
    for (int i = 0; i < vars.size(); i++){
        if (rank == 0)
            if (normal == 2)
                std::cout << vars[i] << ' ';
            else if (normal == 1)
                std::cout << vars[i] * N << ' ';
    }
    if (rank ==0)
        std::cout << std::endl;

    if (rank == 0)
        std::cout << "true probs = ";
    for (int i = 0; i < probs.size(); i++){
        if (rank == 0)
            std::cout << probs[i] << ' ';
    }
    if (rank ==0)
        std::cout << std::endl;

    //if (rank == 0){
    //    std::cout << "beta true stdev = " << calc_stdev(beta_true) << std::endl;
    //}

    // noise precision calculation
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

    std::vector<double> beta_true_scaled = beta_true;
    if (normal == 1)
        for (int i0=0; i0<M; i0++)
            beta_true_scaled[i0] *= sqrt(N);
    
    std::vector<double> y = dataset.Ax(beta_true_scaled.data(), normal);

    if (rank == 0)
        std::cout << "Var(Ax) = " << pow(calc_stdev(y), 2) << std::endl;


    for (int i = 0; i < N; i++)
        if (rank != 0)
            y[i] += noise_val[i];    
        else if (rank == 0)
            y[i] += noise[i];

    dataset.set_phen(y);

    std::string filepath_out_y = opt.get_out_dir() + opt.get_out_name() + "_y.txt";
    store_vec_to_file(filepath_out_y, y);

    if (rank == 0){
        std::cout << "Var(y) = " << pow(calc_stdev(y), 2) << std::endl;
        double true_R2_tmp = calc_stdev(noise) / calc_stdev(y);
        std::cout << "true R2 = " << 1 - true_R2_tmp*true_R2_tmp << std::endl;
    }

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // running EM-VAMP algorithm on the data
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    double gamw_init = 2;
    double gam1 = 1e-6;
    vamp emvamp(M, gam1, gamw_init, beta_true, rank, opt);
    std::vector<double> x_est = emvamp.infere(&dataset);

    if (normal == 2){
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

    if (rank == 0){
        std::cout << "var(y) = " << pow(calc_stdev(y), 2) << std::endl;
        double true_R2_tmp = calc_stdev(noise) / calc_stdev(y);
        std::cout << "true R2 = " << 1 - true_R2_tmp*true_R2_tmp << std::endl;
        std::cout << "true gamw = " << gamw << std::endl;
        std::cout << "noise prec = " << 1.0 / pow(calc_stdev(noise), 2) << std::endl;
    }

    MPI_Finalize();
    return 0;
}