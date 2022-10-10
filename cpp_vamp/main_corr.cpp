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

    
    //size_t Mt = 115233;
    size_t Mt = 8430446;
    size_t N = 458747;

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

    //const std::string bedfp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/1kg_chr20_genotypes.bed";
    //const std::string phenfp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/1kg_chr20_genotypes.txt";
    //const std::string bedfp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPvsGibbsvsLasso/genomes/12000_1.bed";
    //const std::string bedfp = "/nfs/scistore13/robingrp/human_data/geno/chr/ukb22828_c21_UKB_EST_v3.bed";
    const std::string bedfp = "/nfs/scistore13/robingrp/human_data/geno/chr/ukb22828_UKB_EST_v3_all.bed";
    //const std::string bedfp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000_pruned_00000001.bed";
    const std::string phenfp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/N_458747_M_115233_y.txt";
    //const std::string phenfp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000_pruned_00001_y.txt";
    //const std::string true_betafp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPvsGibbsvsLasso/genomes/12000_1_beta_true.txt";
    //const std::string phen_out = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/beta_true_out.txt";


    // simulating data for realistic valus of parameters
    
    std::vector<double> vars{0, N*3.0246351e-07, N*1.2863391e-03};
    std::vector<double> vars_init{0, N*3.0246351e-05, N*1.2863391e-04};
    vars_init = vars;

    std::vector<double> probs{7.1100000e-01, 2.6440000e-01, 2.4600000e-02};
    std::vector<double> probs_init{6.000000e-01, 3.000000e-01, 1.0000000e-01};
    probs_init = probs;


    srand(10);
    std::vector<double> beta_true(M, 0.0); 
    
    std::string loc_beta_true = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/M_115233_N_458747_beta_true.txt";
    if (rank == 0){
        std::vector<double> beta_true_tmp = simulate(Mt, vars, probs);
        store_vec_to_file(loc_beta_true, beta_true_tmp);
        //for (int ran = 1; ran < nranks; ran++)
        //    MPI_Send(beta_true.data(), Mt, MPI_DOUBLE, ran, 0, MPI_COMM_WORLD);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    beta_true = read_vec_from_file(loc_beta_true, M, S);
    MPI_Barrier(MPI_COMM_WORLD);
    
    //std::cout << "rank = " << rank << ", beta_true[4] = " << beta_true[4] << std::endl;
    //std::cout << "rank = " << rank << ", beta_true[5] = " << beta_true[5] << std::endl;
    //std::cout << "rank = " << rank << ", beta_true[6] = " << beta_true[6] << std::endl;
    
    
    //printing out true variances
    if (rank == 0)
        std::cout << "true variances = ";
    for (int i = 0; i < vars.size(); i++){
        if (rank == 0)
            std::cout << vars[i] << ' ';
    }
    if (rank ==0)
        std::cout << std::endl;

    if (rank == 0){
        std::cout << "beta true stdev = " << calc_stdev(beta_true) << std::endl;
    }

    //std::cout << "rank = " << rank << ", before read_phen" << std::endl;
    data dataset(phenfp, bedfp, N, M, Mt, S, rank);
    dataset.read_phen();
    //std::cout << "rank = " << rank << ", after read_phen" << std::endl;
    dataset.read_genotype_data();
    dataset.compute_markers_statistics();
    std::vector<double> y = dataset.Ax( beta_true.data() );

    if (rank == 0)
        std::cout << "Ax stdev = " << calc_stdev(y) << std::endl;

    double SNR = 1;
    double gamw = noise_prec_calc(SNR, vars, probs, Mt, N);
    gamw = 1 / pow(calc_stdev(y),2);
    if (rank == 0)
        std::cout << "gamw = " << gamw << std::endl;

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

    //if (rank == 0)
    //    std::cout << "noise_val stdev = " << calc_stdev(noise_val) << std::endl;

    for (int i = 0; i < N; i++)
        if (rank != 0)
            y[i] += noise_val[i];    
        else if (rank == 0)
            y[i] += noise[i];
    
    dataset.set_phen(y);
    

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // running EM-VAMP algorithm on the data
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    double gam1 = 1e-6; //, gamw = 2.112692482840060;
    int max_iter = 12;
    double rho = 0.8;
    std::string out_dir = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/sig_estimates/";
    std::string out_name = "x1_hat_height_main"; 
    std::string model = "linear";
    gamw = 1;
    
    vamp emvamp(N, M, Mt, gam1, gamw, max_iter, rho, vars_init, probs_init, beta_true, rank, out_dir, out_name, model);
    std::vector<double> x_est = emvamp.infere(&dataset);
    

    //std::vector<double> CGsol = emvamp.CG_solver(std::vector<double> (M, 1.0), &dataset);
    //std::cout << "CGsol[0] = " << CGsol[0] << ", rank = " << rank << std::endl;
    
    //std::cout << "CGsol[3000] = " << CGsol[3000] << ", rank = " << rank << std::endl;

    MPI_Finalize();
    return 0;
}