#include <mpi.h>
#include <iostream>
#include <fstream>
#include <cmath> // contains definition of ceil
#include <bits/stdc++.h>  // contains definition of INT_MAX
#include <immintrin.h> // contains definition of _mm_malloc
#include <cstdlib> // contains definition of srand()
#include <numeric>
#include "utilities.hpp"
#include "phenotype.hpp"
#include "vamp.hpp"


double generate_mixture_gaussians(int K_grp, std::vector<double> eta, std::vector<double> pi)
{
    //std::cout << "eta[1] = " << eta[1] << std::endl;
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());  
    std::uniform_real_distribution<double> unif(0.0,1.0);
    double u = unif(generator);
    double c_sum = 0;
    double out_val = 0;
    for (int j=0; j<K_grp; j++)
    {   
        c_sum += pi[j];
        if (u <= c_sum)
        {
            if (eta[j] != 0)
            {
                std::normal_distribution<double> gauss_beta_gen( 0.0, sqrt( eta[j] ) ); //2nd parameter is stddev
                out_val = gauss_beta_gen(generator);
            }
            else
                out_val = 0;  // spike is being set at zero
            break;
        }
    }
    return out_val;
}


 std::vector<double> simulate(int M, std::vector<double> eta, std::vector<double> pi){

    int K_grp = eta.size();
    std::vector<double> signal(M, 0.0);
    for (int i = 0; i < M; i++){
        signal[i] = generate_mixture_gaussians(K_grp, eta, pi);
    }
    return signal;
 }

 double noise_prec_calc( double SNR, std::vector<double> vars, std::vector<double> probs, int Mt, int N ){

    double expe = 0;
    //std::cout << "probs[1] = " << probs[1] << std::endl;
    for( int i = 0; i < vars.size(); i++ )
        expe += vars[i] * probs[i];
    std::cout <<"expe = " << expe << std::endl;
    double gamw = SNR * N / Mt / expe;
    return gamw;
 }

std::vector<double> read_vec_from_file(std::string filename, int M, int S){
    double value;
    int it = 0;
    std::vector<double> V;
    std::ifstream inFile(filename);
    std::string word;
    while (inFile >> value){
        if ( it >= S && it < S+M ){
            V.push_back( value );
        }
        else if ( it >= S+M ){
            break;
        }  
        it += 1;
    } 
    //std::cout << "it | S | M = " << it <<" " << S << " " << M << std::endl;
    //std::cout << "length( V ) = " << V.size() << std::endl;
    return V;
}

 void store_vec_to_file(std::string filepath, std::vector<double> vec){

    std::fstream file;
    file.open(filepath);
 
    for(int i=0; i<vec.size(); i++)
        file << vec[i] << std::endl;

    file.close();
 }



int main()
{

    // starting parallel processes
    MPI_Init(NULL, NULL);

    int rank = 0;
    int nranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    std::cout << "rank = " << rank << std::endl;
    

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // setting blocks of markers 
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%

    
    //size_t Mt = 1812841;
    //size_t N = 2504;
    size_t Mt = 1272;
    size_t N = 44992;

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
    const std::string bedfp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/exploring_corr_effects_on_spectrum_12102021/ukb_imp_v3_UKB_EST_uncorrpeople_N45000_pruned_00001.bed";
    const std::string phenfp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPvsGibbsvsLasso/genomes/12000_1_y.txt";
    const std::string true_betafp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPvsGibbsvsLasso/genomes/12000_1_beta_true.txt";
    const std::string phen_out = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/cpp_VAMP/beta_true_out.txt";


    // simulating data for realistic valus of parameters
    std::vector<double> probs{7.1100000e-01, 2.6440000e-01, 2.4600000e-02};
    std::vector<double> vars{0, N*3.0246351e-07, N*1.2863391e-03};
    //std::vector<double> probs{7.1100000e-01, 2.89e-1};
    //std::vector<double> vars{0, N*3.0246351e-06};
    std::vector<double> beta_true = simulate(M, vars, probs);
    /*
    std::cout << "beta_true[1] = " << beta_true[1] << std::endl;
    std::cout << "beta_true[2] = " << beta_true[2] << std::endl;
    std::cout << "beta_true[3] = " << beta_true[3] << std::endl;
    std::cout << "beta_true[4] = " << beta_true[4] << std::endl;
    std::cout << "beta_true[5] = " << beta_true[5] << std::endl;
    std::cout << "beta_true[6] = " << beta_true[6] << std::endl;
    */
    

    // empirical standard deviation of beta true
    double sum = std::accumulate(beta_true.begin(), beta_true.end(), 0.0);
    double mean = sum / beta_true.size();
    //std::cout << "mean = " << mean << std::endl;

    double sq_sum = std::inner_product(beta_true.begin(), beta_true.end(), beta_true.begin(), 0.0);
    double stdev = std::sqrt(sq_sum / beta_true.size() - mean * mean);
    if (rank == 0)
        std::cout << "beta true stdev^2 = " << stdev * stdev << std::endl;
    

    double SNR = 1;
    double gamw = noise_prec_calc( SNR, vars, probs, Mt, N );
    //std::cout << "gamw = " << gamw << std::endl;

    //srand(10);
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());  
    std::normal_distribution<double> gauss_beta_gen( 0, 1 / sqrt(gamw) ); //2nd parameter is stddev
    std::vector<double> noise(N, 0.0);
    for (int i = 0; i < N; i++){
        noise[i] = gauss_beta_gen(generator);
    }        
    std::cout << "noise[2] = " <<noise[2] << std::endl;
    std::cout << "nranks = " << nranks << std::endl;
    for (int ran = 0; ran < nranks; ran++){
        std::cout << "ran = " <<ran << std::endl;
        MPI_Send(noise.data(), N, MPI_DOUBLE, ran, 0, MPI_COMM_WORLD);
    }

    std::cout << "after sending" << std::endl;
    double *noise_val = (double*) _mm_malloc(size_t(N) * sizeof(double), 32);
    MPI_Recv(noise_val, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (rank == 0)
        std::cout << "noise_val[2] = " << noise_val[2] << std::endl;
    phenotype phen(phenfp, bedfp, N, M, Mt, S, rank);
    phen.read_file();
    phen.read_genotype_data();
    phen.compute_markers_statistics();
    std::vector<double> y = phen.Ax( beta_true.data() );

    
    // empirical variance of Ax
    double sum_Ax = std::accumulate(y.begin(), y.end(), 0.0);
    double mean_Ax = sum_Ax / y.size();
    //std::cout << "mean_Ax = " << mean_Ax << std::endl;

    double sq_sum_Ax = std::inner_product(y.begin(), y.end(), y.begin(), 0.0);
    double stdev_Ax = std::sqrt(sq_sum_Ax / y.size() - mean_Ax * mean_Ax);
    if (rank == 0)
        std::cout << "Ax stdev^2 = " << stdev_Ax * stdev_Ax << std::endl;



    // empirical variance of the noise
    double sum_noise = std::accumulate(noise_val, noise_val + N, 0.0);
    double mean_noise = sum_noise / N;
    //std::cout << "mean = " << mean << std::endl;

    double sq_sum_noise = std::inner_product(noise_val, noise_val + N, noise_val, 0.0);
    double stdev_noise = std::sqrt(sq_sum_noise / N - mean_noise * mean_noise);
    if (rank == 0)
        std::cout << "noise stdev^2 = " << stdev_noise * stdev_noise << std::endl;

    if (rank == 0)
        std::cout << "Ax[2] = " << y[2] << std::endl;
    for (int i = 0; i < N; i++)
        y[i] += noise_val[i];    
    if (rank == 0)    
        std::cout << "y[2] = " << y[2] << std::endl;
    phen.set_data( y );
    
    
    //std::vector<double> data = phen.get_data();
    //std::cout << "data[1] = " << data[1] << std::endl; 
    //std::vector<double> beta_true = read_vec_from_file(true_betafp, M, S);

    //phen.store_vec_to_file(phen_out, y);
    //phen.read_file();

    
    //std::cout << "length( beta_true ) = " << beta_true.size() << std::endl;
    //std::cout << "beta_true[1] = " << beta_true[1] << std::endl;
    //std::cout << "beta_true[2] = " << beta_true[2] << std::endl;
    //std::cout << "beta_true[3] = " << beta_true[3] << std::endl;

    /*
    double *mave = phen.get_mave();

    std::cout << "mave[0]: " << mave[0] << std::endl;

    std::vector<double> test_y(M, 0.0);
    std::cout << "dot product: " << phen.dot_product(1, test_y.data(), mave[1], 1) << std::endl;

    std::vector<double> test_y2(N, 0.0);
    test_y2[1] = 1;

    std::vector<double> outAx = phen.Ax( test_y2.data() );
    std::cout << "Ax product[1]: " << outAx[1] << std::endl;
    */


    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // running EM-VAMP algorithm on the data
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    double gam1 = 1e-6; //, gamw = 2.112692482840060;
    int max_iter = 3;
    double rho = 0.98;
    vamp emvamp( N, M, Mt, gam1, gamw, max_iter, rho, vars, probs, beta_true, rank );
    std::vector<double> x_est = emvamp.infere( &phen );
    std::cout << "[main] after infere() " << std::endl;
    
    MPI_Finalize();
    return 0;
}