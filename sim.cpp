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


    //%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // setting blocks of markers 
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%

    size_t Mt = opt.get_Mt();
    size_t N = opt.get_N();

    std::vector<double> MS = divide_work(Mt);
    int M = MS[0];
    int S = MS[1];
    int Mm = MS[2];
    long unsigned int seed = opt.get_seed();

    data dataset(std::vector<double> (N, 0.0), opt.get_bed_file(), N, M, Mt, S, rank);

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // running EM-VAMP algorithm on the data
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    // simulating data for realistic values of parameters

    std::vector<double> vars_init; // = opt.get_vars();
    std::vector<double> probs_init; //  = opt.get_probs();
    int CV = opt.get_CV();
    double h2 = opt.get_h2();
    //int CVhat = 2*CV;
    //CVhat = CV;
    //double h2hat = 0.8 * h2;
    //h2hat = h2;

    //int L = opt.get_num_mix_comp();

    //double prob_eq = (double) CVhat / Mt / (L-1) ;
    // prob_eq = (double) CVhat / Mt / (2 - 1.0 / pow(2, L-1));
    
    //double min_vars = 0.1 / CVhat;

    //std::vector<double> vars_init; // {0};
    //std::vector<double> probs_init; // {1 - (double) CVhat / Mt};

    //double curr_var = min_vars;
    //curr_var = h2hat / CVhat;

    //for (int i = 1; i<L; i++){
    //    probs_init.push_back(prob_eq);
    //    vars_init.push_back(curr_var);
    //    curr_var *= 10;
    //    prob_eq /= 2;
    //}

    std::vector<double> vars_true{0, h2 / CV};
    std::vector<double> probs_true{1 - (double) CV / Mt, (double) CV / Mt}; 


    //scaling variances
    //if (rank == 0)
    //    std::cout << "init scaled variances = ";
    //for (int i = 0; i < vars_init.size(); i++)
    //    if (rank == 0)
    //       std::cout << vars_init[i] * N << ' ';

    //if (rank ==0)
    //    std::cout << std::endl;

    //if (rank == 0)
    //    std::cout << "init probs = ";
    //for (int i = 0; i < probs_init.size(); i++)
    //    if (rank == 0)
    //        std::cout << probs_init[i] << ' ';
            
    //if (rank ==0)
    //    std::cout << std::endl;

        
    //printing out true variances
    if (rank == 0)
        std::cout << "true scaled variances = ";
    for (int i = 0; i < vars_true.size(); i++)
        if (rank == 0)
            std::cout << vars_true[i] * N << ' ';
    
    if (rank ==0)
        std::cout << std::endl;

    if (rank == 0)
        std::cout << "true probs = ";
    for (int i = 0; i < probs_true.size(); i++){
        if (rank == 0)
            std::cout << probs_true[i] << ' ';
    }
    if (rank ==0)
        std::cout << std::endl;

    // noise precision calculation
    double gamw = 1 / (1 - h2);
    if (rank == 0)
        std::cout << "true gamw = " << gamw << std::endl;


    std::vector<double> beta_true(M, 0.0); 
    std::vector<double> y;

    // loading true signal and phenotype in case they are user-provided
    std::vector< std::string > true_signal_files = opt.get_true_signal_files();
    

    if (!true_signal_files.empty()){

        std::string true_signal_file = true_signal_files[0];
    
        std::string phen_file = (opt.get_phen_files())[0];

        y= read_vec_from_file(phen_file, N, 0);

        dataset.set_phen(y);

        beta_true = mpi_read_vec_from_file(true_signal_file, M, S);

    }
    else
    {

        // simulating beta
        if (rank == 0){

            std::vector<double> beta_true_tmp = simulate(Mt, vars_true, probs_true, seed);
            
            for (int i0=S; i0<S+M; i0++)
                beta_true[i0-S] = beta_true_tmp[i0];
            
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

        // storing true beta
        std::string filepath_out = opt.get_out_dir() + opt.get_out_name() + "_beta_true.bin";
        mpi_store_vec_to_file(filepath_out, beta_true, S, M);


        std::random_device rand_dev;
        // std::mt19937 generator(rand_dev());  
        std::mt19937 generator{seed};
        std::normal_distribution<double> gauss_beta_gen( 0, 1 / sqrt(gamw) ); //2nd parameter is stddev
        std::vector<double> noise(N, 0.0);

        for (int i = 0; i < N; i++)
            noise[i] = gauss_beta_gen(generator);
        
        if (rank == 0){

            for (int ran = 1; ran < nranks; ran++)
                MPI_Send(noise.data(), N, MPI_DOUBLE, ran, 0, MPI_COMM_WORLD);

            std::cout << "noise prec = " << 1.0 / pow(calc_stdev(noise), 2) << std::endl;

        }

        double *noise_val = (double*) _mm_malloc(size_t(N) * sizeof(double), 32);

        if (rank != 0)
            MPI_Recv(noise_val, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        std::vector<double> beta_true_scaled = beta_true;
        for (int i0=0; i0<M; i0++)
            beta_true_scaled[i0] *= sqrt(N);
        
        y = dataset.Ax(beta_true_scaled.data());

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
    }

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // running EM-VAMP algorithm on the data
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    double gamw_init = 2; // 1 / (1 - h2hat);
    //gamw_init = gamw;

    double gam1 = 1e-8;
    //gam1 = 1e-3;

    //vars_init = vars_true;
    //probs_init = probs_true;

    vamp emvamp(N, M, Mt, gam1, gamw_init, opt.get_iterations(), opt.get_rho(), vars_init, probs_init, beta_true, rank, opt.get_out_dir() , opt.get_out_name(), opt.get_model(), opt);

    //vamp emvamp(M, gam1, gamw_init, beta_true, rank, opt);

    std::vector<double> x_est = emvamp.infere(&dataset);

    if (rank == 0){
        
        std::cout << "var(y) = " << pow(calc_stdev(y), 2) << std::endl;

        // double true_R2_tmp = calc_stdev(noise) / calc_stdev(y);

        // std::cout << "true R2 = " << 1 - true_R2_tmp*true_R2_tmp << std::endl;

        std::cout << "true gamw = " << gamw << std::endl;

        // std::cout << "noise prec = " << 1.0 / pow(calc_stdev(noise), 2) << std::endl;
    }

    MPI_Finalize();

    return 0;
}