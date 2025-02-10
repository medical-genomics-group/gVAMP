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

    data dataset(std::vector<double> (N, 0.0), opt.get_bed_file(), N, M, Mt, S, rank);
        

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // running EM-VAMP algorithm on the data
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    // simulating data for realistic values of parameters

    //std::vector<double> vars_init = opt.get_vars();
    //std::vector<double> probs_init = opt.get_probs();
    int CV = opt.get_CV();
    double h2 = opt.get_h2();
    int CVhat = CV/2;
    //int CVhat = CV;
    //double h2hat = 0.8 * h2;
    double h2hat = 1.2*h2;

    double v = 1;

    int L = opt.get_num_mix_comp();

    // double prob_eq = (double) CVhat / Mt / (L-1) ;
    double prob_eq = (double) CVhat / Mt / (2 - 1.0 / pow(2, L-1));
    
    double min_vars = 0.01 / CVhat;

    std::vector<double> vars_init {0};
    std::vector<double> probs_init {1 - (double) CVhat / Mt};

    double curr_var = min_vars;
    for (int i = 1; i<L; i++){
        probs_init.push_back(prob_eq);
        vars_init.push_back(curr_var);
        curr_var *= 10;
        prob_eq /= 2;
    }


    double probit_var = opt.get_probit_var();

    std::vector<double> vars_true{0, h2 / CV};
    std::vector<double> probs_true{1 - (double) CV / Mt, (double) CV / Mt}; 


    //scaling variances
    if (rank == 0)
        std::cout << "init scaled variances = ";
    for (int i = 0; i < vars_init.size(); i++)
        if (rank == 0)
            std::cout << vars_init[i] * N << ' ';

    if (rank ==0)
        std::cout << std::endl;

    if (rank == 0)
        std::cout << "init probs = ";
    for (int i = 0; i < probs_init.size(); i++)
        if (rank == 0)
            std::cout << probs_init[i] << ' ';
            
    if (rank ==0)
        std::cout << std::endl;

    // simulating beta
    std::vector<double> beta_true(M, 0.0); 
    std::vector<double> beta_true_tmp;

    // storing true beta
    std::string filepath_out = opt.get_out_dir() + opt.get_out_name() + "_probit_beta_true.bin";
    
    if (rank == 0){
        
        beta_true_tmp = simulate(Mt, vars_true, probs_true);
        
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

        beta_true_tmp = beta_true_full;

   }
    
    MPI_Barrier(MPI_COMM_WORLD);

    mpi_store_vec_to_file(filepath_out, beta_true_tmp, S, M);

    //std::cout << " rank = " << rank << ", beta_true[M-1] = " << beta_true[M-1] << std::endl;

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

    std::vector<double> beta_true_scaled = beta_true;
    for (int i0=0; i0<M; i0++)
        beta_true_scaled[i0] *= sqrt( (double) N / v);
    
    std::vector<double> y = dataset.Ax(beta_true_scaled.data());

    if (rank == 0)
        std::cout << "Var(Xbeta) = " << pow(calc_stdev(y), 2) << std::endl;

    if (opt.get_cov_file() != ""){

        //std::vector<double> cov_effect{0.5, -0.1};
        std::vector<double> cov_effect;

        int C = opt.get_C();

        for (int c=0; c<C; c++)
            cov_effect.push_back( ( 2 * (double) (c % 2) - 1) * 0.25 );

        dataset.read_covariates(opt.get_cov_file(), C);
  
        std::vector<double> Zx_temp = dataset.Zx(cov_effect);

        std::transform (y.begin(), y.end(), Zx_temp.begin(), y.begin(), std::plus<double>());

    }

    if (rank == 0)
        std::cout << "Var(g) = " << pow(calc_stdev(y), 2) << std::endl;

    std::random_device unif_dev;
    std::mt19937 ugenerator(unif_dev());  
    std::uniform_real_distribution<double> unif(0.0,1.0);

    for (int i=0; i<N; i++){

        double u = unif(ugenerator);

        double prob = normal_cdf(y[i] / sqrt(probit_var));

        if (u <= prob)
            y[i] = 1;
        else 
            y[i] = 0;
    }

    dataset.set_phen(y);

    std::string filepath_out_y = opt.get_out_dir() + opt.get_out_name() + "_probit_y.txt";
    store_vec_to_file(filepath_out_y, y);

    if (rank == 0){

        std::cout << "count of 1s = " << std::accumulate(y.begin(), y.end(), 0) << std::endl;

        std::cout << "prob of 1s = " << (double) std::accumulate(y.begin(), y.end(), 0) / N << std::endl;

    }

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    // running EM-VAMP algorithm on the data
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    double gam1 = 1e-8;

    //vars_init = vars_true;
    //probs_init = probs_true;

    //vars_init = std::vector<double> {0, h2 / CV - 1e-7, h2 / CV + 1e-7};
    //probs_init = std::vector<double> {1 - (double) CV / Mt, (double) CV / 2 / Mt, (double) CV / 2 / Mt}; 


    vamp emvamp(N, M, Mt, gam1, 1, opt.get_iterations(), opt.get_rho(), vars_init, probs_init, beta_true, rank, opt.get_out_dir(), opt.get_out_name(), opt.get_model(), opt);

    std::vector<double> x_est = emvamp.infere(&dataset);

    MPI_Finalize();

    return 0;
}