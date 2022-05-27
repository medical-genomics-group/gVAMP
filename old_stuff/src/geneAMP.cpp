#include <cstdlib>
#include <Eigen/Eigen>

#include "geneAMP.hpp"
#include "BayesRRm.h"

#include "data.hpp"
#include "distributions_boost.hpp"
#include "options.hpp"
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>
#include <sys/stat.h>
#include <libgen.h>
#include <string.h>
#include <boost/range/algorithm.hpp>
#include <sys/time.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <mm_malloc.h>
#include <mpi.h>
#include <omp.h>
#include <cmath>
// #include "geneAMP_arms.h"
#include "dense.hpp"
#include "sparse.hpp"
#include "utils.hpp"

#include <iterator>
#include <boost/random/uniform_01.hpp>
#include <boost/random/mersenne_twister.hpp>

#include "constants.hpp"

#include "KPM_Chebyshev.hpp"
#include "sparse_mult_tools.hpp"
#include "denoiser.hpp"

geneAMP::~geneAMP() {}


double geneAMP::generating_mixture_gaussians(int K_grp, VectorXd eta_gen, VectorXd pi_gen)
{
    std::uniform_real_distribution<double> unif(0.0,1.0);
    double u = unif(generator);
    double c_sum = 0;
    double out_val = 0;
    for (int j=0; j<K_grp; j++)
    {   
        c_sum += pi_gen(j);
        if (u <= c_sum)
        {
            if (eta(j) != 0)
            {
                std::normal_distribution<double> gauss_beta_gen(0.0,sqrt(eta_gen(j))); //2nd parameter is stddev
                out_val = gauss_beta_gen(generator);
            }
            else
            {
                out_val = 0;  // spike is being set at zero
            }
            break;
        }
    }
    return out_val;
}

void geneAMP::init(unsigned int individualCount, unsigned int Mtot, unsigned int fixedCount)
{
	// Read the failure indicator vector
	if(individualCount != (data.fail).size()){
		cout << "Number of phenotypes "<< individualCount << " was different from the number of failures " << (data.fail).size() << endl;
		exit(1);
	}

	//initialize epsilon vector as the phenotype vector
	y = data.y.cast<double>().array();

	//Store the vector of failures only in the structure used for sampling alpha
	used_data_alpha.failure_vector = data.fail.cast<double>();
}
    


//EO: MPI GIBBS
//-------------
int geneAMP::runMpiGibbs_bW() {

	const unsigned int numFixedEffects(data.numFixedEffects);

    char   buff[LENBUF];
    char   buff_gamma[LENBUF_gamma];
    int    nranks, rank, name_len, result;
    double dalloc = 0.0;

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // MPI_File   outfh, betfh, epsfh, gamfh, cpnfh, mrkfh, xivfh; 
    // MPI_File   xbetfh, xcpnfh;
    MPI_Status status;
    MPI_Info   info;


    //AMP iterations MPI file
    MPI_File beta_est_AMP;
    MPI_File beta_true_AMP;

    // Set up processing options
    // -------------------------
    if (rank < 0) {
        opt.printBanner();
        opt.printProcessingOptions();
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
        if (rank == 0) {
            int nt = omp_get_num_threads();
            if (omp_get_thread_num() == 0) 
                printf("INFO   : OMP parallel regions will use %d thread(s)\n", nt);
        }
    }


    uint Ntot       = data.set_Ntot(rank, opt);
    const uint Mtot = data.set_Mtot(rank, opt);

    //Reset the dist
    //@@@EO WHY THIS ONE ??? dist.reset_rng((uint)(opt.seed + rank*1000));
    dist.reset_rng((uint)(opt.seed + rank*1000));
	
    if (rank == 0)
        printf("INFO   : Full dataset includes Mtot=%d markers and Ntot=%d individuals.\n", Mtot, Ntot);


    // Define global marker indexing
    // -----------------------------
    int MrankS[nranks], MrankL[nranks], lmin = 1E9, lmax = 0;
    assign_blocks_to_tasks(data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot, nranks, rank, MrankS, MrankL, lmin, lmax);

    uint M = MrankL[rank];
    if (rank % 10 == 0) {
        printf("INFO   : rank %4d will handle a block of %6d markers starting at %d\n", rank, MrankL[rank], MrankS[rank]);
    }


    // EO: Define blocks of individuals (for dumping epsilon)
    // Note: hack the marker block definition function to this end
    // Note: at this stage Ntot is not yet adjusted for missing phenotypes,
    //       hence the correction in the call
    // --------------------------------------------------------------------
    int IrankS[nranks], IrankL[nranks];
    define_blocks_of_markers(Ntot - data.numNAs, IrankS, IrankL, nranks);


    // Invariant initializations (from scratch / from restart)
    // -------------------------------------------------------
    string betaestAMP = opt.mcmcOut + ".beta_est_AMP." + std::to_string(rank);
    string betatrueAMP = opt.mcmcOut + ".beta_true_AMP." + std::to_string(rank);

    init(Ntot - data.numNAs, Mtot,numFixedEffects);


    MPI_File_delete(betaestAMP.c_str(), MPI_INFO_NULL);
    MPI_File_delete(betatrueAMP.c_str(), MPI_INFO_NULL);

    MPI_Barrier(MPI_COMM_WORLD);

    check_mpi(MPI_File_open(MPI_COMM_SELF, betaestAMP.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &beta_est_AMP), __LINE__, __FILE__);

    check_mpi(MPI_File_open(MPI_COMM_SELF, betatrueAMP.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &beta_true_AMP), __LINE__, __FILE__);

    // First element of the .bet, .cpn and .acu files is the
    // total number of processed markers
    // -----------------------------------------------------
    MPI_Offset offset = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();
    
    double tl = -mysecond();

    // Read the data (from sparse representation by default)
    // -----------------------------------------------------
    size_t *N1S, *N1L,  *N2S, *N2L,  *NMS, *NML;
    N1S = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(N1S, __LINE__, __FILE__);
    N1L = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(N1L, __LINE__, __FILE__);
    N2S = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(N2S, __LINE__, __FILE__);
    N2L = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(N2L, __LINE__, __FILE__);
    NMS = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(NMS, __LINE__, __FILE__);
    NML = (size_t*)_mm_malloc(size_t(M) * sizeof(size_t), 64);  check_malloc(NML, __LINE__, __FILE__);
    dalloc += 6.0 * double(M) * sizeof(size_t) / 1E9;


    // Boolean mask for using BED representation or not (SPARSE otherwise)
    // For markers with USEBED == true then the BED representation is 
    // converted on the fly to SPARSE the time for the corresponding marker
    // to be processed
    // --------------------------------------------------------------------
    bool *USEBED;
    USEBED = (bool*)_mm_malloc(M * sizeof(bool), 64);  check_malloc(USEBED, __LINE__, __FILE__);
    for (int i=0; i<M; i++) USEBED[i] = false;
    int nusebed = 0;


    uint *I1, *I2, *IM;
    size_t taskBytes = 0;

    if (opt.readFromBedFile) {
        data.load_data_from_bed_file(opt.bedFile, Ntot, M, rank, MrankS[rank],
                                     N1S, N1L, I1,
                                     N2S, N2L, I2,
                                     NMS, NML, IM,
                                     taskBytes);
    } else {
        string sparseOut = opt.get_sparse_output_filebase(rank);
        data.load_data_from_sparse_files(rank, nranks, M, MrankS, MrankL, sparseOut,
                                         N1S, N1L, I1,
                                         N2S, N2L, I2,
                                         NMS, NML, IM,
                                         taskBytes);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    tl += mysecond();

    if (rank == 0) {
        printf("INFO   : rank %3d took %.3f seconds to load  %lu bytes  =>  BW = %7.3f GB/s\n", rank, tl, taskBytes, (double)taskBytes * 1E-9 / tl);
        fflush(stdout);
    }


    // Correct each marker for individuals with missing phenotype
    // ----------------------------------------------------------
    if (data.numNAs > 0) {

        if (rank == 0)
            printf("INFO   : applying %d corrections to genotype data due to missing phenotype data (NAs in .phen).\n", data.numNAs);

        data.sparse_data_correct_for_missing_phenotype(N1S, N1L, I1, M, USEBED);
        data.sparse_data_correct_for_missing_phenotype(N2S, N2L, I2, M, USEBED);
        data.sparse_data_correct_for_missing_phenotype(NMS, NML, IM, M, USEBED);

        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) printf("INFO   : finished applying NA corrections.\n");

        // Adjust N upon number of NAs
        Ntot -= data.numNAs;
        if (rank == 0 && data.numNAs > 0)
            printf("INFO   : Ntot adjusted by -%d to account for NAs in phenotype file. Now Ntot=%d\n", data.numNAs, Ntot);
    }

    // Compute statistics (from sparse info)
    // -------------------------------------
    //if (rank == 0) printf("INFO   : start computing statistics on Ntot = %d individuals\n", Ntot);
    double dN   = (double) Ntot;
    double dNm1 = (double)(Ntot - 1);
    double *mave, *mstd, *sum_failure, *sum_failure_fix; 

    mave = (double*)_mm_malloc(size_t(M) * sizeof(double), 64);  check_malloc(mave, __LINE__, __FILE__);
    mstd = (double*)_mm_malloc(size_t(M) * sizeof(double), 64);  check_malloc(mstd, __LINE__, __FILE__);
    sum_failure = (double*)_mm_malloc(size_t(M) * sizeof(double), 64);  check_malloc(mstd, __LINE__, __FILE__);

    sum_failure_fix = (double*)_mm_malloc(size_t(numFixedEffects) * sizeof(double), 64);  check_malloc(mstd, __LINE__, __FILE__);

    dalloc += 2 * size_t(M) * sizeof(double) / 1E9;

    double tmp0, tmp1, tmp2;
    double temp_fail_sum = used_data_alpha.failure_vector.array().sum();
    for (int i=0; i<M; ++i) {
        // For now use the old way to compute means
        mave[i] = (double(N1L[i]) + 2.0 * double(N2L[i])) / (dN - double(NML[i]));      
       
        tmp1 = double(N1L[i]) * (1.0 - mave[i]) * (1.0 - mave[i]);
        tmp2 = double(N2L[i]) * (2.0 - mave[i]) * (2.0 - mave[i]);
        tmp0 = double(Ntot - N1L[i] - N2L[i] - NML[i]) * (0.0 - mave[i]) * (0.0 - mave[i]);
        //TODO At some point we need to turn sd to 1/sd for speed
        //mstd[i] = sqrt(double(Ntot - 1) / (tmp0+tmp1+tmp2));
        mstd[i] = sqrt( (tmp0+tmp1+tmp2)/double(Ntot - 1));

        int temp_sum = 0;
        for(size_t ii = N1S[i]; ii < (N1S[i] + N1L[i]) ; ii++){
            temp_sum += used_data_alpha.failure_vector(I1[ii]);
        }
        for(size_t ii = N2S[i]; ii < (N2S[i] + N2L[i]) ; ii++){
            temp_sum += 2*used_data_alpha.failure_vector(I2[ii]);
        }
        sum_failure[i] = (temp_sum - mave[i] * temp_fail_sum) / mstd[i];

        //printf("marker %6d mean %20.15f, std = %20.15f (%.1f / %.15f)  (%15.10f, %15.10f, %15.10f)\n", i, mave[i], mstd[i], double(Ntot - 1), tmp0+tmp1+tmp2, tmp1, tmp2, tmp0);
    }



    MPI_Barrier(MPI_COMM_WORLD);

    const auto et2 = std::chrono::high_resolution_clock::now();
    const auto dt2 = et2 - st2;
    const auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();
    if (rank == 0)   std::cout << "INFO   : time to preprocess the data: " << du2 / double(1000.0) << " seconds." << std::endl;


    // Build list of markers    
    // ---------------------
    // for (int i=0; i<M; ++i)
    //     markerI.push_back(i);


    // Processing part
    // ---------------
    const auto st3 = std::chrono::high_resolution_clock::now();

    double *y, *tmpEps, *deltaEps, *dEpsSum, *deltaSum, *epsilon , *tmpEps_vi, *tmp_deltaEps;
    const size_t NDB = size_t(Ntot) * sizeof(double);


    cout << "sizeof(double) = " << sizeof(double) << " vs sizeof(long double) = " << sizeof(long double) << endl;

    // Main iteration loop
    // -------------------
    //bool replay_it = false;
    double tot_sync_ar1  = 0.0;
    double tot_sync_ar2  = 0.0;
    int    tot_nsync_ar1 = 0;
    int    tot_nsync_ar2 = 0;
    int    *glob_info, *tasks_len, *tasks_dis, *stats_len, *stats_dis;
    if (opt.sparseSync) {
        glob_info  = (int*)    _mm_malloc(size_t(nranks * 2) * sizeof(int),    64);  check_malloc(glob_info,  __LINE__, __FILE__);
        tasks_len  = (int*)    _mm_malloc(size_t(nranks)     * sizeof(int),    64);  check_malloc(tasks_len,  __LINE__, __FILE__);
        tasks_dis  = (int*)    _mm_malloc(size_t(nranks)     * sizeof(int),    64);  check_malloc(tasks_dis,  __LINE__, __FILE__);
        stats_len  = (int*)    _mm_malloc(size_t(nranks)     * sizeof(int),    64);  check_malloc(stats_len,  __LINE__, __FILE__);
        stats_dis  = (int*)    _mm_malloc(size_t(nranks)     * sizeof(int),    64);  check_malloc(stats_dis,  __LINE__, __FILE__);
    }


    const int ninit  = 4;
    const int npoint = 100;
    const int nsamp  = 1;
    const int ncent  = 4;

    //#################################################################
    //                  AMP ALGORITHM
    //#################################################################





    // 0. INITIALIZATION OF OBJECTS USED

    // defining mixture groups 

    pi(0) = 0.05;
    pi(1) = 0.70;
    pi(2) = 0.25;
    pi = pi / pi.sum();

    eta(0) = 0;
    eta(1) = 0.1;
    eta(2) = 1;
   
    // eta(1) = data.y.array().abs().matrix().mean()  /10 / M;
    // eta(0) = sqrt(eta(1));
    // eta(2) = eta(1)*eta(1);
    // eta(3) = 1e-12;
    // eta(3) = data.y.array().abs().matrix().mean()  /10 / M;
    // eta(4) = eta(3) * eta(3);

    cout << "Probabilities of the mixture components " << pi.transpose() << endl;
    cout << "Variance of the mixture components " << eta.transpose() << endl;

    //EM algorithm mixture initialization
    eta_EM = eta;
    VectorXd eta_EM_prev = eta_EM;
    pi_EM = pi; 
    VectorXd pi_EM_prev = pi;
    MatrixXd gamma_EM = MatrixXd::Zero(M, K_groups);
    VectorXd nk_EM = VectorXd::Zero(K_groups);
    //means are always zero

    









    // 1. APPROXIMATION OF SPECTRAL DENSITY OF (X^T)*X

    int KPM_calc = 0;
    int exact_calc = 0;

    if (KPM_calc)
    {
        int num_points = 100;
        VectorXd points = VectorXd::Zero(num_points);
        for (int point_ind=0; point_ind<num_points; point_ind++)
        {
            points(point_ind) = 0 + 1.0 / num_points * point_ind;
        }
        int M_deg = 25;
        double scaling = 20000.0;
        VectorXd DOS = VectorXd::Zero(num_points);
        DOS = KPM(points, num_points, M_deg, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, Ntot,  M, scaling); //DOS= Density Of States
        cout << "DOS: " << DOS << endl;
    }
    

    if (exact_calc)
    {
        //calculating spectrum of genotype matrix
        double scaling = 12000.0;
        MatrixXd corr_mat_MM = MatrixXd::Random(M,M);
        for (int j1 = 0; j1 < M; j1++)
        {
            VectorXd vj1 = VectorXd::Zero(M);
            vj1(j1) = 1;
            VectorXd tmpNtot_spec = VectorXd::Zero(Ntot);
            tmpNtot_spec = xbeta_mult(vj1, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scaling);
            vj1 = xtr_mult(tmpNtot_spec, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scaling);
            for (int r_ind = 0; r_ind < M ; r_ind++)
            {
                corr_mat_MM(r_ind, j1) = vj1(r_ind);
            }
        }

        // cout << "done constructing corr_mat_MM" << endl;

        SelfAdjointEigenSolver<MatrixXd> es((corr_mat_MM + corr_mat_MM.transpose() )/2);
        VectorXd z = es.eigenvalues().real();
        cout << z.transpose() << endl;
        cout << "second cumulant est:" << z.mean() << endl;
    }


    //setting the scale for genotype matrix
    double scale = sqrt(Ntot) * 20;
    // scale = 250;










    // 2. GENERATING TRUE SIGNAL AND PHENOTYPE VALUES

    std::uniform_real_distribution<double> unif(0.0,1.0);
    VectorXd beta_true(M);
    for (int i=0; i<M; i++)
    {
        beta_true(i) = geneAMP::generating_mixture_gaussians(K_groups, eta, pi);
    }
    cout << "beta_true(1): "<< beta_true(1) << endl;
    cout << "beta_true(2): "<< beta_true(2) << endl;
    cout << "beta_true(3): "<< beta_true(3) << endl;


    //normalization of true signal so that it complies with assumptions of Marco's sanity check
    double signal_normalization_const = sqrt( M / beta_true.squaredNorm() );
    cout << "signal_normalization_const: " << signal_normalization_const << endl;
    beta_true = signal_normalization_const * beta_true;
    eta_EM = signal_normalization_const * signal_normalization_const * eta_EM;
    cout << "eta_EM new: " << eta_EM << endl;



    //generating noise
    VectorXd noise(Ntot);
    double sigma_noise = 0.10;
    std::normal_distribution<double> gauss_noise(0.0,sigma_noise); // second parameter is stddev, not variance
    for (int i=0; i<Ntot; i++) 
    {
       noise(i) = gauss_noise(generator);
    }

    //generating y
    VectorXd y_phen =VectorXd::Zero(Ntot);
    VectorXd xbeta = VectorXd::Zero(Ntot);
    xbeta= xbeta_mult(beta_true, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale);
    y_phen = xbeta + noise;
    cout << "y_phen(1): "<< y_phen(1)<< endl;


    //AMP algorithm initialization of objects
    VectorXd r_prev =VectorXd::Zero(Ntot);
    VectorXd r = r_prev;
    VectorXd beta_v = VectorXd::Zero(M);
    VectorXd beta_v_prev = VectorXd::Zero(M);
    VectorXd beta_eff_obs = VectorXd::Zero(M);
    std::normal_distribution<double> gauss_beta_init(0.0,1);
    double delta = Ntot / M;








    // 3. CALCULATION OF INITIAL APPROXIMATION - spectral / all zeros / true signal + small noise

    int spectral_init = 1;
    int all_zeros = 0;
    int true_plus_noise = 0;
    

    if ( spectral_init == 1 )
    {
        // 3.A. SPECTRAL INITIALIZATION

        for (int i = 0; i < M; i++)
        {
            beta_v(i) = gauss_beta_init(generator);
        }
        cout << "beta_v(1): " << beta_v(1) << endl;

        int counter = 0;
        do
        {
            beta_v_prev = beta_v;
            VectorXd xbeta_tmp(Ntot);
            xbeta_tmp = xbeta_mult(beta_v, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale);
            for (int i = 0; i < Ntot; i++)
            {
                xbeta_tmp(i) = y_phen(i) * xbeta_tmp(i);
            }
            beta_v = xtr_mult(xbeta_tmp, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale);
            beta_v = beta_v / beta_v.lpNorm<2>();
            counter +=1;
        } while ( abs ( abs(beta_v.dot(beta_v_prev)) / beta_v_prev.lpNorm<2>() - 1 ) > 1e-12 && counter < 50);

        cout << beta_v(1) << " " << beta_v(2) << endl;
        cout << beta_v_prev(1) << " " << beta_v_prev(2) << endl;
        cout << "corr(beta_true, beta_v): " << beta_true.dot(beta_v) / beta_true.lpNorm<2>() / beta_v.lpNorm<2>() << endl; 
        cout << "counter: " << counter << endl;
    }
    else if ( all_zeros == 1 )
    {
        // 3.B. ALL-ZERO INITIALIZATION 

        beta_v = VectorXd::Zero(M);
        cout << "Initializing Beta as zero vector." << endl;
    }
    else if ( true_plus_noise == 1)
    {
        // 3.C TRUE SIGNAL + SMALL NOISE

        double sigma_noise_init = 1;
        std::normal_distribution<double> gauss_noise_init(0.0,sigma_noise);
        for (int i=0; i<M; i++) 
        {
           beta_v(i) = beta_true(i) + gauss_noise_init(generator);
        }
        beta_eff_obs = beta_v;
    }














    // 4. CORE OF THE AMP ALGORITHM INCLUDING STATE EVOLUTION PARAMETERS ESTIMATION

    //defining damping factors 
    double damping_b = 1;
    double damping_b_eff = 1;

    //State evolution parameters - analytic calculation
    double sigma_analytic = -1; // sigma_analytic is variance parameter from state equation formula
    double b = 0;

    cout << "data.y.mean(): " << data.y.mean() << endl;
    //y_phen = (data.y.array() - data.y.mean()).matrix(); // / sqrt(Ntot); //sqrt(Ntot); //true phenotype values

    //AMP iterations
    for (int outer_iter = 0; outer_iter < iterAMP; outer_iter++)
    {
        if (all_zeros == 1)
        {
            sigma_analytic = pow(sigma_noise,2) + eta.dot(pi)/delta;
        }

        for (int iter = 0; iter < iterNumb; iter++)
        {

            // 4.A. Error measures output

            cout << "ITERATION NO."<<iter<< endl;
            cout << "(not relevant for real data) ||beta-muk*beta_true||/ ||beta_true||: "<< (muk*beta_true - beta_eff_obs).lpNorm<2>() / beta_true.lpNorm<2>() << endl;
            cout << "(not relevant for real data) l2 prediction error normalized (beta_true): "<< (y_phen - xbeta_mult(beta_eff_obs/muk, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<2>() / (y_phen - xbeta_mult(beta_true, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<2>() << endl;
            cout << "(not relevant for real data) ||beta-muk*beta_true||/ sqrt(M): "<< (muk*beta_true - beta_eff_obs).lpNorm<2>() / sqrt(M) << endl;
            cout << "l2 prediction error: " << (y_phen - xbeta_mult(beta_eff_obs/muk, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<2>() << endl;
            cout << "(not relevant for real data) l2 beta_true prediction error: " << (y_phen - xbeta_mult(beta_true, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<2>() << endl; 
            cout << "l2 prediction error / sqrt(Ntot) : "<< (y_phen - xbeta_mult(beta_eff_obs/muk, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<2>() / sqrt(Ntot) << endl;
            cout << "y_phen abs mean: "  << y_phen.array().abs().mean() << endl;
            cout << "beta_v abs mean: "  << beta_v.array().abs().mean() << endl;
            cout << "linf prediction error: "<< (y_phen - xbeta_mult(beta_eff_obs / muk, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<Infinity>() << endl;
            VectorXd y_mean_1 = VectorXd::Ones(Ntot); 
            y_mean_1 = y_mean_1 * y_phen.mean();
            cout << "R2: " << 1 - (y_phen - xbeta_mult(beta_eff_obs / muk, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).array().square().sum()/(y_phen - y_mean_1).array().square().sum() << endl;
            cout << "<beta^(k), beta>/||beta^(k)|| / ||beta||: "<< beta_eff_obs.dot(beta_true) / beta_eff_obs.lpNorm<2>() / beta_true.lpNorm<2>() << endl;
            cout << "sigma^2/muk^2: "<< sigma / muk/muk << endl;


            // MARCO'S SANITY CHECK

            if ( iter == 1 )
            {
                
                cout << endl << "Marco's sanity check: " << endl;

                cout << "<beta^(k), beta>/||beta^(k)|| / ||beta||: "<< beta_eff_obs.dot(beta_true) / beta_eff_obs.lpNorm<2>() / beta_true.lpNorm<2>() << endl;

                cout << "delta: " << Ntot / M << endl;

            }


        
            VectorXd xbeta = VectorXd::Zero(Ntot);

            xbeta = xbeta_mult(beta_v, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale);
 
            r = y_phen - xbeta + b*r_prev;



            //calculating X^T*r while taking sparsity into account 
            
            VectorXd c_beta = VectorXd::Zero(M); //vector of beta corrections

            c_beta = xtr_mult(r, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale);

            beta_v = damping_b_eff * (beta_v + c_beta) + ( 1 - damping_b_eff ) * beta_v;

            beta_eff_obs = beta_v;

            sigma = r.squaredNorm()/Ntot; // estimator of sigma_k^2
            
            muk = sqrt( abs(beta_eff_obs.squaredNorm() / M - sigma) /  pi_EM.dot(eta_EM) );
            
            cout << "sigma: "<<sigma << ", sigma_analytic: "<< sigma_analytic << " | " << "muk: "<< muk <<  endl;




            // 4.B. updating analytic state evolution parameters
            if ( K_groups == 1 )
            {
                sigma_analytic = pow(sigma_noise,2) + sigma_analytic * muk * muk * eta(0) / (muk * muk * eta(0) + sigma_analytic)/delta; // check again!
            }
            else
            {
                //we perform MCMC sampling to calculate E[(beta - fk(beta + sigma_k * G))]

                int MCMCNoSamples = 10000;

                std::normal_distribution<double> standard_normal(0.0,1);

                VectorXd evolution_sigma_samples(MCMCNoSamples);

                for (int j = 0; j < MCMCNoSamples; j++)
                {
                     double g = standard_normal(generator);

                     double beta_tilde = geneAMP::generating_mixture_gaussians(K, eta, pi);

                     evolution_sigma_samples(j) = pow( beta_tilde - fk ( beta_tilde + sqrt(sigma_analytic)*g, sigma, muk, K_groups, pi, eta ), 2 );
                }

                sigma_analytic = evolution_sigma_samples.mean()/delta + pow(sigma_noise,2);
            }
 
            //applying denoiser to our corrected signal

            VectorXd beta_v_damping_tmp = beta_v;

            for (int j = 0; j<M; j++)
            {
                beta_v_damping_tmp(j) = fk ( beta_v(j), sigma, muk, K_groups, pi_EM, eta_EM );
            }

            beta_v = damping_b * beta_v_damping_tmp + ( 1 - damping_b ) * beta_v;


            //applying fkd and finding a mean of beta_eff_obs
            
            VectorXd beta_v_tmp = VectorXd::Zero(M);

            for (int j = 0; j<M; j++)
            {
                beta_v_tmp(j) = fkd ( beta_eff_obs(j), sigma, muk, K_groups, pi_EM, eta_EM );
            } 

            b = beta_v_tmp.mean() * M / Ntot; 

            cout << "b: "<< b << " | " << "scale: " << scale << endl << endl << endl;

            r_prev = r;      

            // scale =  scale * muk;
        }









        // 5. EM ALGORITHM FOR ESTIMATING THE PRIOR DISTRIBUTION
        
        int counter = 0;
        
        //EM algorithm
        do
        {
            eta_EM_prev = eta_EM;
            pi_EM_prev = pi_EM;
            counter +=1;

            //E_step
            for(int l = 0; l < M; l++)
            {
                double zz = (abs(beta_v(l))<1e5*std::numeric_limits<double>::epsilon());
                double suma = 0;
                for (int u = 0; u < K_groups; u++)
                {
                    if (eta_EM(u)!=0)
                        suma += pi_EM(u) * exp( -0.5*beta_v(l) * beta_v(l) / eta_EM(u)) / sqrt(eta_EM(u) );
                    else
                        suma += pi_EM(u) * zz;
                }

                for (int g = 0; g< K_groups; g++)
                {
                    if (eta_EM(g)!=0)
                        gamma_EM(l,g) = pi_EM(g) * exp( -0.5 * beta_v(l) * beta_v(l) / eta_EM(g)) / sqrt(eta_EM(g)) /suma;
                    else
                        gamma_EM(l,g) = pi_EM(g) / suma * zz;
                }
            }


            //M-step
            for (int g = 0; g < K_groups; g++)
            {
                nk_EM(g) = gamma_EM.col(g).sum();
                pi_EM(g) = nk_EM(g) / M;
                eta_EM(g) = (beta_v.array().square() * gamma_EM.col(g).array()).sum() /  nk_EM(g);
            }
        } 
        while ( (sqrt( (pi_EM-pi_EM_prev).array().square().sum() / pi_EM_prev.array().square().sum())  > 5e-10 || sqrt((eta_EM-eta_EM_prev).array().square().sum() / eta_EM_prev.array().square().sum()) > 5e-10 ) && counter < 1000);
    
        cout << "FINAL ERROR MEAURES:" << endl;
        cout << "counter: "<< counter << endl;    
        cout << "pi_EM: " << endl << pi_EM << endl;
        cout << "eta_EM: " << endl << eta_EM << endl;
        cout << "(not relevant for real data) ||beta-muk*beta_true||/ ||beta_true||: "<< (muk*beta_true - beta_eff_obs).lpNorm<2>() / beta_true.lpNorm<2>() << endl;
        cout << "(not relevant for real data) l2 prediction error normalized (beta_true): "<< (y_phen - xbeta_mult(beta_eff_obs/muk, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<2>() / (y_phen - xbeta_mult(beta_true, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<2>() << endl;
        cout << "(not relevant for real data) ||beta-muk*beta_true||/ sqrt(M): "<< (muk*beta_true - beta_eff_obs).lpNorm<2>() / sqrt(M) << endl;
        cout << "l2 prediction error: " << (y_phen - xbeta_mult(beta_eff_obs/muk, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<2>() << endl;
        cout << "l2 beta_true prediction error: " << (y_phen - xbeta_mult(beta_true, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<2>() << endl; 
        cout << "l2 prediction error / sqrt(Ntot) : "<< (y_phen - xbeta_mult(beta_eff_obs/muk, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<2>() / sqrt(Ntot) << endl;
        cout << "y_phen abs mean: "  << y_phen.array().abs().mean() << endl;
        cout << "beta_v abs mean: "  << beta_v.array().abs().mean() << endl;
        cout << "linf prediction error: "<< (y_phen - xbeta_mult(beta_eff_obs /muk, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<Infinity>() << endl;
        VectorXd y_mean = VectorXd::Ones(Ntot); 
        y_mean = y_mean * y_phen.mean();
        cout << "R2: " << 1 - (y_phen - xbeta_mult(beta_eff_obs / muk, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).array().square().sum()/(y_phen - y_mean).array().square().sum() << endl;
        cout << "<beta^(k), beta>/||beta^(k)|| / ||beta||: "<< beta_eff_obs.dot(beta_true) / beta_eff_obs.lpNorm<2>() / beta_true.lpNorm<2>() << endl;
        cout << "sigma^2/muk^2: "<< sigma / muk/muk << endl;
    }
    offset = 0;
    
    VectorXd beta_out = beta_eff_obs/muk;
    cout << "Writting "<< M << " coefficient values. beta_est(0): "<< beta_out(0) << endl;
    check_mpi(MPI_File_write_at(beta_est_AMP, offset, beta_out.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_write_at(beta_true_AMP, offset, beta_true.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);
    









    // 6. CALCULATING LeastSquare SOLUTION USING IMPORTANCE SAMPLING

    // MatrixXd A = MatrixXd::Zero(M,M);
    // VectorXd row_squares(Ntot);
    // for (int j=0; j<M; j++)
    // {
    //     row_squares -= pow(mave[j]/mstd[j],2) * VectorXd::Ones(Ntot);
    // }
    // for (int j = 0; j < M; j++)
    // {
    //     for (size_t ii = N1S[j]; ii < (N1S[j] + N1L[j]); ii++)
    //     {
    //         row_squares(I1[ii]) += ( 1/pow(mstd[j],2) - 2*mave[j]/mstd[j] );
    //     }
    //     for (size_t ii = N2S[j]; ii < (N2S[j] + N2L[j]); ii++)
    //     {
    //         row_squares(I2[ii]) += ( 4/pow(mstd[j],2) - 4*mave[j]/mstd[j] );
    //     }        
    // }
    // row_squares /= row_squares.sum();

    // cout << "row_squares(0): "<< row_squares(0) << endl;

    // //std::vector<double> w = { 2, 2, 1, 1, 2, 2, 1, 1, 2, 1000 };
    // uniform_01<> dist;
    // boost::random::mt19937 gen(342575235);
    // std::vector<double> vals;
    // // for (auto iter : w) {
    // //     vals.push_back(std::pow(dist(gen), 1. / iter));
    // //     //vals.push_back(std::pow(dist(gen), 1. / iter));
    // // }
    // for (int j=0; j<Ntot; j++)
    // {
    //     vals.push_back(std::pow(dist(gen), 1. / row_squares(j)));
    // }
    // // Sorting vals, but retain the indices. 
    // // There is unfortunately no easy way to do this with STL.
    // std::vector<std::pair<int, double>> valsWithIndices;
    // for (size_t iter = 0; iter < vals.size(); iter++) {
    //     valsWithIndices.emplace_back(iter, vals[iter]);
    // }
    // std::sort(valsWithIndices.begin(), valsWithIndices.end(), [](auto x, auto y) {return x.second > y.second; });

    // std::vector<size_t> samples;
    // int sampleSize = 15;
    // sampleSize = M;
    // for (auto iter = 0; iter < sampleSize; iter++) {
    //     samples.push_back(valsWithIndices[iter].first);
    // }
    // // for (auto iter : samples) {
    // //     std::cout << iter << " ";
    // // }
    // cout << "samples(0): "<< samples[0] << endl;

    // //this part is inefficient -> todo: consider restoring data so that it is searchable by the observation index
    // MatrixXd A_helper = MatrixXd::Zero(M,sampleSize);
    // for (int j = 0; j<M; j++)
    // {
    //     for (int i = 0; i < sampleSize; i++)
    //     {
    //         A_helper(j,i) -= mave[j]/mstd[j] / sqrt(Ntot);
    //     }
    // }

    // for (int j = 0; j < M; j++)
    // {
    //     for (size_t ii = N1S[j]; ii < (N1S[j] + N1L[j]); ii++)
    //     {
    //         auto ind = std::find(samples.begin(), samples.end(), I1[ii]);
    //          if (ind != samples.end())
    //           {
    //             int index = ind - samples.begin();
    //             A_helper(j, index) += ( 1/pow(mstd[j],2) - 2*mave[j]/mstd[j] ) / Ntot;
    //           }
    //     }
    //     for (size_t ii = N2S[j]; ii < (N2S[j] + N2L[j]); ii++)
    //     {
    //         auto ind = std::find(samples.begin(), samples.end(), I2[ii]);
    //         if (ind != samples.end())
    //         {
    //              int index = ind - samples.begin();
    //              A_helper(j, index) += ( 4/pow(mstd[j],2) - 4*mave[j]/mstd[j] ) / Ntot;
    //         }
    //     }        
    // }
    // for (int j=0; j<sampleSize; j++)
    // {
    //     A += A_helper.col(j) * A_helper.col(j).transpose()/sampleSize / row_squares(samples[j]);
    // }

    // cout << "A(0,0): " << A(0,0) << endl;

    // VectorXd LS_sol = A.partialPivLu().solve(xtr_mult(y_phen, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale));

    // cout << "LS_sol(1): " << LS_sol(1) << endl;

    // cout << "(not relevant) LS l2 prediction error normalized (beta_true): "<< (y_phen - xbeta_mult(LS_sol, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<2>() / (y_phen - xbeta_mult(beta_true, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale)).lpNorm<2>() << endl;

    // //generating y_test
    // VectorXd y_phen_test =VectorXd::Zero(Ntot);
    // VectorXd xbeta = VectorXd::Zero(Ntot);
    // xbeta= xbeta_mult(beta_true, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2, scale);
    // y_phen_test = xbeta + noise;
    // cout << "y_phen(1): "<< y_phen_test(1)<< endl;



    // Close output files
    check_mpi(MPI_File_close(&beta_est_AMP), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&beta_true_AMP), __LINE__, __FILE__);

    // Release memory
    _mm_free(mave);
    _mm_free(mstd);
     _mm_free(USEBED);
    _mm_free(sum_failure);
    _mm_free(N1S);
    _mm_free(N1L);
    _mm_free(I1);
    _mm_free(N2S); 
    _mm_free(N2L);
    _mm_free(I2);
    _mm_free(NMS);
    _mm_free(NML);
    _mm_free(IM);

    // if (opt.sparseSync) {
    //     _mm_free(glob_info);
    //     _mm_free(tasks_len);
    //     _mm_free(tasks_dis);
    //     _mm_free(stats_len);
    //     _mm_free(stats_dis);
    // }

    const auto et3 = std::chrono::high_resolution_clock::now();
    const auto dt3 = et3 - st3;
    const auto du3 = std::chrono::duration_cast<std::chrono::milliseconds>(dt3).count();
    if (rank == 0)
        printf("INFO   : rank %4d, time to process the data: %.3f sec, with %.3f (%.3f, %.3f) = %4.1f%% spent on allred (%d, %d)\n",
               rank, du3 / double(1000.0),
               tot_sync_ar1 + tot_sync_ar2, tot_sync_ar1, tot_sync_ar2,
               (tot_sync_ar1 + tot_sync_ar2) / (du3 / double(1000.0)) * 100.0,
               tot_nsync_ar1, tot_nsync_ar2);

    return 0;
}
