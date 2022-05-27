/*
 * BayesRRm.cpp
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */

#include <cstdlib>
#include <chrono>
#include <numeric>
#include <random>
#include <algorithm>
#include <libgen.h>
#include <string>
#include <boost/range/algorithm.hpp>
#include <sys/time.h>
#include <iostream>
#include <ctime>
#include <mm_malloc.h>
#include <omp.h>
#include "dotp_lut.h"
#include "dense.hpp"
#include "sparse.hpp"
#include "utils.hpp"
#include "BayesRRm.h"
#include "data.hpp"
#include "distributions_boost.hpp"
#include "options.hpp"
#include "xfiles.h"
#include "depsx.hpp"


BayesRRm::BayesRRm(Data &data, Options &opt)
    : data(data)
    , opt(opt)
    , bedFile(opt.bedFile + ".bed")
    , seed(opt.seed)
    , max_iterations(opt.chainLength)
    , burn_in(opt.burnin)
    , showDebug(false)
{
    double* ptr = &opt.S[0];
    cva = (Eigen::Map<Eigen::VectorXd>(ptr, static_cast<long>(opt.S.size()))).cast<double>();
}


BayesRRm::~BayesRRm() {}

void BayesRRm::init_from_scratch() {

    iteration_start = 0;
}


//EO: .csv, .bet, .acu, .cpn, .mus         : written every --opt.thin
//    .eps, .mrk, .gam, .xiv, .xbet, .xcpn : last itertion only written every --opt.save
// 
//    one can restart from .bet and .cpn files or .xbet and .xcpn files,
//    by using --restart --ignore-xfiles rather than --restart
//
void BayesRRm::init_from_restart(const int K, const uint M, const uint  Mtot, const uint Ntot,
                                 const int* MrankS, const int* MrankL, const bool use_xfiles_in_restart) {

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0)
        printf("RESTART: from files: %s.* files\n", opt.mcmcOut.c_str());

    /*
    //EO: the .csv files is read to decide where to restart from
    data.read_mcmc_output_csv_file(opt.mcmcOut,
                                   opt.thin, opt.save,
                                   K, sigmaG, sigmaE, estPi,
                                   iteration_to_restart_from,
                                   first_thinned_iteration,
                                   first_saved_iteration);
    if (rank == 0) {
        printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
        printf("RESTART: Reading .cvs file %s\n", (opt.mcmcOut + ".csv").c_str());
        printf("RESTART: --thin %d  -- save %d\n", opt.thin, opt.save);
        printf("RESTART: iteration_to_restart_from = %d\n", iteration_to_restart_from);
        printf("RESTART: first_thinned_iteration   = %d\n", first_thinned_iteration);
        printf("RESTART: first_saved_iteration     = %d\n", first_saved_iteration);
        fflush(stdout);
    }
    */

    data.read_mcmc_output_out_file(opt.mcmcOut,
                                   opt.thin, opt.save,
                                   K, sigmaG, sigmaE, estPi,
                                   iteration_to_restart_from,
                                   first_thinned_iteration,
                                   first_saved_iteration);
    if (rank == 0) {
        printf("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@\n");
        printf("RESTART: Reading .out file %s\n", (opt.mcmcOut + ".out").c_str());
        printf("RESTART: --thin %d  --save %d\n", opt.thin, opt.save);
        printf("RESTART: iteration_to_restart_from = %d\n", iteration_to_restart_from);
        printf("RESTART: first_thinned_iteration   = %d\n", first_thinned_iteration);
        printf("RESTART: first_saved_iteration     = %d\n", first_saved_iteration);
        fflush(stdout);
    }
    
    //EO: Kill the processing if one's try to restart from iteration 0 as 
    //    we do not save this iteration, as this makes no sense to do so
    if (iteration_to_restart_from == 0) {
        printf("\nFATAL  : There is no point in restarting a chain from iteration 0 (not saved anyway)\n");
        printf("         => restart your analysis from scratch\n\n");
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }


    MPI_Barrier(MPI_COMM_WORLD);

    // .bet  saved at --opt.thin
    // .xbet saved at --opt.save, single line with last iteration
    data.read_mcmc_output_bet_file(opt.mcmcOut, Mtot,
                                   iteration_to_restart_from,
                                   first_thinned_iteration,
                                   opt.thin,
                                   MrankS, MrankL,
                                   use_xfiles_in_restart,
                                   Beta);

    data.read_mcmc_output_cpn_file(opt.mcmcOut, Mtot,
                                   iteration_to_restart_from,
                                   first_thinned_iteration,
                                   opt.thin,
                                   MrankS, MrankL,
                                   use_xfiles_in_restart,
                                   components);

    data.read_mcmc_output_mus_file(opt.mcmcOut,
                                   iteration_to_restart_from,
                                   first_thinned_iteration,
                                   opt.thin,
                                   mu_restart);

    data.read_mcmc_output_eps_file(opt.mcmcOut, Ntot,
                                   iteration_to_restart_from,
                                   epsilon_restart);

    data.read_mcmc_output_idx_file(opt.mcmcOut, "mrk", M,
                                   iteration_to_restart_from, opt.bayesType,
                                   markerI_restart);

    if (opt.covariates) {
        data.read_mcmc_output_gam_file(opt.mcmcOut, data.X.cols(),
                                       iteration_to_restart_from,
                                       gamma_restart);
        data.read_mcmc_output_idx_file(opt.mcmcOut, "xiv", (uint)data.X.cols(),
                                       iteration_to_restart_from, opt.bayesType,
                                       xI_restart);
    }


    if (opt.bayesType == "bayesFHMPI") {
        read_ofile_t1(lbvfp, iteration_to_restart_from, M, lambda_var.data(), MPI_DOUBLE);
        read_ofile_t1(nuvfp, iteration_to_restart_from, M, nu_var.data(),     MPI_DOUBLE);

        read_ofile_hsa(cslfp,
                       iteration_to_restart_from, first_thinned_iteration, opt.thin,
                       c_slab.size(), c_slab.data(), MPI_DOUBLE);
	
        read_ofile_hsa(taufp,
                       iteration_to_restart_from, first_thinned_iteration, opt.thin,
                       tau.size(), tau.data(), MPI_DOUBLE);

        read_ofile_hsa(htafp,
                       iteration_to_restart_from, first_thinned_iteration, opt.thin,
                       hypTau.size(), hypTau.data(), MPI_DOUBLE);
    }


    // Adjust starting iteration number.
    iteration_start = iteration_to_restart_from + 1;


    MPI_Barrier(MPI_COMM_WORLD);
}


//EO: MPI GIBBS
//-------------
int BayesRRm::runMpiGibbs() {

    typedef Matrix<bool, Dynamic, 1> VectorXb;

    int    nranks, rank, name_len, result;
    double dalloc = 0.0;

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status status;
    MPI_Info   info;

    // Display banner if wished
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

    //@@@@@ BW
    //Reset the dist
    //dist.reset_rng((uint)(opt.seed + rank*1000));


    if (rank == 0)
        printf("INFO   : Full dataset includes Mtot=%d markers and Ntot=%d individuals.\n", Mtot, Ntot);


    // Handle marker groups: by default we assume a single group with all markers belonging
    // to it. In this case -S option is used for the mixture. Otherwise, we expect a group file
    // covering all markers with a group mixture file covering all groups.
    // Warning: passed mixture shall not contain the 0.0 value, it is added by default.

    // First, load the mixture components (adding 0.0 as first element/column)
    if (opt.groupIndexFile != "" && opt.groupMixtureFile != "") {
        data.readGroupFile(opt.groupIndexFile);
        data.readmSFile(opt.groupMixtureFile);
    } else {
        // Define one group for all markers (group 0)
        data.groups.resize(Mtot);
        data.groups.setZero();
        assert(data.numGroups == 1);
        data.mS.resize(data.numGroups, cva.size() + 1);
        data.mS(0, 0) = 0.0;
        for (int i=0; i<cva.size(); i++) {
            if (cva(i) <= 0.0)
                throw("FATAL  : mixture value can only be strictly positive");
            data.mS(0, i + 1) = cva(i);
        }
    }
    assert(data.numGroups == data.mS.rows());

    // Display for control
    if (rank == 0)  data.printGroupMixtureComponents();

    
    // data.mS contains all mixture components, 0.0 included
    const unsigned int  K   = int(data.mS.cols());
    const unsigned int  km1 = K - 1;

    const int numGroups = data.numGroups;

    // Length of a marker in [byte] in BED representation (1 ind is 2 bits, so 1 byte is 4 inds)
    size_t snpLenByt  = (Ntot %  4) ? Ntot /  4 + 1 : Ntot /  4;

    // Length of a marker in [uint] in BED representation (1 ind is 2 bits, so 1 uint is 16 inds)
    size_t snpLenUint = (Ntot % 16) ? Ntot / 16 + 1 : Ntot / 16;

    if (rank == 0)  printf("INFO   : snpLenByt = %zu bytes; snpLenUint = %zu\n", snpLenByt, snpLenUint);


    // Define global marker indexing
    // -----------------------------
    int MrankS[nranks], MrankL[nranks], lmin = 1E9, lmax = 0;
    assign_blocks_to_tasks(data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot, nranks, rank, MrankS, MrankL, lmin, lmax);

    uint M = MrankL[rank];
    if (rank % 10 == 0) {
        printf("INFO   : rank %3d will handle a block of %6d markers starting at %d\n", rank, MrankL[rank], MrankS[rank]);
    }


    // EO: Define blocks of individuals (for dumping epsilon)
    // Note: hack the marker block definition function to this end
    // Note: at this stage Ntot is not yet adjusted for missing phenotypes,
    //       hence the correction in the call
    // --------------------------------------------------------------------
    int IrankS[nranks], IrankL[nranks];
    define_blocks_of_markers(Ntot - data.numNAs, IrankS, IrankL, nranks);

    VectorXi groups     = data.groups;
    cVa.resize(numGroups, K);                   // component-specific variance
    cVaI.resize(numGroups, K);                  // inverse of the component variances
    std::vector<int>    markerI;

    VectorXd            muk(K);                 // mean of k-th component marker effect size
    VectorXd            denom(K-1);             // temporal variable for computing the inflation of the effect variance for a given non-zero component
    VectorXi            m0(numGroups);          // non-zero elements per group
    cass.resize(numGroups,K);                   // variable storing the component assignment per group; EO RENAMING was v
    MatrixXi            sum_cass(numGroups,K);  // To store the sum of v elements over all ranks per group;         was sum_v
    VectorXd            Acum(M);
    VectorXb            adaV(M);                // Daniel adaptative scan Vector, ones will be sampled, 0 will be set to 0
    std::vector<int>    mark2sync;
    std::vector<double> dbet2sync;
    VectorXd            sum_scaledBSQN(numGroups); //sum of scaled squared norms

    dalloc +=     M * sizeof(int)    / 1E9; // for components
    dalloc += 2 * M * sizeof(double) / 1E9; // for Beta and Acum


    // Adapt the --thin and --save options such that --save >= --thin and --save%--thin = 0
    // ------------------------------------------------------------------------------------
    if (opt.save < opt.thin) {
        opt.save = opt.thin;
        if (rank == 0) printf("WARNING: opt.save was lower that opt.thin ; opt.save reset to opt.thin (%d)\n", opt.thin);
    }
    if (opt.save%opt.thin != 0) {
        if (rank == 0) printf("WARNING: opt.save (= %d) was not a multiple of opt.thin (= %d)\n", opt.save, opt.thin);
        opt.save = int(opt.save/opt.thin) * opt.thin;
        if (rank == 0) printf("         opt.save reset to %d, the closest multiple of opt.thin (%d)\n", opt.save, opt.thin);
    }


    // Invariant initializations (from scratch / from restart)
    // -------------------------------------------------------
    set_output_filepaths(opt.mcmcOut, std::to_string(rank));

    priorPi.resize(numGroups,K);
    priorPi.setZero();

    estPi.resize(numGroups,K);
    estPi.setZero();

    gamma.setZero();

    //fixed effects matrix
    X = data.X;

    priorPi.col(0).array() = 0.5;
    cVa.col(0).array()     = 0.0;
    cVaI.col(0).array()    = 0.0;

    muk[0]     = 0.0;
    mu         = 0.0;

    for (int i=0; i<numGroups; i++) {
        cVa.row(i).segment(1,km1)     = data.mS.row(i).segment(1,km1);
        cVaI.row(i).segment(1,km1)    = cVa.row(i).segment(1,km1).cwiseInverse();
        priorPi.row(i).segment(1,km1) = priorPi(i,0) * cVa.row(i).segment(1,km1).array() / cVa.row(i).segment(1,km1).sum();
    }

    cout << "cVa: " << cVa << endl;

    estPi = priorPi;
    //cout << "estPi after init " << estPi << endl;

    std::vector<unsigned int> xI(data.X.cols());
    std::iota(xI.begin(), xI.end(), 0);
    xI_restart.resize(data.X.cols());

    Beta.resize(M);
    Beta.setZero();


    // FHDT parameters
    hypTau.resize(numGroups);
    tau.resize(numGroups);
    hypTau.setZero();
    tau.setZero();
    scaledBSQN.resize(numGroups);
    scaledBSQN.setZero();
    sum_scaledBSQN.setZero();

    lambda_var.resize(Beta.size());
    nu_var.resize(Beta.size());
    c_slab.resize(numGroups);
    
    c_slab.setZero();
    lambda_var.setZero();
    nu_var.setZero();
    
    
    for(int jj=0; jj<numGroups; ++jj){
        hypTau[jj] = dist.inv_gamma_rate_rng(0.5, I_TAU0SQ);
        tau[jj]    = dist.inv_gamma_rate_rng(0.5 * v0t, v0t / hypTau[jj]);
        c_slab[jj] = dist.inv_scaled_chisq_rng(v0c, s02c);
    }
    lambda_var.array() = c_slab.sum() / (double)Mtot;
    
    if (rank == 0) {
        std::cout << "INFO Sampling initial hypTau" <<"\n";
        std::cout << "tau0: "<< tau0 << "\n";
        std::cout << "hypTau: "<< hypTau <<"\n";
        std::cout << "INFO Sampling initial tau" <<"\n";
        std::cout << "tau: " << tau <<"\n";
        std::cout << "INFO Sampling initial slab" <<"\n";
        std::cout << "c: " << c_slab <<"\n";
        std::cout << "v0L" << v0L << "\n";
        //if using newtonian uncomment this
        //SamplerLocalVar  lambdaSampler(dist,v0L,c_slab/(double)Mtot);    
        std::cout << "INFO Sampling initial lambda with values: "<< c_slab.sum()/(double)Mtot<<"\n";
        std::cout << "<------------ FH initialisation finished -------------------->\n";
    }

    components.resize(M);
    components.setZero();

    epsilon_restart.resize(Ntot - data.numNAs);
    epsilon_restart.setZero();

    gamma_restart.resize(data.X.cols());
    gamma_restart.setZero();

    markerI_restart.resize(M);
    std::fill(markerI_restart.begin(), markerI_restart.end(), 0);

    sigmaG.resize(numGroups);

    sigmaE = 0.0;

    VectorXd dirc(K);
    dirc.array() = 1.0;


    // Build global repartition of markers over the groups
    VectorXi MtotGrp(numGroups);
    MtotGrp.setZero();
    for (int i=0; i<Mtot; i++) {
        MtotGrp[groups[i]] += 1;
    }

    // In case of a restart, we first read the latest dumps
    // ----------------------------------------------------
    if (opt.restart) {

        init_from_restart(K, M, Mtot, Ntot - data.numNAs, MrankS, MrankL, opt.useXfilesInRestart);

        if (rank == 0)
            data.print_restart_banner(opt.mcmcOut.c_str(),  iteration_to_restart_from, iteration_start);

        dist.read_rng_state_from_file(rngfp);

        // Rename output files so that we do not erase from failed job!
        opt.mcmcOutNam += "_rs";
        opt.mcmcOut = opt.mcmcOutDir + "/" + opt.mcmcOutNam;
        set_output_filepaths(opt.mcmcOut, std::to_string(rank));

    } else {

        init_from_scratch();        

        dist.reset_rng((uint)(opt.seed + rank*1000));
        
        //EO: sample sigmaG and broadcast from rank 0 to all the others
        for(int i=0; i<numGroups; i++) {
            //sigmaG[i] = dist.unif_rng();
            sigmaG[i] = dist.beta_rng(1.0, 1.0);
        }
        check_mpi(MPI_Bcast(sigmaG.data(), sigmaG.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
    }

    // Set sigmaG of empty groups to zero
    for (int i=0; i<numGroups; i++)
        if (MtotGrp[i] == 0)  sigmaG[i] = 0.0;


    //TODO@EO: check need for the barriers
    MPI_Barrier(MPI_COMM_WORLD);

    set_list_of_files_to_tar(opt.mcmcOut, nranks);
    MPI_Barrier(MPI_COMM_WORLD);

    delete_output_files();
    MPI_Barrier(MPI_COMM_WORLD);

    open_output_files();


    // CHECK
    printf("rank %d, tau = %20.15f, hypTau = %20.15f\n", rank, tau.sum(), hypTau.sum());


    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();

    double tl = -mysecond();


    // Read the data (from sparse representation by default)
    // -----------------------------------------------------
    size_t *N1S, *N1L,  *N2S, *N2L,  *NMS, *NML;
    uint   *I1,         *I2,         *IM;

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


    size_t taskBytes = 0;

    string sparseOut = opt.get_sparse_output_filebase(rank);

    if (opt.readFromBedFile && !opt.readFromSparseFiles) {

        data.load_data_from_bed_file(opt.bedFile, Ntot, M, rank, MrankS[rank],
                                     N1S, N1L, I1,
                                     N2S, N2L, I2,
                                     NMS, NML, IM,
                                     taskBytes);
        //for (int i=0; i<M; i++)
        //    printf("OUT BED bef rank %02d, marker %7d: 1,2,M  S = %6lu, %6lu, %6lu, L = %6lu, %6lu, %6lu\n", rank, i, N1S[i], N2S[i], NMS[i], N1L[i], N2L[i], NML[i]);

    } else if (opt.readFromSparseFiles && !opt.readFromBedFile) {
        
        data.load_data_from_sparse_files(rank, nranks, M, MrankS, MrankL, sparseOut,
                                         N1S, N1L, I1,
                                         N2S, N2L, I2,
                                         NMS, NML, IM,
                                         taskBytes);
        //for (int i=0; i<M; i++)
        //    printf("OUT SPARSE bef rank %02d, marker %7d: 1,2,M  S = %6lu, %6lu, %6lu, L = %6lu, %6lu, %6lu\n", rank, i, N1S[i], N2S[i], NMS[i], N1L[i], N2L[i], NML[i]);

    } else if (opt.mixedRepresentation) {

        // Test on binary reading of BED file
        //EO: be aware that we use an inverted BED format!
        /*
        bitset<8> bo = 0b10011100;
        unsigned allele1, allele2;
        unsigned raw1, raw2;
        int k = 0;
        while (k < 7) {
            int miss = 0;
            raw1    = bo[k];
            allele1 = (!bo[k++]);
            raw2    = bo[k];
            allele2 = (!bo[k++]);
            if (allele1 == 0 && allele2 == 1) miss = 1;
            printf("raw bed = %d %d; allele1, allele2 = %d %d  miss? %d\n", raw1, raw2, allele1, allele2, miss);
        }
        */

        data.load_data_from_mixed_representations(opt.bedFile, sparseOut,
                                                  rank, nranks, Ntot, M, MrankS, MrankL,
                                                  N1S, N1L, I1,
                                                  N2S, N2L, I2,
                                                  NMS, NML, IM,
                                                  opt.threshold_fnz, USEBED,
                                                  taskBytes);

        // Store once for all the number of elements for markers stored as BED 
        for (int i=0; i<M; i++) {
            if (USEBED[i]) {
                nusebed += 1;
                size_t X1 = 0, X2 = 0, XM = 0;
                data.sparse_data_get_sizes_from_raw(reinterpret_cast<char*>(&I1[N1S[i]]), 1, snpLenByt, data.numNAs, X1, X2, XM);
                //printf("data.sparse_data_get_sizes_from_raw => (%2d, %3d): X1 = %9lu, X2 = %9lu, XM = %9lu < << <<<\n", rank, i, X1, X2, XM);

                //EO: N{1,2,M}L structures must not be changed anymore!
                //------------------------------------------------------
                N1L[i] = X1;  N2L[i] = X2;  NML[i] = XM;
            }
        }

    } else {
        printf("FATAL  : either BED, SPARSE or MIXED");
        exit(1);
    }

    tl += mysecond();


    sparse_info_t sparse_info;
    sparse_info.N1S = N1S;
    sparse_info.N1L = N1L;
    sparse_info.I1  = I1;
    sparse_info.N2S = N2S;
    sparse_info.N2L = N2L;
    sparse_info.I2  = I2;
    sparse_info.NMS = NMS;
    sparse_info.NML = NML;
    sparse_info.IM  = IM;



    //EO: BW cannot be evaluated properly in case of mixed representation as NA corrections
    //    are applied in the loading routine
    //
    if (opt.mixedRepresentation) {
        printf("INFO   : rank %3d took %.3f seconds to load  %lu bytes and apply NA corrections to BED markers\n", rank, tl, taskBytes);        
    } else {
        printf("INFO   : rank %3d took %.3f seconds to load  %lu bytes  =>  BW = %7.3f GB/s\n", rank, tl, taskBytes, (double)taskBytes * 1E-9 / tl);
    }
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);


    // For mixed representation get global number of markers dealt with as BED
    //
    if (opt.mixedRepresentation) {
        int nusebed_tot = 0;
        MPI_Reduce(&nusebed, &nusebed_tot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) printf("INFO   : there is %d markers using BED representation out of %d (%6.2f%%)\n", nusebed_tot, Mtot, (double)nusebed_tot/(double)Mtot * 100.0);
    }


    // Correct each marker for individuals with missing phenotype
    // ----------------------------------------------------------
    if (rank == 0) 
        printf("INFO   : BEFORE NA correction: Ntot = %d; snpLenByt = %zu [byte]; snpLenUint = %zu [uint]\n", Ntot, snpLenByt, snpLenUint);

    if (data.numNAs > 0) {

        double tna = MPI_Wtime();

        if (rank == 0)
            printf("INFO   : applying %d corrections to genotype data due to missing phenotype data (NAs)...\n", data.numNAs);

        data.sparse_data_correct_for_missing_phenotype(N1S, N1L, I1, M, USEBED);
        data.sparse_data_correct_for_missing_phenotype(N2S, N2L, I2, M, USEBED);
        data.sparse_data_correct_for_missing_phenotype(NMS, NML, IM, M, USEBED);

        MPI_Barrier(MPI_COMM_WORLD);

        tna = MPI_Wtime() - tna;

        if (rank == 0) {
            if (opt.mixedRepresentation) {
                printf("INFO   : ... finished applying NA corrections in %.2f seconds for SPARSE markers in mixed representation.\n", tna);
            } else {
                printf("INFO   : ... finished applying NA corrections in %.2f seconds.\n", tna);
            }
        }

        // Adjust N upon number of NAs
        Ntot -= data.numNAs;

        // Length of a marker in [byte] in BED representation (1 ind is 2 bits, so 1 byte is 4 inds)
        snpLenByt  = (Ntot %  4) ? Ntot /  4 + 1 : Ntot /  4;

        // Length of a marker in [uint] in BED representation (1 ind is 2 bits, so 1 uint is 16 inds)
        snpLenUint = (Ntot % 16) ? Ntot / 16 + 1 : Ntot / 16;
    }

    if (rank == 0) 
        printf("INFO   : AFTER  NA correction: Ntot = %d; snpLenByt = %zu [byte]; snpLenUint = %zu [uint]\n", Ntot, snpLenByt, snpLenUint);
    fflush(stdout);

    // Compute dalloc increment from NA adjusted structures
    //
    for (int i=0; i<M; i++) {
        if (USEBED[i]) { // Although N2L and NML do contain the info, I2 and IM were not allocated for BED markers! Done on the fly
            dalloc += (double) (snpLenUint * sizeof(uint)) * 1E-9;
        } else {
            dalloc += (double)(N1L[i] + N2L[i] + NML[i]) * (double) sizeof(uint) * 1E-9;
        }
    }


    // Compute statistics (from sparse info)
    //
    double  dN   = (double) Ntot;
    double  dNm1 = (double)(Ntot - 1);
    double* mave = (double*)_mm_malloc(size_t(M) * sizeof(double), 64);  check_malloc(mave, __LINE__, __FILE__);
    double* mstd = (double*)_mm_malloc(size_t(M) * sizeof(double), 64);  check_malloc(mstd, __LINE__, __FILE__);
    
    dalloc += 3 * size_t(M) * sizeof(double) * 1E-9;

    double tmp0, tmp1, tmp2;
    for (int i=0; i<M; ++i) {
        mave[i] = (double(N1L[i]) + 2.0 * double(N2L[i])) / (dN - double(NML[i]));
        tmp1 = double(N1L[i]) * (1.0 - mave[i]) * (1.0 - mave[i]);
        tmp2 = double(N2L[i]) * (2.0 - mave[i]) * (2.0 - mave[i]);
        tmp0 = double(Ntot - N1L[i] - N2L[i] - NML[i]) * (0.0 - mave[i]) * (0.0 - mave[i]);
        mstd[i] = sqrt(double(Ntot - 1) / (tmp0+tmp1+tmp2));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto et2 = std::chrono::high_resolution_clock::now();
    const auto dt2 = et2 - st2;
    const auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();
    if (rank == 0)
        printf("INFO   : overall time to preprocess the data: %.3f seconds\n", (double) du2 / 1000.0);


    // Build list of markers
    // ---------------------
    for (int i=0; i<M; ++i) 
        markerI.push_back(i);

    
    // Processing part
    // ---------------
    const auto st3 = std::chrono::high_resolution_clock::now();

    double   *y, *epsilon, *tmpEps, *deltaEps, *dEpsSum;
    const size_t NDB = size_t(Ntot) * sizeof(double);
    y          = (double*)_mm_malloc(NDB, 64);  check_malloc(y,          __LINE__, __FILE__);
    epsilon    = (double*)_mm_malloc(NDB, 64);  check_malloc(epsilon,    __LINE__, __FILE__);
    tmpEps     = (double*)_mm_malloc(NDB, 64);  check_malloc(tmpEps,     __LINE__, __FILE__);
    deltaEps   = (double*)_mm_malloc(NDB, 64);  check_malloc(deltaEps,   __LINE__, __FILE__);
    dEpsSum    = (double*)_mm_malloc(NDB, 64);  check_malloc(dEpsSum,    __LINE__, __FILE__);
    dalloc += NDB * 6 / 1E9;

    double totalloc = 0.0;
    MPI_Reduce(&dalloc, &totalloc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("INFO   : overall allocation %.3f GB\n", totalloc);


    set_array(dEpsSum, 0.0, Ntot);

    // Set gamma and xI, following a restart or not
    if (opt.covariates) {

        if (rank == 0)
            printf("INFO   : using covariate file: %s\n", opt.covariatesFile.c_str());

    	gamma = VectorXd(data.X.cols());
    	gamma.setZero();

        if (opt.restart) {
            for (int i=0; i<data.X.cols(); i++) {
                gamma[i] = gamma_restart[i];
                xI[i]    = xI_restart[i];
            }
        }
    }

    // //simulating data
    // VectorXd eta(3);
    // VectorXd pi(3);
    // int K_groups = 3;

    // eta(0) = 0.01;
    // eta(1) = 0.001;
    // eta(2) = 0.0001;

    // pi(0) = 0.05;
    // pi(1) = 0.1;
    // pi(2) = 0.85;
    // pi = pi / pi.sum();

    // //simulating beta_true
    // std::default_random_engine generator;
    // std::uniform_real_distribution<double> unif(0.0,1.0);
    // VectorXd beta_true(M);
    // for (int i=0; i<M; i++)
    // {
    //     double u = unif(generator);
    //     double c_sum = 0;
    //     for (int j=0; j<K_groups; j++)
    //     {   
    //         c_sum += pi(j);
    //         if (u <= c_sum)
    //         {
    //             if (eta(j) != 0)
    //             {
    //                 std::normal_distribution<double> gauss_beta(0.0,sqrt(eta(j))); //2nd parameter is stddev
    //                 beta_true(i) = gauss_beta(generator);
    //             }
    //             else
    //             {
    //                 beta_true(i) = 0;
    //             }
    //             break;
    //         }
    //     }
    // }

    // //generating noise
    // VectorXd noise(Ntot);
    // double sigma_noise = 0.01;
    // std::normal_distribution<double> gauss_noise(0.0,sigma_noise);
    // for (int i=0; i<Ntot; i++) 
    // {
    //    noise(i) = gauss_noise(generator);
    // }

    // //generating y
    // VectorXd y_phen =VectorXd::Zero(Ntot);
    // VectorXd xbeta = VectorXd::Zero(Ntot);
    // xbeta= BayesRRm::xbeta_mult(beta_true, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2);
    // y_phen = xbeta + noise;
    // cout << "y_phen(1): "<< y_phen(1)<< endl;

    // Copy, center and scale phenotype observations
    for (int i=0; i<Ntot; ++i) y[i] = data.y(i);
    center_and_scale(y, Ntot);

    // for(int i=0; i<Ntot; ++i) y[i] = y_phen[i];


    // In case of restart we reset epsilon to last dumped state (sigmaE as well, see init_from_restart)
    if (opt.restart) {
        for (int i=0; i<Ntot; ++i)  epsilon[i] = epsilon_restart[i];
        markerI = markerI_restart;
        mu      = mu_restart;
    } else {
        for (int i=0; i<Ntot; ++i)  epsilon[i] = y[i];
        sigmaE = 0.0;
        for (int i=0; i<Ntot; ++i)  sigmaE += epsilon[i] * epsilon[i];
        sigmaE = sigmaE / dN * 0.5;
    }
    //printf("sigmaE = %20.15f\n", sigmaE);
    //printf("mu = %20.15f\n", mu);


    //EO: for now, we use adav to monitor markers belonging to groups
    //    with sigmaG being null; so no need to dump additional information
    //    in case of restart
    adaV.setOnes();
    for (int i=0; i<M; i++) {
        if (sigmaG[groups[MrankS[rank] + i]] == 0.0) {
            adaV[i] = 0;
        }
    }


    VectorXd sum_beta_squaredNorm(numGroups);
    VectorXd beta_squaredNorm(numGroups);

    logL.resize(K);

    sigmaF = s02F;

    // A counter on previously saved thinned iterations
    uint n_thinned_saved = 0;

    // Main iteration loop
    // -------------------
    double tot_sync_ar1  = 0.0;
    double tot_sync_ar2  = 0.0;
    int    tot_nsync_ar1 = 0;
    int    tot_nsync_ar2 = 0;

    if (opt.bayesType == "bayesFHMPI") {
        assert(lambda_var.size() > 0);
        assert(lambda_var.size() == Beta.size());
    }

    VectorXd beta_est = VectorXd::Zero(M);

    for (uint iteration=iteration_start; iteration<opt.chainLength; iteration++) {

        //printf("@@@### ITERATION %5d\n", iteration);

        double start_it = MPI_Wtime();
        double it_sync_ar1  = 0.0;
        double it_sync_ar2  = 0.0;
        int    it_nsync_ar1 = 0;
        int    it_nsync_ar2 = 0;

        for (int i=0; i<Ntot; ++i) epsilon[i] += mu;

        double epssum  = 0.0;
        for (int i=0; i<Ntot; ++i) epssum += epsilon[i];
        //printf("epssum = %20.15f with Ntot=%d elements on iteration = %d\n", epssum, Ntot, iteration);

        // update mu
        mu = dist.norm_rng(epssum / dN, sigmaE / dN);
        //printf("it %d, rank %d: mu = %20.15f with dN = %10.1f, sigmaE = %20.15f\n", iteration, rank, mu, dN, sigmaE);
        //printf("it %d, rank %d: c_slab[0] = %20.15f\n", iteration, rank, c_slab[0]);

        // We substract again now epsilon =Y-mu-X*beta
        for (int i=0; i<Ntot; ++i) epsilon[i] -= mu;


        //EO: watch out, std::shuffle is not portable, so do no expect identical
        //    results between Intel and GCC when shuffling the markers is on!!
        //------------------------------------------------------------------------
        boost::uniform_int<> unii(0, M-1);
        boost::variate_generator< boost::mt19937&, boost::uniform_int<> > generator(dist.rng, unii);
        if (opt.shuffleMarkers) {
            boost::range::random_shuffle(markerI, generator);
        }

        m0.setZero();
        cass.setZero();


        for (int i=0; i<Ntot; ++i) tmpEps[i] = epsilon[i];

        double cumSumDeltaBetas       = 0.0;
        double task_sum_abs_deltabeta = 0.0;
        int    sinceLastSync          = 0;


        // Loop over (shuffled) markers
        // ----------------------------
        for (int j = 0; j < lmax; j++) {
	 
            sinceLastSync += 1;

            double deltaBeta  = 0.0;

            if (j < M) {
	        
                const int marker = markerI[j];
                int abs_mark_pos = MrankS[rank] + marker;  // absolute marker position
                double beta      = Beta(marker);
                double sigE_G    = sigmaE / sigmaG[groups[abs_mark_pos]];
                double sigG_E    = sigmaG[groups[abs_mark_pos]] / sigmaE;
                double i_2sigE   = 1.0 / (2.0 * sigmaE);

                double lambda_tilde = 0.0;

                if (opt.bayesType == "bayesFHMPI") {

                    // Now the variance per mixture changes
                    nu_var(marker) = dist.inv_gamma_rate_rng(0.5 + 0.5 * v0L, v0L / lambda_var[marker] + 1.0);//sample nu_var in case needed
                    
                    lambda_tilde   = tau[groups[abs_mark_pos]] * c_slab[groups[abs_mark_pos]] / (tau[groups[abs_mark_pos]] + c_slab[groups[abs_mark_pos]] * lambda_var[marker]);

                    if (opt.verbosity > 2) {
                        printf("iteration %d, marker %d, tau = %20.15f\n", iteration, marker, tau[groups[abs_mark_pos]]);
                        //std::cout << "marker      : " << marker <<"\n";
                        //std::cout << "tau         : " << tau    << "\n";
                        //std::cout << "c_slab      : " << c_slab << "\n";
                        //std::cout << "local       : " << lambda_var[marker] << "\n";
                        //std::cout << "lambda_tilde: " << lambda_tilde << "\n";
                    }
                }

                
                if (adaV[marker]) {

                    for (int i=1; i<=km1; ++i) {
                        
                        if (opt.bayesType == "bayesFHMPI") {
                            denom(i-1) = dNm1 + sigmaE / lambda_tilde;
                        } else {
                            denom(i-1) = dNm1 + sigE_G * cVaI(groups[abs_mark_pos], i);
                        }
                        //printf("it %d, rank %d, m %d: denom[%d] = %20.15f, cvai = %20.15f\n", iteration, rank, marker, i-1, denom(i-1), cVaI(groups[abs_mark_pos], i));
                    }

                    // Compute dot product
                    double num  = 0.0;
                    if (USEBED[marker]) {
                        avx_bed_dot_product(&I1[N1S[marker]], epsilon, Ntot, snpLenByt, mave[marker], mstd[marker], num);
                    } else {
                        num = sparse_dotprod(epsilon,
                                             I1, N1S[marker], N1L[marker],
                                             I2, N2S[marker], N2L[marker],
                                             IM, NMS[marker], NML[marker],
                                             mave[marker],    mstd[marker], Ntot, marker);
                    }
                    
                    //continue;
                    
                    num += beta * double(Ntot - 1);
                    //printf("it %d, rank %d, mark %d: num = %22.15f, %20.15f, %20.15f\n", iteration, rank, marker, num, mave[marker], mstd[marker]);

                    //muk for the other components is computed according to equations
                    muk.segment(1, km1) = num / denom.array();                    
                    //cout << "muk = " << endl << muk << endl;
                    
                    //first component probabilities remain unchanged
                    for (int i=0; i<K; i++)
                        logL[i] = log(estPi(groups[abs_mark_pos], i));

                    // Update the log likelihood for each component
                    for (int i=1; i<1+km1; i++) {
                        if (opt.bayesType == "bayesFHMPI") {
                            logL[i] = logL[i]
                                - 0.5 * log((lambda_tilde / sigmaE) * dNm1 + 1.0)
                                + muk[i] * num * i_2sigE;
                        } else {
                            logL[i] = logL[i]
                                - 0.5 * log(sigG_E * dNm1 * cVa(groups[abs_mark_pos], i) + 1.0)
                                +  muk[i] * num * i_2sigE;
                        }
                    }

                    double prob = dist.unif_rng();
                    //printf("%d/%d/%d  prob = %15.10f\n", iteration, rank, j, prob);

                    double acum = 0.0;
                    if (((logL.segment(1,km1).array()-logL[0]).abs().array() > 700.0).any()) {
                        acum = 0.0;
                    } else{
                        acum = 1.0 / ((logL.array()-logL[0]).exp().sum());
                    }
                    
                    //printf("it %d, marker %d, acum = %20.15f, prob = %20.15f\n", iteration, marker, acum, prob);
                    //continue;

                    Acum(marker) = acum;

                    for (int k=0; k<K; k++) {

                        if (prob <= acum || k == km1) { // if prob is less than acum or if we are already in the last mixt.

                            if (k==0) {
                                Beta(marker) = 0.0;
                            } else {
                                Beta(marker) = dist.norm_rng(muk[k], sigmaE/denom[k-1]);
                                //printf("@B@ beta update %4d/%4d/%4d muk[%4d] = %15.10f with prob=%15.10f <= acum = %15.10f, denom = %15.10f, sigmaE = %15.10f: beta = %15.10f\n", iteration, rank, marker, k, muk[k], prob, acum, denom[k-1], sigmaE, Beta(marker));
                            }
                            cass(groups[abs_mark_pos], k) += 1;
                            components[marker]  = k;
                            break;

                        } else {
                            //if too big or too small
                            if (k+1 >= K) {
                                printf("FATAL  : iteration %d, marker = %d, prob = %15.10f, acum = %15.10f logL overflow with %d => %d\n", iteration, marker, prob, acum, k+1, K);
                                MPI_Abort(MPI_COMM_WORLD, 1);
                            }

                            if (((logL.segment(k+1,K-(k+1)).array()-logL[k+1]).abs().array() > 700.0).any()) {
                                acum += 0.0d; // we compare next mixture to the others, if to big diff we skip
                            } else{
                                acum += 1.0d / ((logL.array()-logL[k+1]).exp().sum()); //if not , sample
                            }
                        }
                    }
                    
                } else { // end of adapative if daniel
                    Beta(marker) = 0.0;
                    Acum(marker) = 1.0; // probability of beta being 0 equals 1.0
                }
                
                fflush(stdout);

                
                double betaOld = beta;
                beta           = Beta(marker);
                deltaBeta      = betaOld - beta;
                //printf("iteration %3d, marker %5d: deltaBeta = %20.15f\n", iteration, marker, deltaBeta);
                
                // FHDT sample horseshoe local parameters
                // this does not depend on other betas, and the rest of the loop does
                // not depend on the results of this, so could be done in a different
                // thread
                if (opt.bayesType == "bayesFHMPI") {                   
                    lambda_var(marker) = dist.inv_gamma_rate_rng(0.5 + 0.5 * v0L, 0.5 * beta * beta / tau[groups[abs_mark_pos]] + v0L / nu_var(marker));
                }
                
                    
                // Compute delta epsilon
                if (deltaBeta != 0.0) {
                    if ((opt.bedSync || opt.sparseSync) && nranks > 1) {
                        mark2sync.push_back(marker);
                        dbet2sync.push_back(deltaBeta);
                    } else {
                        if (USEBED[marker]) {
                            bed_scaadd(&I1[N1S[marker]], Ntot, deltaBeta, mave[marker], mstd[marker], deltaEps);
                        } else {
                            sparse_scaadd(deltaEps, deltaBeta, 
                                          I1,  N1S[marker], N1L[marker],
                                          I2,  N2S[marker], N2L[marker],
                                          IM,  NMS[marker], NML[marker],
                                          mave[marker], mstd[marker], Ntot);
                        }
                        // Update local sum of delta epsilon
                        add_arrays(dEpsSum, deltaEps, Ntot);
                    }
                }

            } else {
                // for tasks with less markers than in biggest task, make contribution null
                deltaBeta = 0.0;
                set_array(deltaEps, 0.0, Ntot);
            }

            task_sum_abs_deltabeta += fabs(deltaBeta);


            // Check whether we have a non-zero beta somewhere
            //
            if (nranks > 1 && (sinceLastSync >= opt.syncRate || j == lmax-1)) {

                //EO: watch out this one for the time reported in sync stats
                //MPI_Barrier(MPI_COMM_WORLD);
                
                double tb = MPI_Wtime();
                check_mpi(MPI_Allreduce(&task_sum_abs_deltabeta, &cumSumDeltaBetas, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
                double te = MPI_Wtime();

                tot_sync_ar1  += te - tb;
                it_sync_ar1   += te - tb;
                tot_nsync_ar1 += 1;
                it_nsync_ar1  += 1;

            } else {

                cumSumDeltaBetas = task_sum_abs_deltabeta;
            }

            //printf("%d/%d/%d: deltaBeta = %20.15f = %10.7f - %10.7f; sumDeltaBetas = %15.10f\n", iteration, rank, marker, deltaBeta, betaOld, beta, cumSumDeltaBetas);
            //fflush(stdout);

            //continue;

            if ( (sinceLastSync >= opt.syncRate || j == lmax-1) && cumSumDeltaBetas != 0.0) {

                double tb = MPI_Wtime();

                delta_epsilon_exchange(opt.bedSync,  opt.sparseSync,
                                       mark2sync,    dbet2sync,
                                       mave,         mstd,
                                       snpLenByt,    snpLenUint, USEBED,
                                       &sparse_info,                                       
                                       Ntot,         data,
                                       dEpsSum,      tmpEps,
                                       epsilon);

                double te = MPI_Wtime();
                tot_sync_ar2  += te - tb;
                it_sync_ar2   += te - tb;
                tot_nsync_ar2 += 1;
                it_nsync_ar2  += 1;

                //double end_sync = MPI_Wtime();
                //printf("INFO   : synchronization time = %8.3f ms\n", (end_sync - beg_sync) * 1000.0);

                // Store epsilon state at last synchronization
                copy_array(tmpEps, epsilon, Ntot);

                set_array(dEpsSum, 0.0, Ntot);

                cumSumDeltaBetas       = 0.0;
                task_sum_abs_deltabeta = 0.0;
                
                sinceLastSync = 0;                
            }

        } // END PROCESSING OF ALL MARKERS

        //continue;
        
        beta_est += Beta;

        // Update beta squared norm by group
        beta_squaredNorm.setZero();
        for (int i=0; i<M; i++) {
            beta_squaredNorm[groups[MrankS[rank] + i]] += Beta[i] * Beta[i];
        }
        //printf("iteration %d, rank %d: beta_squaredNorm[0] = %20.15f\n", iteration, rank, beta_squaredNorm[0]);



        if (opt.bayesType == "bayesFHMPI") {
	    // Scaled sum of squares
            scaledBSQN.setZero();
	    sum_scaledBSQN.setZero();
#pragma novector
            for (int i = 0; i < M; i++) {
                scaledBSQN[groups[MrankS[rank] + i]] +=  (Beta[i] * Beta[i]) / lambda_var[i];
            }
            //printf("iteration %d, rank %d: scaledBSQN = %20.15f\n", iteration, rank, scaledBSQN);
        }

        // Transfer global to local
        // ------------------------
        if (nranks > 1) {
            MPI_Barrier(MPI_COMM_WORLD);
            check_mpi(MPI_Allreduce(beta_squaredNorm.data(), sum_beta_squaredNorm.data(), beta_squaredNorm.size(), MPI_DOUBLE,  MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            check_mpi(MPI_Allreduce(cass.data(),             sum_cass.data(),             cass.size(),             MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            beta_squaredNorm = sum_beta_squaredNorm;
            cass             = sum_cass;
	    if(opt.bayesType == "bayesFHMPI"){
	        check_mpi(MPI_Allreduce(scaledBSQN.data(), sum_scaledBSQN.data(), scaledBSQN.size(), MPI_DOUBLE,  MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
		scaledBSQN = sum_scaledBSQN;
	    }
        }

        // Update global parameters
        // ------------------------
        for (int i=0; i<numGroups; i++) {

            // Skip empty groups 
            if (MtotGrp[i] == 0)   continue;

            //printf("%d: m0 = %d - %d, v0G = %20.15f\n", i, MtotGrp[i], cass(i, 0), v0G);
            m0[i] = MtotGrp[i] - cass(i, 0);

            // Skip groups with m0 being null or empty cass (adaV in action)
            if (m0[i] == 0 || cass.row(i).sum() == 0) {
                for (int ii=0; ii<M; ii++) {
                    if (groups[MrankS[rank] + ii] == i) {
                        adaV[ii] = 0;
                    }
                }
                sigmaG[i] = 0.0;
                continue;
            }

            // AH: naive check that v0G and s02G were supplied from file
            if (opt.priorsFile != "") {
                v0G  = data.priors(i, 0);
                s02G = data.priors(i, 1);
            }

            // AH: similarly for dirichlet priors
            if (opt.dPriorsFile != "") {
                // get vector of parameters for the current group
                dirc = data.dPriors.row(i).transpose().array();
            }
           

            if (opt.bayesType == "bayesFHMPI") {
                // FHDT sample hyper parameters
                hypTau[i]    = dist.inv_gamma_rate_rng(0.5 + 0.5 * v0t, I_TAU0SQ + 1.0 / tau[i]);
                tau[i]       = dist.inv_gamma_rate_rng(0.5 * (m0[i] + v0t), v0t / hypTau[i] + (0.5 * scaledBSQN[i]));
                c_slab[i] = dist.inv_scaled_chisq_rng(v0c + (double)m0[i], (beta_squaredNorm[i] * (double)m0[i] + v0c * s02c) / (v0c + (double)m0[i]));
                sigmaG[i] = beta_squaredNorm[i];
            } else {                
                sigmaG[i] = dist.inv_scaled_chisq_rng(v0G + (double) m0[i], (beta_squaredNorm[i] * (double) m0[i] + v0G * s02G) / (v0G + (double) m0[i]));
            }         
            
            //printf("???? %d: %d, bs %20.15f, m0 %d -> sigmaG[i] = %20.15f, call(%20.15f, %20.15f)\n", rank, i, beta_squaredNorm[i], m0[i], sigmaG[i], v0G + (double) m0[i],  (beta_squaredNorm[i] * (double) m0[i] + v0G * s02G) / (v0G + (double) m0[i]));
            
            // we moved the pi update here to use the same loop
            VectorXd dirin = cass.row(i).transpose().array().cast<double>() + dirc.array();
            estPi.row(i) = dist.dirichlet_rng(dirin);
        }
        
        fflush(stdout);
        
        //printf("rank %d own sigmaG[0] = %20.15f with Mtot = %d and m0[0] = %d\n", rank, sigmaG[0], Mtot, int(m0[0]));

        // Broadcast sigmaG of rank 0
        check_mpi(MPI_Bcast(sigmaG.data(), sigmaG.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
	// If FH we have to broadcast the hyperparameters too
	if (opt.bayesType == "bayesFHMPI"){
	  check_mpi(MPI_Bcast(sigmaG.data(), sigmaG.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
	  check_mpi(MPI_Bcast(tau.data(), tau.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
	  check_mpi(MPI_Bcast(hypTau.data(), hypTau.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
	  check_mpi(MPI_Bcast(c_slab.data(), c_slab.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
	}
        //printf("rank %d has sigmaG = %15.10f\n", rank, sigmaG);
        //cout << "sigmaG = " << sigmaG << endl;


        if (rank == 0) {
            printf("\nINFO   : global cass on iteration %d:\n", iteration);
            for (int i=0; i<numGroups; i++) {
                printf("         MtotGrp[%3d] = %8d  | ", i, MtotGrp[i]);
                if (MtotGrp[i] == 0) {
                    printf(" (empty group)");
                } else if (sigmaG[i] == 0.0) {
                    printf(" excluded (sigmaG set to zero)");
                } else {
                    printf(" cass:");
                    for (int ii=0; ii<K; ii++) {
                        printf(" %8d", cass(i, ii));
                    }
                    assert(cass.row(i).sum() == MtotGrp[i]);
                }
                printf("\n");
            }
        }

        fflush(stdout);


        // For the fixed effects
        // ---------------------
        uint gamma_length = 0;
        
        if (opt.covariates) {
            
            if (iteration == 0 && rank == 0)
                cout << "COVARIATES with X of size " << data.X.rows() << "x" << data.X.cols() << endl;

            std::shuffle(xI.begin(), xI.end(), dist.rng);

            double gamma_old, num_f, denom_f;
            double sigE_sigF = sigmaE / sigmaF;

            gamma_length = data.X.cols();

            for (int i=0; i<gamma_length; i++) {

                gamma_old = gamma(xI[i]);
                num_f     = 0.0;
                denom_f   = 0.0;

                for (int k=0; k<Ntot; k++) {
                    num_f += data.X(k, xI[i]) * (epsilon[k] + gamma_old * data.X(k, xI[i]));
                }

                denom_f      = dNm1 + sigE_sigF;
                gamma(xI[i]) = dist.norm_rng(num_f/denom_f, sigmaE/denom_f);

                for (int k = 0; k<Ntot ; k++) {
                    epsilon[k] = epsilon[k] + (gamma_old - gamma(xI[i])) * data.X(k, xI[i]);
                    //cout << "adding " << (gamma_old - gamma(xI[i])) * data.X(k, xI[i]) << endl;
                }
            }
            //the next line should be uncommented if we want to use a prior for the ridge parameter.
            //sigmaF = inv_scaled_chisq_rng(0.001 + F, (gamma.squaredNorm() + 0.001)/(0.001+F));
            sigmaF = s02F;
        }

        MPI_Barrier(MPI_COMM_WORLD);

        double e_sqn = 0.0d;
        for (int i=0; i<Ntot; ++i) e_sqn += epsilon[i] * epsilon[i];
        //printf("e_sqn = %20.15f, v0E = %20.15f, s02E = %20.15f\n", e_sqn, v0E, s02E);

        //EO: sample sigmaE and broadcast the one from rank 0 to all the others
        sigmaE  = dist.inv_scaled_chisq_rng(v0E+dN, (e_sqn + v0E*s02E)/(v0E+dN));
        
        check_mpi(MPI_Bcast(&sigmaE, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);
        //printf("sigmaE = %20.15f\n", sigmaE);

        double end_it = MPI_Wtime();
        //if (rank == 0) printf("TIME_IT: Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it);


        //printf("%d epssqn = %15.10f %15.10f %15.10f %6d => %15.10f\n", iteration, e_sqn, v0E, s02E, Ntot, sigmaE);
        if (rank%10==0) {
            printf("RESULT : it %4d, rank %4d: proc = %9.3f s, sync = %9.3f (%9.3f + %9.3f), n_sync = %8d (%8d + %8d) (%7.3f / %7.3f), sigmaG = %15.10f, sigmaE = %15.10f, betasq = %15.10f, m0 = %10d\n",
                   iteration, rank, end_it-start_it,
                   it_sync_ar1  + it_sync_ar2,  it_sync_ar1,  it_sync_ar2,
                   it_nsync_ar1 + it_nsync_ar2, it_nsync_ar1, it_nsync_ar2,
                   (it_sync_ar1) / double(it_nsync_ar1) * 1000.0,
                   (it_sync_ar2) / double(it_nsync_ar2) * 1000.0,
                   sigmaG.sum(), sigmaE, beta_squaredNorm.sum(), m0.sum());
            fflush(stdout);
        }


        // Broadcast estPi from rank 0
        check_mpi(MPI_Bcast(estPi.data(), estPi.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD), __LINE__, __FILE__);

        
        // Write output files with full history written at modulo opt.thin
        if (iteration%opt.thin == 0) {

            if (rank == 0) {
                write_ofile_csv(fh(csvfp), iteration, sigmaG, sigmaE, m0, n_thinned_saved, estPi); // txt
                write_ofile_out(fh(outfp), iteration, sigmaG, sigmaE, m0, n_thinned_saved, estPi); // bin
            }

            write_ofile_h1(fh(betfp), rank, Mtot, iteration, n_thinned_saved, MrankS[rank], M, Beta.data(),       MPI_DOUBLE);
            write_ofile_h1(fh(acufp), rank, Mtot, iteration, n_thinned_saved, MrankS[rank], M, Acum.data(),       MPI_DOUBLE);
            write_ofile_h1(fh(cpnfp), rank, Mtot, iteration, n_thinned_saved, MrankS[rank], M, components.data(), MPI_INTEGER);

            write_ofile_s1(fh(musfp), iteration, n_thinned_saved, mu,     MPI_DOUBLE);

            //printf("writing tau at it %d, rank %d = %20.15f - %20.15f\n", iteration, rank, tau, hypTau);

            write_ofile_hsa(fh(cslfp), iteration, n_thinned_saved, c_slab.size(), c_slab.data(), MPI_DOUBLE);
	    write_ofile_hsa(fh(taufp), iteration, n_thinned_saved, tau.size(), tau.data(), MPI_DOUBLE);
	    write_ofile_hsa(fh(htafp), iteration, n_thinned_saved, hypTau.size(), hypTau.data(), MPI_DOUBLE);

            n_thinned_saved += 1;
        }


        // Single-line files overwritten at modulo opt.save
        if (iteration > 0 && iteration%opt.save == 0) {

            // task-wise files
            dist.write_rng_state_to_file(rngfp);
            write_ofile_t1(fh(epsfp), iteration, Ntot,           epsilon,        MPI_DOUBLE);
            write_ofile_t1(fh(mrkfp), iteration, markerI.size(), markerI.data(), MPI_UNSIGNED);
            if (opt.covariates) {
                write_ofile_t1(fh(gamfp), iteration, gamma_length,  gamma.data(), MPI_DOUBLE);
                write_ofile_t1(fh(xivfp), iteration, gamma_length,  xI.data(),    MPI_DOUBLE);
            }
            if (opt.bayesType == "bayesFHMPI") {
                write_ofile_t1(fh(lbvfp), iteration, M, lambda_var.data(), MPI_DOUBLE);
                write_ofile_t1(fh(nuvfp), iteration, M, nu_var.data(),     MPI_DOUBLE);
            }


            // task-shared files
            write_ofile_t2(fh(xbetfp), rank, MrankS[rank], Mtot, iteration, M, Beta.data(),       MPI_DOUBLE);
            write_ofile_t2(fh(xcpnfp), rank, MrankS[rank], Mtot, iteration, M, components.data(), MPI_INTEGER);


            //EO system call to create a tarball of the dump
            //----------------------------------------------
            MPI_Barrier(MPI_COMM_WORLD);
            if (rank == 0) {
                time_t now = time(0);
                tm *   ltm = localtime(&now);
                int    n   = 0;
                char tar[LENBUF];

                n=sprintf(tar, "dump_%s_%05d__%4d-%02d-%02d_%02d-%02d-%02d.tar",
                          opt.mcmcOutNam.c_str(), iteration,
                          1900 + ltm->tm_year, 1 + ltm->tm_mon, ltm->tm_mday,
                          ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
                assert(n > 0);

                printf("INFO   : will create tarball %s in %s with file listed in %s.\n",
                       tar, opt.mcmcOutDir.c_str(), lstfp.c_str());
                //std::system(("ls " + opt.mcmcOut + ".*").c_str());
                //string cmd = "tar -czf " + opt.mcmcOutDir + "/tarballs/" + targz + " -T " + lstfp;
                string cmd = "tar -cf " + opt.mcmcOutDir + "/tarballs/" + tar + " -T " + lstfp + " 2>/dev/null";
                //cout << "cmd >>" << cmd << "<<" << endl;
                std::system(cmd.c_str());
            }

            MPI_Barrier(MPI_COMM_WORLD);
        }

        //double end_it = MPI_Wtime();
        //if (rank == 0) printf("TIME_IT: Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it);

        MPI_Barrier(MPI_COMM_WORLD);
    }

    beta_est /= opt.chainLength;
    // cout << "(not relevant) ||beta-beta_true||/ ||beta_true||: "<< (beta_true - beta_est).lpNorm<2>() / beta_true.lpNorm<2>() << endl;
    // cout << "(not relevant) l2 prediction error normalized (beta_true): "<< (y_phen - xbeta_mult(beta_est, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2)).lpNorm<2>() / (y_phen - xbeta_mult(beta_true, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2)).lpNorm<2>() << endl;
    // cout << "(not relevant) ||beta-beta_true||/ sqrt(M): "<< (beta_true - beta_est).lpNorm<2>() / sqrt(M) << endl;
    // cout << "l2 prediction error: " << (y_phen - xbeta_mult(beta_est, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2)).lpNorm<2>() << endl;
    // cout << "l2 beta_true prediction error: " << (y_phen - xbeta_mult(beta_true, Ntot, M, N1S, N1L, N2S, N2L, mstd, mave, I1, I2)).lpNorm<2>() << endl;


    close_output_files();


    // Release memory
    _mm_free(y);
    _mm_free(epsilon);
    _mm_free(tmpEps);
    _mm_free(deltaEps);
    _mm_free(dEpsSum);
    _mm_free(mave);
    _mm_free(mstd);
    _mm_free(USEBED);
    _mm_free(N1S);
    _mm_free(N1L);
    _mm_free(I1);
    _mm_free(N2S);
    _mm_free(N2L);
    _mm_free(I2);
    _mm_free(NMS);
    _mm_free(NML);
    _mm_free(IM);


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



MPI_File BayesRRm::fh(const std::string fp) {

    return get_fh(file_handlers, fp);
}


void BayesRRm::set_output_filepaths(const string mcmcOut, const string rank_str) {

    lstfp  = mcmcOut + ".lst";

    csvfp  = mcmcOut + ".csv";
    outfp  = mcmcOut + ".out";
    betfp  = mcmcOut + ".bet";
    xbetfp = mcmcOut + ".xbet";
    cpnfp  = mcmcOut + ".cpn";
    xcpnfp = mcmcOut + ".xcpn";
    acufp  = mcmcOut + ".acu";

    rngfp  = mcmcOut + ".rng." + rank_str;
    mrkfp  = mcmcOut + ".mrk." + rank_str;
    xivfp  = mcmcOut + ".xiv." + rank_str;
    epsfp  = mcmcOut + ".eps." + rank_str;
    gamfp  = mcmcOut + ".gam." + rank_str;
    musfp  = mcmcOut + ".mus." + rank_str;
    lbvfp  = mcmcOut + ".lbv." + rank_str;
    nuvfp  = mcmcOut + ".nuv." + rank_str;
    cslfp  = mcmcOut + ".csl." + rank_str;
    taufp  = mcmcOut + ".tau." + rank_str;
    htafp  = mcmcOut + ".hta." + rank_str;
    

    world_files.clear();

    world_files.push_back(csvfp);
    world_files.push_back(outfp);
    world_files.push_back(betfp);
    world_files.push_back(xbetfp);
    world_files.push_back(cpnfp);
    world_files.push_back(xcpnfp);
    world_files.push_back(acufp);

    self_files.clear();

    self_files.push_back(rngfp);
    self_files.push_back(mrkfp);
    self_files.push_back(xivfp);
    self_files.push_back(epsfp);
    self_files.push_back(gamfp);
    self_files.push_back(musfp);
    self_files.push_back(lbvfp);
    self_files.push_back(nuvfp);
    self_files.push_back(cslfp);
    self_files.push_back(taufp);
    self_files.push_back(htafp);
}

void BayesRRm::open_output_files() {

    file_handlers.clear();

    open_output_files_(world_files);
    open_output_files_(self_files);

}

void BayesRRm::open_output_files_(const std::vector<string> files) {

    for (auto&& f: files) {

        MPI_File fh;
        MPI_Comm comm = MPI_COMM_WORLD;

        if (is_self_file(f)) comm = MPI_COMM_SELF;

        check_mpi(MPI_File_open(comm,
                                f.c_str(),  
                                MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL,
                                MPI_INFO_NULL,
                                &fh)
                  ,  __LINE__, __FILE__);
        
        file_handlers.insert({f, fh});
    }
}


void BayesRRm::close_output_files() {

    close_output_files_(world_files);
    close_output_files_(self_files);
}


void BayesRRm::close_output_files_(const std::vector<string> files) {

    for (auto&& f: files) {
        fh_it fit = file_handlers.find(f);
        if (fit == file_handlers.end()) {
            cout << "*FATAL*: file " << f << " not found in file_handlers map!" << endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        check_mpi(MPI_File_close(&fit->second),  __LINE__, __FILE__);
    }
}


void::BayesRRm::delete_output_files() {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        for (auto&& f: world_files) {
            //std::cout << "deleting WORLD file: " << f << '\n';
            MPI_File_delete(f.c_str(), MPI_INFO_NULL);
        }
    }

    for (auto&& f: self_files) {
        //std::cout << "deleting SELF file: " << f << '\n';
        MPI_File_delete(f.c_str(), MPI_INFO_NULL);
    }
}


void BayesRRm::set_local_filehandler(MPI_File &fh, const std::string fp) {
    
    fh_it fit = file_handlers.find(fp);

    if (fit != file_handlers.end()) {
        fh = fit->second;
        cout << "found fh for file " << fp << ", " << fit->first << endl;
    } else {
        printf("*FATAL*: file %s not found in file_handlers map", fp.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


bool BayesRRm::is_world_file(const string fp) {

    return std::find(world_files.begin(), world_files.end(), fp) != world_files.end();
}


bool BayesRRm::is_self_file(const string fp) {

    return std::find(self_files.begin(), self_files.end(), fp) != self_files.end();
}


void BayesRRm::set_list_of_files_to_tar(const string mcmcOut, const int nranks) {
    
    ofstream listFile;
    
    listFile.open(lstfp);
    listFile << csvfp  << "\n";
    listFile << outfp  << "\n";
    listFile << xbetfp << "\n"; // Only tar the last saved iteration, no need for full history
    listFile << xcpnfp << "\n"; // Idem
    listFile << acufp  << "\n";

    for (int i=0; i<nranks; i++) {
        listFile << mcmcOut + ".rng." + std::to_string(i) << "\n";
        listFile << mcmcOut + ".mrk." + std::to_string(i) << "\n";
        listFile << mcmcOut + ".xiv." + std::to_string(i) << "\n";
        listFile << mcmcOut + ".eps." + std::to_string(i) << "\n";
        listFile << mcmcOut + ".gam." + std::to_string(i) << "\n";
        listFile << mcmcOut + ".mus." + std::to_string(i) << "\n";
    }

    listFile.close();
}


//EO: rough check on RAM requirements.
//    Simulates the RAM usage on a single node to see if would fit,
//    assuming at least the same amount of RAM is available on the nodes
//    that would be assigned to the job
//----------------------------------------------------------------------
int BayesRRm::checkRamUsage() {

    int nranks;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    if (opt.checkRam && nranks != 1) {
        printf("#FATAL#: --check-RAM option runs only in single task mode (SIMULATION of --check-RAM-tasks with max --check-RAM-tasks-per-node tasks per node)!\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (opt.checkRamTasks <= 0) {
        printf("#FATAL#: --check-RAM-tasks must be strictly positive! Was %d\n", opt.checkRamTasks);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if (opt.checkRamTpn <= 0) {
        printf("#FATAL#: --check-RAM-tasks-per-node must be strictly positive! Was %d\n", opt.checkRamTpn);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    uint Mtot = data.set_Mtot(rank, opt);

    // Alloc memory for sparse representation
    size_t *N1S, *N1L,  *N2S, *N2L,  *NMS, *NML;
    N1S = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(N1S, __LINE__, __FILE__);
    N1L = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(N1L, __LINE__, __FILE__);
    N2S = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(N2S, __LINE__, __FILE__);
    N2L = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(N2L, __LINE__, __FILE__);
    NMS = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(NMS, __LINE__, __FILE__);
    NML = (size_t*)_mm_malloc(size_t(Mtot) * sizeof(size_t), 64);  check_malloc(NML, __LINE__, __FILE__);

    MPI_File   slfh;
    MPI_Status status;

    string sparseOut = opt.get_sparse_output_filebase(rank);

    string sl;

    sl = sparseOut + ".sl1";
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &slfh), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(slfh, 0, N1L, Mtot, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&slfh), __LINE__, __FILE__);
    sl = sparseOut + ".sl2";
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &slfh), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(slfh, 0, N2L, Mtot, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&slfh), __LINE__, __FILE__);
    sl = sparseOut + ".slm";
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &slfh), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(slfh, 0, NML, Mtot, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&slfh), __LINE__, __FILE__);

    // Given
    int tpn = 0;
    tpn    = opt.checkRamTpn;
    nranks = opt.checkRamTasks;
    if (opt.markerBlocksFile != "") nranks = data.numBlocks;

    int nnodes = int(ceil(double(nranks)/double(tpn)));
    printf("INFO  : will simulate %d ranks on %d nodes with max %d tasks per node.\n", nranks, nnodes, tpn);

    int proctasks = 0;

    printf("Estimation RAM usage when dataset is processed with %2d nodes and %2d tasks per node\n", nnodes, tpn);

    int MrankS[nranks], MrankL[nranks], lmin = 1E9, lmax = 0;
    assign_blocks_to_tasks(data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot, nranks, rank, MrankS, MrankL, lmin, lmax);
    printf("INFO   : longest  task has %d markers.\n", lmax);
    printf("INFO   : smallest task has %d markers.\n", lmin);

    double min = 1E9, max = 0.0;
    int nodemin = 0, nodemax = 0;

    // Replicate SLURM block task assignment strategy
    // ----------------------------------------------
    const int nfull = nranks + nnodes * (1 - tpn);
    printf("INFO   : number of nodes fully loaded: %d\n", nfull);

    // Save max
    const int tpnmax = tpn;

    for (int node=0; node<nnodes; node++) {

        double ramnode = 0.0;

        if (node >= nfull) tpn = tpnmax - 1;

        // Array of pointers allocated memory
        uint *allocs1[tpn], *allocs2[tpn], *allocsm[tpn];

        for (int i=0; i<tpn; i++) {
            size_t n1 = 0, n2 = 0, nm = 0;
            for (int m=0; m<MrankL[node*tpn + i]; m++) {
                n1 += N1L[MrankS[node*tpn + i] + m];
                n2 += N2L[MrankS[node*tpn + i] + m];
                nm += NML[MrankS[node*tpn + i] + m];
            }
            double GB = double((n1+n2+nm)*sizeof(uint))*1E-9;
            ramnode += GB;
            printf("   - t %3d  n %2d will attempt to alloc %.3f + %.3f + %.3f GB of RAM\n", i, node, n1*sizeof(uint)*1E-9,  n2*sizeof(uint)*1E-9,  nm*sizeof(uint)*1E-9);

            allocs1[i] = (uint*)_mm_malloc(n1 * sizeof(uint), 64);  check_malloc(allocs1[i], __LINE__, __FILE__);
            allocs2[i] = (uint*)_mm_malloc(n2 * sizeof(uint), 64);  check_malloc(allocs2[i], __LINE__, __FILE__);
            allocsm[i] = (uint*)_mm_malloc(nm * sizeof(uint), 64);  check_malloc(allocsm[i], __LINE__, __FILE__);

            printf("   - t %3d  n %2d sm %7d  l %6d markers. Number of 1s: %15lu, 2s: %15lu, ms: %15lu => RAM: %7.3f GB; RAM on node: %7.3f with %d tasks\n", i, node, MrankS[node*tpn + i], MrankL[node*tpn + i], n1, n2, nm, GB, ramnode, tpn);

            proctasks++;
        }

        // free memory on the node
        for (int i=0; i<tpn; i++) {
            _mm_free(allocs1[i]);
            _mm_free(allocs2[i]);
            _mm_free(allocsm[i]);
        }

        if (ramnode < min) { min = ramnode; nodemin = node; }
        if (ramnode > max) { max = ramnode; nodemax = node; }

    }

    if (proctasks != nranks) {
        printf("#FATAL#: Cannot fit %d tasks on %d nodes with %d x %d + %d x %d tasks per node! Ended up with %d.\n", nranks, nnodes, nfull, tpnmax, nnodes-nfull, tpn, proctasks);
    }

    printf("\n");
    printf("    => max RAM required on a node will be %7.3f GB on node %d\n", max, nodemax);
    printf("    => setting up your sbatch with %d tasks and %d tasks per node should work; Will require %d nodes!\n", nranks, tpnmax, nnodes);
    printf("\n");

    // Free previously allocated memory
    _mm_free(N1S); _mm_free(N1L);
    _mm_free(N2S); _mm_free(N2L);
    _mm_free(NMS); _mm_free(NML);

    return 0;
}
