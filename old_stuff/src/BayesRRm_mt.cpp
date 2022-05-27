#include <cstdlib>
#include "BayesRRm.h"
#include "BayesRRm_mt.h"
#include "data.hpp"
#include "distributions_boost.hpp"
#include "options.hpp"
#include "samplewriter.h"
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
#include <ctime>
//#include <mpi.h>
//#include "mpi_utils.hpp"
#include <omp.h>
#include "dense.hpp"

BayesRRm_mt::~BayesRRm_mt()
{
}


void partial_sparse_dotprod_mt(const double*  __restrict__ vin1,
                               const uint8_t* __restrict__ mask,
                               const uint*    __restrict__ IX,
                               const size_t   NXS,
                               const size_t   NXL,
                               const bool     interleave,
                               const int      NT,
                               const int      Ntot,
                               const double   fac,
                               double*        __restrict__ m8) {
    
    //double t1 = -mysecond();
    //for (int ii=0; ii<1024; ii++) {

    if (interleave) {
#ifdef __INTEL_COMPILER
        __assume_aligned(vin1, 64);
        __assume_aligned(m8,   64);
        __assume_aligned(IX,   64);
        __assume_aligned(mask, 64);
#endif
        for (size_t i=NXS; i<NXS+NXL; ++i) {
            const uint index = IX[i];
            const double* p = &vin1[NT*index];
            for (int j=0; j < NT; j++) {
                if (mask[NT*index + j] == 1) {
                    m8[j] += fac * *(p + j);
                }
            }
        }        

    } else {

        //TODO: OMP task?

        for (int i=0; i<NT; i++) {
            const uint ioff = i * Ntot;
#ifdef __INTEL_COMPILER
            __assume_aligned(vin1, 64);
            __assume_aligned(IX,   64);
            __assume_aligned(mask, 64);
            __assume_aligned(m8,   64);
#endif
#ifdef _OPENMP
#pragma omp parallel for reduction(+: m8[i])
#endif
            for (size_t j=NXS; j<NXS+NXL; j++) {
                //if (mask[ioff + IX[j]] == 1) {
                m8[i] += fac * vin1[ioff + IX[j]] * (double)mask[ioff + IX[j]];
                    //}
            }
        }
    }
}
                          

void BayesRRm_mt::sparse_dotprod_mt(const double* __restrict__ vin1, const uint8_t* __restrict__ mask,
                                    const uint*   __restrict__ I1,   const size_t __restrict__ N1S,  const size_t __restrict__ N1L,
                                    const uint*   __restrict__ I2,   const size_t __restrict__ N2S,  const size_t __restrict__ N2L,
                                    const uint*   __restrict__ IM,   const size_t NMS,  const size_t NML,
                                    const double  mu, const double sig_inv, const int Ntot, const int marker,
                                    double* __restrict__ m8, const int NT, const bool interleave) {
    
    double syt = 0.0;
   
    for (int i=0; i<NT; i++)
        m8[i] = 0.0;

    partial_sparse_dotprod_mt(vin1, mask, I1, N1S, N1L, interleave, NT, Ntot, 1.0, m8);
    partial_sparse_dotprod_mt(vin1, mask, I2, N2S, N2L, interleave, NT, Ntot, 2.0, m8);

    //double t1 = -mysecond();
    //for (int ii=0; ii<1024; ii++) {}
    //t1 += mysecond();
    //printf("kernel 1 BW = %g\n", double(N1L)*8.*sizeof(double) / 1024. / 1024. / t1);

    for (int j=0; j<NT; j++)
        m8[j] *= sig_inv;

    double* syt8 = (double*) _mm_malloc(8, 64);  check_malloc(syt8, __LINE__, __FILE__);

    for (int j=0; j < NT; j++)
        syt8[j] = 0.0;

    const int NNT = NT * Ntot;

    sum_mt_vector_elements_f64(vin1, NT, Ntot, interleave, syt8);

    /*
    if (interleave) {
        for (int j=0; j < NT; j++) {
        __assume_aligned(vin1, 64);
        __assume_aligned(syt8, 64);
#ifdef _OPENMP_XXX
#pragma omp parallel for
#endif
            for (int i=0; i<NNT; i+=NT) {
                syt8[j] += vin1[i + j];
            }
        }
    } else {

        for (int i=0; i<NT; i++) {
            const int ioff = i * Ntot;            
            syt8[i] = sum_array_elements(&vin1[ioff], Ntot);
        }
    }
    */

    partial_sparse_dotprod_mt(vin1, mask, IM, NMS, NML, interleave, NT, Ntot, -1.0, syt8);

    for (int j=0; j<NT; j++)
        m8[j] -= mu * sig_inv * syt8[j];

    _mm_free(syt8);
}




void partial_sparse_scaadd_mt(const uint*   __restrict__ IX,
                              const size_t               NXS,
                              const size_t               NXL,
                              const bool                 interleave,
                              const int                  NT,
                              const int                  Ntot,
                              const double* __restrict__ aux,
                              double*       __restrict__ vout) {
    
    if (interleave) {
        for (size_t i=NXS; i<NXS+NXL; ++i) {
            const uint index = IX[i];
            double* p = &vout[NT*index];
#pragma unroll
            for (int j=0; j<NT; j++) {
                *(p + j) = aux[j];
            }
        }
    } else {
        for (int i=0; i<NT; i++) {
            for (size_t j=NXS; j<NXS+NXL; j++) {
                vout[i * Ntot + IX[j]] = aux[i];
            }
        }
    }
}

void BayesRRm_mt::sparse_scaadd_mt(double*       __restrict__ vout,
                                   const double* __restrict__ dMULT,
                                   const uint*   __restrict__ I1,    const size_t N1S, const size_t N1L,
                                   const uint*   __restrict__ I2,    const size_t N2S, const size_t N2L,
                                   const uint*   __restrict__ IM,    const size_t NMS, const size_t NML,
                                   const double  mu,                 const double sig_inv,
                                   const int     Ntot,               const int NT,     const bool interleave) {
    
    // Check whether any trait needs an update
    bool all_zeros = true;
    for (int i=0; i<NT; i++) {
        if (dMULT[i] != 0.0) {
            all_zeros = false;
            break;
        }
    }

    if (all_zeros) {

        set_array(vout, 0.0, NT * Ntot);

    } else {
        
        double* aux = (double*)_mm_malloc(NT * sizeof(double), 64);  check_malloc(aux, __LINE__, __FILE__);

        for (int i=0; i<NT; i++)
            aux[i] = mu * sig_inv * dMULT[i];
        set_mt_vector_f64(vout, aux, NT, Ntot, interleave);

        for (int i=0; i<NT; i++)
            aux[i] = 0.0;
        partial_sparse_scaadd_mt(IM, NMS, NML, interleave, NT, Ntot, aux, vout);

        for (int i=0; i<NT; i++)
            aux[i] = dMULT[i] * (1.0 - mu) * sig_inv;
        partial_sparse_scaadd_mt(I1, N1S, N1L, interleave, NT, Ntot, aux, vout);

        for (int i=0; i<NT; i++)
            aux[i] = dMULT[i] * (2.0 - mu) * sig_inv;
        partial_sparse_scaadd_mt(I2, N2S, N2L, interleave, NT, Ntot, aux, vout);

        _mm_free(aux);
    }
}

void BayesRRm_mt::set_mt_vector_f64(double* __restrict__ vec,
                                    const double*        val,
                                    const int            NT,
                                    const int            N,
                                    const bool           interleave) {
    
    if (interleave) {
        const int NNT = N * NT;
#ifdef __INTEL_COMPILER
        __assume_aligned(vec, 64);
        __assume_aligned(val,  64);
#endif
        for (int i=0; i<NNT; i+=NT) {
#pragma unroll
            for (int j=0; j<NT; j++) {
                vec[i + j] = -val[j];
            }
        }
    } else {
        for (int i=0; i<NT; i++) {
            for (int j=0; j<N; j++) {
                vec[i * N + j] = -val[i];
            }
        }
    }    
}


void BayesRRm_mt::sum_mt_vector_elements_f64(const double* __restrict__ vec,
                                             const int                  NT,
                                             const int                  N,
                                             const bool                 interleave,
                                             double*       __restrict__ syt8) {
    if (interleave) {
        const int NNT = NT * N;
        for (int j=0; j < NT; j++) {
#ifdef __INTEL_COMPILER
        __assume_aligned(vec, 64);
        __assume_aligned(syt8, 64);
#endif
#ifdef _OPENMP
#pragma omp parallel for simd reduction(+: syt8[j])
#endif
            for (int i=0; i<NNT; i+=NT) {
                syt8[j] += vec[i + j];
            }
        }
    } else {
        for (int i=0; i<NT; i++) {
            const int ioff = i * N;            
            syt8[i] = sum_array_elements(&vec[ioff], N);
        }
    }
}




/*
 *----------------------------
 * EO: MPI GIBBS MULTI-TRAITS
 *----------------------------
 * Notes: as opposed to the original version of the Gibbs sampling and its single-trait
 * MPI version, the MPI multi-trait handles NAs in phenotypes on-the-fly, that is they 
 * are not removed at reading time but kept in, meaning that the sparse representation 
 * also contains entries for the phenotype-specific NAs.
 */
int BayesRRm_mt::runMpiGibbsMultiTraits() {

#ifdef _OPENMP
    int tn = omp_get_thread_num();
    printf("MPI GIBBS MULTI-TRAITS on thread %d\n", tn);
    //#warning "Using OpenMP"
#endif
    char   buff[LENBUF]; 
    int    nranks, rank, name_len, result;
    double dalloc = 0.0;

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_File   outfh, betfh, epsfh, cpnfh, acufh, mrkfh;
    MPI_Status status;
    MPI_Info   info;
    MPI_Offset offset, betoff, cpnoff, epsoff;

    // Set up processing options
    // -------------------------
    if (rank < 0) {
        opt.printBanner();
        opt.printProcessingOptions();
    }

    // Set Ntot and Mtot
    // -----------------
    uint Ntot = set_Ntot(rank, opt, data);
    const uint Mtot = set_Mtot(rank);
    if (rank == 0)
        printf("INFO   : Full dataset includes Mtot=%d markers and Ntot=%d individuals.\n", Mtot, Ntot);


    // Define global marker indexing
    int MrankS[nranks], MrankL[nranks], lmin = 1E9, lmax = 0;
    mpi_assign_blocks_to_tasks(data.numBlocks, data.blocksStarts, data.blocksEnds, Mtot, nranks, rank, MrankS, MrankL, lmin, lmax);

    // M is the number of markers to be handled by the task
    uint M = MrankL[rank];
    printf("INFO   : rank %4d will handle a block of %6d markers starting at %d\n", rank, MrankL[rank], MrankS[rank]);


    // EO: Define blocks of individuals (for dumping epsilon)
    // Note: hack the marker block definition function to this end
    // Note: as opposed to the single-trait version, we need to consider
    //       all individuals here, so we need not to adjust for NAs.
    // --------------------------------------------------------------------
    int IrankS[nranks], IrankL[nranks];
    mpi_define_blocks_of_markers(Ntot, IrankS, IrankL, nranks);


    const unsigned int  K      = int(cva.size()) + 1;
    const unsigned int  km1    = K - 1;
    const int           NT     = opt.phenotypeFiles.size();

    std::vector<int>    markerI;
    VectorXd            muk(K);           // mean of k-th component marker effect size
    MatrixXd            muk8(NT, K);
    VectorXd            denom(K-1);       // temporal variable for computing the inflation of the effect variance for a given non-zero component
    MatrixXd            denom8(NT, K-1);
    VectorXd            cVa(K);           // component-specific variance
    VectorXd            cVaI(K);          // inverse of the component variances
    double              num;              // storing dot product
    int                 m0;               // total number of markers in model
    MatrixXi            sum_cass8(NT, K); // To store the sum of the cass elements over all ranks
    VectorXd            Acum(M);
    MatrixXd            Acum8(NT, M);
    VectorXd            Gamma(data.numFixedEffects);
    //daniel The following variables are related to the restarting
    typedef Matrix<bool, Dynamic, 1> VectorXb;
    VectorXb            adaV(M);   //daniel adaptative scan Vector, ones will be sampled, 0 will be set to 0

    double* Beta8 = (double*)_mm_malloc(M * sizeof(double) * NT, 64);  check_malloc(Beta8, __LINE__, __FILE__);

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
    string outfp = opt.mcmcOut + ".csv";
    string betfp = opt.mcmcOut + ".bet";
    string cpnfp = opt.mcmcOut + ".cpn";
    string acufp = opt.mcmcOut + ".acu";
    string rngfp = opt.mcmcOut + ".rng." + std::to_string(rank);
    string mrkfp = opt.mcmcOut + ".mrk." + std::to_string(rank);
    string epsfp = opt.mcmcOut + ".eps." + std::to_string(rank);
    string covfp = opt.mcmcOut + ".cov." + std::to_string(rank);
    

    cass8.resize(NT, K);   cass8.setZero();
    priorPi.resize(K);     priorPi.setZero();
    pi.resize(K);          pi.setZero();
    pi8.resize(NT, K);     pi8.setZero();

    Acum8.setZero();
    gamma.setZero();

    X = data.X; //fixed effects matrix

    priorPi[0] = 0.5;
    cVa[0]     = 0.0;
    cVaI[0]    = 0.0;

    muk8.setZero();
    cVa.segment(1,km1)     = cva;
    cVaI.segment(1,km1)    = cVa.segment(1,km1).cwiseInverse();
    priorPi.segment(1,km1) = priorPi[0] * cVa.segment(1,km1).array() / cVa.segment(1,km1).sum();

    for (int i=0; i<NT; ++i) {
        pi8.row(i) = priorPi;
        mu8[i]     = 0.0;
        sigmaE8[i] = 0.0;
    }

#ifdef __INTEL_COMPILER
    __assume_aligned(Beta8, 64);
#endif
    for (int i=0; i<M*NT; i++)
        Beta8[i] = 0.0;

    components8.resize(NT, M);  components8.setZero();

    //EO TODO
    //epsilon_restart.resize(Ntot - data.numNAs);
    //epsilon_restart.setZero();

    markerI_restart.resize(M);
    std::fill(markerI_restart.begin(), markerI_restart.end(), 0);


    // In case of a restart, we first read the latest dumps
    // ----------------------------------------------------
    if (opt.restart) {

        cout << "ADJUST RESTART IN MT" << endl;
        exit(1);

        init_from_restart(K, M, Mtot, Ntot, MrankS, MrankL, use_xfiles_in_restart);

        if (rank == 0)
            data.print_restart_banner(opt.mcmcOut.c_str(),  iteration_restart, iteration_start);

        // Load dumped PRNG state
        //printf("RESTART: rank %3d reading PRNG state from file %s!\n", rank, rngfp.c_str());
        dist.read_rng_state_from_file(rngfp);

        // Rename output files so that we do not erase from failed job!
        //if (rank == 0)  printf("RESTART: renaming output files to %s\n", (opt.mcmcOut+"_rs").c_str());
        opt.mcmcOut = opt.mcmcOut + "_rs";
        outfp = opt.mcmcOut + ".csv";
        betfp = opt.mcmcOut + ".bet";
        cpnfp = opt.mcmcOut + ".cpn";
        acufp = opt.mcmcOut + ".acu";
        rngfp = opt.mcmcOut + ".rng." + std::to_string(rank);
        epsfp = opt.mcmcOut + ".eps." + std::to_string(rank);
        covfp = opt.mcmcOut + ".cov." + std::to_string(rank);
        
        //exit(0);
    } else {        
        init_from_scratch();
        
        // Init tasks' prng
        for (int i=0; i<NT; i++) {
            dist8[i].reset_rng((uint)(opt.seed + rank*1000));
            sigmaG8[i] = dist8[i].beta_rng(1.0, 1.0);
        }
    }


    // Delete old files (fp appended with "_rs" in case of restart, so that
    // original files are kept untouched) and create new ones
    // --------------------------------------------------------------------
    if (rank == 0) {
        MPI_File_delete(outfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(betfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(cpnfp.c_str(), MPI_INFO_NULL);
        MPI_File_delete(acufp.c_str(), MPI_INFO_NULL);
    }
    MPI_File_delete(epsfp.c_str(), MPI_INFO_NULL);
    MPI_File_delete(mrkfp.c_str(), MPI_INFO_NULL);
    MPI_Barrier(MPI_COMM_WORLD);

    check_mpi(MPI_File_open(MPI_COMM_WORLD, outfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &outfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, betfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &betfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, cpnfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &cpnfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, acufp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &acufh), __LINE__, __FILE__);

    check_mpi(MPI_File_open(MPI_COMM_SELF,  epsfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &epsfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_SELF,  mrkfp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &mrkfh), __LINE__, __FILE__);


    // First element of the .bet, .cpn and .acu files is the
    // total number of processed markers
    // -----------------------------------------------------
    betoff = size_t(0);
    if (rank == 0) {
        check_mpi(MPI_File_write_at(betfh, betoff, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(cpnfh, betoff, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
        check_mpi(MPI_File_write_at(acufh, betoff, &Mtot, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();

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


    // Correct each marker for individuals with missing phenotype
    // WARNING: for multi-phen we use masks
    // ----------------------------------------------------------
    if (data.numNAs > 0 && !opt.multi_phen) {
        MPI_Barrier(MPI_COMM_WORLD);
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
    if (rank == 0) printf("INFO   : start computing statistics on Ntot = %d individuals\n", Ntot);

    double dNtot     = (double) Ntot;
    double dNtotm1   = (double)(Ntot - 1);
    const size_t NDB = size_t(Ntot * NT) * sizeof(double);

    double *mave, *mstd;
    mave    = (double*) _mm_malloc(size_t(NT) * size_t(M) * sizeof(double), 64);  check_malloc(mave, __LINE__, __FILE__);
    mstd    = (double*) _mm_malloc(size_t(NT) * size_t(M) * sizeof(double), 64);  check_malloc(mstd, __LINE__, __FILE__);
    uint8_t *nanMask;
    nanMask = (uint8_t*)_mm_malloc(NDB, 64);  check_malloc(nanMask, __LINE__, __FILE__);

    // Fill the NAN masks
    if (opt.interleave) {
        for (int i=0; i<NT; i++) {
            for (int j=0; j<Ntot; j++) {
                nanMask[j * NT + i] = (uint8_t) data.phenosNanMasks(i, j);
            }
        }
    } else {
        for (int i=0; i<NT; i++) {
            for (int j=0; j<Ntot; j++) {
                nanMask[i * Ntot + j] = (uint8_t) data.phenosNanMasks(i, j);
            }
        }
    }
    

    //TODO: use nanMask with if statements
    for (int i=0; i<NT; i++) {
        
        for (int j=0; j<M; j++) {
            
            int n1nona = 0, n2nona = 0, nmnona = 0;
            mave[i*M + j] = 0.0;
            
            for (size_t k=N1S[j]; k<N1S[j]+N1L[j]; k++) {
                mave[i*M + j] += data.phenosNanMasks(i, I1[k]) * 1.0;
            }
            n1nona = int(mave[i*M + j]);
            
            for (size_t k=N2S[j]; k<N2S[j]+N2L[j]; k++) {
                mave[i*M + j] += data.phenosNanMasks(i, I2[k]) * 2.0;
            }
            n2nona = int(mave[i*M + j]) - n1nona;
            assert(n2nona%2 == 0);
            n2nona /= 2;
            
            // Discard missing data on NAs
            int monnas = 0;
            for (size_t k=NMS[j]; k < NMS[j]+NML[j]; k++) {
                if (data.phenosNanMasks(i, IM[k]) == 0.0) {
                    monnas += 1;
                }
            }
            nmnona = NML[j] - monnas;
            //printf("EQUIVA N1L = %d, N2L = %d, NML = %d\n", n1nona, n2nona, nmnona);
            
            //printf("phen %d, marker %d has %d monnas\n", i, j, monnas);
            mave[i*M + j] /= (dNtot - data.phenosNanNum(i) - double(NML[j]- monnas));

            double const mean = mave[i*M + j];

            mstd[i*M + j] = 0.0;

            double tmp = 0.0;
            for (size_t k=N1S[j]; k<N1S[j]+N1L[j]; k++) {
                tmp += data.phenosNanMasks(i, I1[k]);
            }
            mstd[i*M + j] += tmp * (1.0 - mean) * (1.0 - mean);

            tmp = 0.0;
            for (size_t k=N2S[j]; k<N2S[j]+N2L[j]; k++) {
                tmp += data.phenosNanMasks(i, I2[k]);
            }
            mstd[i*M + j] += tmp * (2.0 - mean) * (2.0 - mean);

            // Compute number of zeros not on NA
            int N0L    = Ntot - N1L[j] - N2L[j] - NML[j];
            int naonze = data.phenosNanNum(i) - (N1L[j]-n1nona) - (N2L[j]-n2nona) - (NML[j]-nmnona);
            int n0nona = N0L - naonze;
            assert(n0nona+n1nona+n2nona+nmnona == Ntot - data.phenosNanNum(i));
            
            mstd[i*M + j] += n0nona * mean * mean; // equiv to (0.0 - mean) * (0.0 - mean);
            tmp =  mstd[i*M + j];
            mstd[i*M + j]  = sqrt(double(n1nona + n2nona + n0nona + nmnona - 1) /  mstd[i*M + j]);

            //if (i==0)
            //    printf("marker %6d mean %20.15f, std = %20.15f (%.1f / %.15f) on phen %d\n", j, mave[i*M + j], mstd[i*M + j], double(n1nona + n2nona + n0nona + nmnona - 1), tmp, i);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const auto et2 = std::chrono::high_resolution_clock::now();
    const auto dt2 = et2 - st2;
    const auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();
    if (rank == 0)   std::cout << "INFO   : time to preprocess the data: " << du2 / double(1000.0) << " seconds." << std::endl;

    
    //cout << "__EARLY_RETURN__" << endl;
    //return 0;


    // Build list of markers    
    // ---------------------
    for (int i=0; i<M; ++i) markerI.push_back(i);


    // Processing part
    // ---------------
    const auto st3 = std::chrono::high_resolution_clock::now();
 
    //double *y, *epsilon, *tmpEps, *previt_eps, *deltaEps, *dEpsSum, *deltaSum;
    double  *y, *epsilon, *tmpEps, *deltaEps, *dEpsSum, *deltaSum;
    y          = (double*) _mm_malloc(NDB, 64);  check_malloc(y,          __LINE__, __FILE__);
    epsilon    = (double*) _mm_malloc(NDB, 64);  check_malloc(epsilon,    __LINE__, __FILE__);
    tmpEps     = (double*) _mm_malloc(NDB, 64);  check_malloc(tmpEps,     __LINE__, __FILE__);
    //previt_eps = (double*)malloc(NDB);  check_malloc(previt_eps, __LINE__, __FILE__);
    deltaEps   = (double*) _mm_malloc(NDB, 64);  check_malloc(deltaEps,   __LINE__, __FILE__);
    dEpsSum    = (double*) _mm_malloc(NDB, 64);  check_malloc(dEpsSum,    __LINE__, __FILE__);
    deltaSum   = (double*) _mm_malloc(NDB, 64);  check_malloc(deltaSum,   __LINE__, __FILE__);

    //TODO: adjust this one
    dalloc += NDB * NT / 1E9;

    double totalloc = 0.0;
    MPI_Reduce(&dalloc, &totalloc, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("INFO   : overall allocation %.3f GB\n", totalloc);

    set_array(dEpsSum, 0.0, Ntot * NT);

    if (opt.covariates) {
    	gamma = VectorXd(data.X.cols()); 
    	gamma.setZero();
    }
    
    // In case of restart we reset epsilon to last dumped state (sigmaE as well, see init_from_restart)
    if (opt.restart) {
        cout << "ADAPT!!" << endl;
        exit(1);
        for (int i=0; i<Ntot; ++i)  epsilon[i] = epsilon_restart[i];
        markerI = markerI_restart;
    } else {
        // Copy centered and scaled phenotype observations
        if (opt.interleave) {
            for (int i=0; i<NT; i++) {
                for (int j=0; j<Ntot; j++) {
                    epsilon[j * NT + i] = data.phenosData(i, j);
                    nanMask[j * NT + i] = (uint8_t) data.phenosNanMasks(i, j);
                }
            }
            for (int i=0; i<NT; i++) {
                sigmaE8[i] = 0.0;
                for (int j=0; j<Ntot; j++) {
                    if (nanMask[j * NT + i] == 1) {
                        sigmaE8[i] += epsilon[j * NT + i] * epsilon[j * NT + i];// * nanMask[j * NT + i];
                    }
                }
                //printf("sigmaE8[%d] = %15.10f on = %.1f elements\n", i, sigmaE8[i], dNtot - double(data.phenosNanNum(i)));
                sigmaE8[i] = sigmaE8[i] / (dNtot - double(data.phenosNanNum(i))) * 0.5;
            }
        } else {
            cout << "STACK PHENOS -- NO INTERLEAVE" << endl;

            for (int i=0; i<NT; i++) {
                sigmaE8[i] = 0.0;
                for (int j=0; j<Ntot; j++) {
                    const int index = i * Ntot + j;
                    epsilon[index] = data.phenosData(i, j);
                    nanMask[index] = (uint8_t) data.phenosNanMasks(i, j);
                    if (nanMask[index] >= 1) {
                        sigmaE8[i] += epsilon[index] * epsilon[index];
                    }
                }
                sigmaE8[i] = sigmaE8[i] / (dNtot - double(data.phenosNanNum(i))) * 0.5;
            }
        }
    }
    //printf("sigmaE8 = [%15.13f, %15.13f], on = %.1f, %.1f elements\n", sigmaE8[0], sigmaE8[7], dNtot - double(data.phenosNanNum(0)), dNtot - double(data.phenosNanNum(7)));

    adaV.setOnes();

    if (opt.restart) {
        //if (rank == 0)  cout << "  !!!! RESTART ADAV!!!!  " << endl;
        //adaV = data.rAdaV;
    }

    //EO TODO: align all 64
    double p8[8]         = {0.0};
    double beta8[8]      = {0.0};
    double betaOld8[8]   = {0.0};
    double sigE_G8[8]    = {0.0};
    double sigG_E8[8]    = {0.0};
    double i_2sigE8[8]   = {0.0};
    double e_sqn8[8]     = {0.0};

    double* deltaBeta8 = (double*)_mm_malloc(size_t(NT) * sizeof(double), 64);  check_malloc(deltaBeta8, __LINE__, __FILE__);
    for (int i=0; i<NT; i++) 
        deltaBeta8[i] = 0.0;

    double   acum8[8]      = {0.0};
    size_t   markoff;
    int      marker, left;
    VectorXd logL(K);
    MatrixXd logL8(NT, K);
    std::vector<unsigned int> xI(data.X.cols());
    std::iota(xI.begin(), xI.end(), 0);
    sigmaF = s02F;

    // A counter on previously saved thinned iterations
    uint n_thinned_saved = 0;


    //double   previt_m0 = 0.0;
    //double   previt_sg = 0.0;
    //double   previt_mu = 0.0;
    //double   previt_se = 0.0;
    //VectorXd previt_Beta(M);


    // Main iteration loop
    // -------------------
    //bool replay_it = false;

    int count__sum_vectors_f64_v2 = 0;

    for (uint iteration=iteration_start; iteration<opt.chainLength; iteration++) {

        double start_it = MPI_Wtime();

        //if (replay_it) {
        //    printf("INFO: replay iteration with m0=%.0f sigG=%15.10f sigE=%15.10f\n", previt_m0, previt_sg, previt_se);
        //    m0        = previt_m0;
        //    sigmaG    = previt_sg;
        //    sigmaE    = previt_se;
        //    mu        = previt_mu;
        //    sync_rate = 0;
        //    for (int i=0; i<Ntot; ++i) epsilon[i] = previt_eps[i];
        //    Beta      = previt_Beta;
        //    replay_it = false;
        //}


        // Store status of iteration to revert back to it if required
        // ----------------------------------------------------------
        //previt_m0 = m0;
        //marion : this should probably be a vector with sigmaGG
        //previt_sg = sigmaG;
        //previt_se = sigmaE;
        //previt_mu = mu;
        //for (int i=0; i<Ntot; ++i) previt_eps[i]  = epsilon[i];
        //for (int i=0; i<M;    ++i) previt_Beta(i) = Beta(i);

        if (opt.interleave) {
            for (int i=0; i<NT; i++) {
                for (int j=0; j<Ntot; j++) {
                    epsilon[j*NT + i] += mu8[i];
                }
            }
        } else {
            for (int i=0; i<NT; i++) {
                for (int j=0; j<Ntot; j++) {
                    epsilon[i*Ntot + j] += mu8[i];
                }
            }
        }


        double* epssum8 = (double*)_mm_malloc(size_t(NT) * sizeof(double), 64);  check_malloc(epssum8, __LINE__, __FILE__);

        if (opt.interleave) {
            for (int i=0; i<NT; i++) {
                epssum8[i] = 0.0;
                for (int j=0; j<Ntot; j++) {
                    epssum8[i] += epsilon[j*NT + i] * data.phenosNanMasks(i, j);
                }
            }
        } else {
            for (int i=0; i<NT; i++) {
                epssum8[i] = 0.0;
                for (int j=0; j<Ntot; j++) {
                    epssum8[i] += epsilon[i*Ntot + j] * data.phenosNanMasks(i, j);
                }
            }
        }

        //printf("epssum8 = [%20.15f, %20.15f] with Ntot - NAs = [%d, %d] elements\n", epssum8[0], epssum8[7], Ntot-data.phenosNanNum(0),  Ntot-data.phenosNanNum(7));

        // update mu
        for (int i=0; i<NT; i++) {
            mu8[i] = dist8[i].norm_rng(epssum8[i] / (dNtot - double(data.phenosNanNum[i])), sigmaE8[i] / (dNtot - double(data.phenosNanNum[i])));
        }
        
        _mm_free(epssum8);

        //printf("it %d, rank %d: mu8[0] = %15.10f with dN = %10.1f\n", iteration, rank, mu8[0], dNtot - double(data.phenosNanNum[0]));

        // We substract again now epsilon =Y-mu-X*beta
        for (int i=0; i<NT; i++) {
            for (int j=0; j<Ntot; j++) {
                epsilon[j*NT + i] -= mu8[i];
            }
        }

        //EO: watch out, std::shuffle is not portable, so do no expect identical
        //    results between Intel and GCC when shuffling the markers is on!!
        //------------------------------------------------------------------------
        if (opt.shuffleMarkers) {
            std::shuffle(markerI.begin(), markerI.end(), dist.rng);
            //std::random_shuffle(markerI.begin(), markerI.end());

            //EO: over shuffling only for maintaining PRNG state the same accross all phenotypes
            for (int i=0; i<NT; i++)
                std::shuffle(markerI.begin(), markerI.end(), dist8[i].rng);
        }

        m0 = 0;
        cass8.setZero();

        for (int i=0; i<NT; i++) {
            sigE_G8[i]  = sigmaE8[i] / sigmaG8[i];
            sigG_E8[i]  = sigmaG8[i] / sigmaE8[i];
            i_2sigE8[i] = 1.0 / (2.0 * sigmaE8[i]);
            //printf("it %d rank %d: sigE_G8 = %15.10f, %15.10f, %15.10f\n", iteration, rank, sigE_G8[i], sigG_E8[i], i_2sigE8[i]);
        }

#ifdef __INTEL_COMPILER
        __assume_aligned(tmpEps,  64);
        __assume_aligned(epsilon, 64);
#endif
        for (int i=0; i<Ntot*NT; ++i) {
            tmpEps[i] = epsilon[i];
        }

        double cumSumDeltaBetas      = 0.0;
        double cumSumDeltaBetas8[8] = {0.0};

        int    sinceLastSync    = 0;

        // Loop over (shuffled) markers
        // ----------------------------
        for (int j = 0; j < lmax; j++) {

            sinceLastSync += 1;

            if (j < M) {

                marker  = markerI[j];
                //cout << "marker = " << marker << endl;

                for (int i=0; i<NT; i++)
                    beta8[i] = Beta8[marker + i * M];
                
                if (adaV[j]) {

                    //we compute the denominator in the variance expression to save computations
                    for (int t=0; t<NT; ++t) {
                        for (int i=1; i<=km1; ++i) {
                            //EO: CHECK THIS ONE!!!!!!!!!!!!!
                            denom8(t, i-1) = (dNtotm1 - data.phenosNanNum(t)) + sigE_G8[t] * cVaI(i);
                            //printf("it %d, rank %d, m %d: denom[%d] = %20.15f\n", iteration, rank, marker, i-1, denom(i-1));
                        }
                    }
                    //cout << "denom8 = " << endl << denom8 << endl;

                    double* num8 = (double*)_mm_malloc(NT * sizeof(double), 64);
                    sparse_dotprod_mt(epsilon, nanMask,
                                      I1, N1S[marker], N1L[marker],
                                      I2, N2S[marker], N2L[marker],
                                      IM, NMS[marker], NML[marker],
                                      mave[marker],    mstd[marker],
                                      Ntot,            marker,
                                      num8,            NT,           opt.interleave);
                    
                    //for (int i=0; i<NT; ++i)  printf("num8[%d] = %15.10f\n", i, num8[i]);
                    
                    //PROFILE
                    //continue;

                    for (int i=0; i<NT; ++i) {
                        num8[i] += beta8[i] * double(Ntot - data.phenosNanNum[i] - 1);
                        //printf("It %d, mr %4d: num8[%d] = %20.15f\n", iteration, marker, i, num8[i]);
                    }

                    //printf("it %d, rank %d, mark %d: num = %20.15f, %20.15f, %20.15f\n", iteration, rank, marker, num, mave[marker], mstd[marker]);
                    
                    //muk for the other components is computed according to equations
                    for (int i=0; i<NT; i++) {
                        muk8.row(i).segment(1, km1) = num8[i] / denom8.row(i).array();
                    }
                    //cout << "muk8 = " << endl << muk8 << endl; 

                    //first component probabilities remain unchanged
                    //cout << "pi8=" << endl << pi8 << endl;

                    for (int i=0; i<NT; ++i)
                        logL8.row(i) = pi8.row(i).array().log();
                    //cout << "logL8 = " << logL8 << endl;

                    // Update the log likelihood for each component
                    for (int i=0; i<NT; i++) {
                        for (int ii=1; ii<=km1; ii++) {
                            logL8(i,ii) = logL8(i,ii) - 0.5d * log(sigG_E8[i] * (dNtotm1 - double(data.phenosNanNum(i))) * cVa(ii) + 1.0d) + muk8(i,ii) * num8[i] * i_2sigE8[i];
                        }
                    }                    
                    //cout << "logL8 =" << endl << logL8 << endl;
                    
                    _mm_free(num8);

                    for (int i=0; i<NT; i++) {
                        p8[i] = dist8[i].unif_rng();
                    }                    
                    //printf("%d/%d/%d  p = %15.10f\n", iteration, rank, j, p);
                    
                    for (int i=0; i<NT; i++)
                        acum8[i] = 0.0;

                    for (int i=0; i<NT; i++) {
                        bool set_to_0 = false;
                        for (int ii=1; ii<=km1; ii++) {
                            if (fabs(logL8(i, ii) - logL8(i, 0)) > 700) {
                                set_to_0 = true;
                                break;
                            }
                        }
                        if (!set_to_0) {
                            double sum_exp = 0.0;
                            for (int ii=0; ii<=km1; ii++)
                                sum_exp += exp(logL8(i, ii) - logL8(i, 0));
                            acum8[i] = 1.0 / sum_exp;
                        }
                    }

                    // Store marker acum for later dump to file
                    for (int i=0; i<NT; i++)
                        Acum8(i, marker) = acum8[i];
                    
                    //EO: handle multiple phenotypes at once
                    for (int i=0; i<NT; i++) {

                        for (int k=0; k<K; k++) {

                            if (p8[i] <= acum8[i] || k == km1) { //if we p is less than acum or if we are already in the last mixt.
                                if (k==0) {
                                    Beta8[marker + i * M] = 0.0;
                                } else {
                                    Beta8[marker + i * M] = dist8[i].norm_rng(muk8(i,k), sigmaE8[i] / denom8(i, k-1));
                                    //if (i == 0)
                                    //    printf("@B@ bet8 update %4d/%4d/%4d muk8[%4d,%4d] = %15.10f with p=%15.10f <= acum=%15.10f, denom = %15.10f, sigmaE = %15.10f: beta = %15.10f\n", iteration, rank, marker, i, k, muk8(i, k), p8[i], acum8[i], denom8(i, k-1), sigmaE8[i], Beta8[marker + i * M]);
                                }                                
                                cass8(i, k)            += 1;
                                components8(i, marker)  = k;
                                break;
                            } else {
                                //if too big or too small
                                if (k+1 >= K) {
                                    printf("FATAL  : iteration %d, marker = %d, p = %15.10f, acum = %15.10f logL overflow with %d => %d\n", iteration, marker, p8[i], acum8[i], k+1, K);
                                    MPI_Abort(MPI_COMM_WORLD, 1);
                                }
                                if (((logL8.row(i).segment(k+1, K-(k+1)).array() - logL8(i, k+1)).abs().array() > 700.0d ).any()) {
                                    acum8[i] += 0.0d; // we compare next mixture to the others, if to big diff we skip
                                } else{
                                    acum8[i] += 1.0d / ((logL8.row(i).array()-logL8(i, k+1)).exp().sum()); //if not , sample
                                }
                            }
                        }
                    }

                } else { // end of adapative if daniel
                    for (int i=0; i<NT; i++) {
                        Beta8[marker + i * M] = 0.0;
                        Acum8(i, marker)      = 1.0;
                    }
                }

                //for (int i=0; i<NT; i++) 
                //    printf("acum8[%d] = %15.10f, p8[%d] = %15.10f\n", i, acum8[i], i, p8[i]);

                bool trigger_eps8_update = false;
                for (int i=0; i<NT; i++) {
                    betaOld8[i]   = beta8[i];
                    beta8[i]      = Beta8[marker + i*M];
                    deltaBeta8[i] = betaOld8[i] - beta8[i];
                    if (deltaBeta8[i] != 0.0) trigger_eps8_update = true;
                } 
                //printf("deltaBeta8[0] = %15.10f\n", deltaBeta8[0]);

                if (trigger_eps8_update) {
                    //cout << "EPS 8 UPDATE!!" << endl;
                    sparse_scaadd_mt(deltaEps, deltaBeta8,
                                     I1, N1S[marker], N1L[marker], 
                                     I2, N2S[marker], N2L[marker], 
                                     IM, NMS[marker], NML[marker], 
                                     mave[marker], mstd[marker], Ntot, NT, opt.interleave);

                    add_arrays(dEpsSum, deltaEps, Ntot * NT);
                }
            }
            // Make the contribution of tasks beyond their last marker nill
            // ------------------------------------------------------------
            else {
                for (int i=0; i<NT; i++)
                    deltaBeta8[i] = 0.0;
                
                set_array(deltaEps, 0.0, Ntot * NT);
            }

            // Check whether we have a non-zero beta somewhere
            // sum of the abs values !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            if (nranks > 1) { 
                double sumDeltaBetas8[8] = {0.0};
                double deltaBeta8_abs[8];
                for (int i=0; i<NT; i++)
                    deltaBeta8_abs[i] = fabs(deltaBeta8[i]);
                check_mpi(MPI_Allreduce(&deltaBeta8_abs, &sumDeltaBetas8, 8, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
                for (int i=0; i<NT; i++)
                    cumSumDeltaBetas8[i] += sumDeltaBetas8[i];
                //printf("cumSumDeltaBetas [%15.10f, %15.10f]\n", cumSumDeltaBetas8[0],  cumSumDeltaBetas8[7]);

            } else {
                for (int i=0; i<NT; i++) 
                    cumSumDeltaBetas8[i] += fabs(deltaBeta8[i]); //EO: check
            } 
            //printf("%d/%d/%d: deltaBeta = %20.15f = %10.7f - %10.7f; sumDeltaBetas = %15.10f\n", iteration, rank, marker, deltaBeta8[0], betaOld8[0], beta8[0], cumSumDeltaBetas8[0]);
            
            bool trigger_update = false;
            for (int i=0; i<NT; i++) {
                if (cumSumDeltaBetas8[i] != 0.0) {
                    trigger_update = true;
                    //cout << "TRIGGER_UPDATED!" << endl;
                }
            }

            //if ( (sync_rate == 0 || sinceLastSync > sync_rate || j == M-1) && cumSumDeltaBetas != 0.0) {
            //if ( (sync_rate == 0 || sinceLastSync > sync_rate || j == lmax-1) && trigger_update) {
            if ( (sinceLastSync >= opt.syncRate || j == lmax-1) && trigger_update) {

                // Update local copy of epsilon
                if (nranks > 1) {
                    check_mpi(MPI_Allreduce(&dEpsSum[0], &deltaSum[0], Ntot*NT, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
                    add_arrays(epsilon, tmpEps, deltaSum, Ntot * NT);
                } else {
                    count__sum_vectors_f64_v2 += 1;
                    //double t1 = -mysecond();
                    //for (int ii=0; ii<100; ii++) {
                        add_arrays(epsilon, tmpEps, dEpsSum,  Ntot * NT);
                        //}
                        //t1 += mysecond();
                        //printf("kernel 1 BW MT = %g\n", double(NT) * double(Ntot) * 3.0 * sizeof(double) / 1024. / 1024. / (t1 / 100.0)); 
                }

                // Store epsilon state at last synchronization
                copy_array(tmpEps, epsilon, Ntot * NT);

                // Reset local sum of delta epsilon
                set_array(dEpsSum, 0.0, Ntot * NT);
                
                // Reset cumulated sum of delta betas
                for (int i=0; i<NT; i++) 
                    cumSumDeltaBetas8[i] = 0.0;

                sinceLastSync = 0;

            }
            //else {
            //
            //    sinceLastSync += 1;
            //}

        } // END PROCESSING OF ALL MARKERS

        //PROFILE
        //continue;


        double beta_squaredNorm8[8] = {0.0};
#ifdef __INTEL_COMPILER
        __assume_aligned(Beta8, 64);
#endif
        for (int i=0; i<NT; i++) {
            for (int ii=0; ii<M; ii++) {
                beta_squaredNorm8[i] += Beta8[i*M + ii] * Beta8[i*M + ii];
            }
        }
        printf("beta_squaredNorm8 = [%15.10f, %15.10f]\n", beta_squaredNorm8[0], beta_squaredNorm8[1]);


        // Transfer global to local
        // ------------------------
        if (nranks > 1) {
            double sum_beta_squaredNorm8[8] = {0.0};
            check_mpi(MPI_Allreduce(&beta_squaredNorm8, &sum_beta_squaredNorm8, NT,            MPI_DOUBLE,  MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            check_mpi(MPI_Allreduce(cass8.data(),        sum_cass8.data(),      cass8.size(),  MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
            cass8  = sum_cass8;
            for (int i=0; i<NT; i++)
                beta_squaredNorm8[i] = sum_beta_squaredNorm8[i];
        }

        cout << "OK1" << endl;

        // Update global parameters
        // ------------------------
        m0 = Mtot - cass8(0, 0);
        for (int i=0; i<NT; i++)
            sigmaG8[i] = dist8[i].inv_scaled_chisq_rng(v0G + double(m0), (beta_squaredNorm8[i] * double(m0) + v0G * s02G) /(v0G + double(m0)));
        //printf("sigmaG8 = [%15.10f, %15.10f]\n", sigmaG8[0], sigmaG8[7]);



        // Check iteration
        // 
        // ---------------
        /*
        if (iteration >= 0) {
            
            double max_sg = 0.0;
            MPI_Allreduce(&sigmaG, &max_sg, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
            if (rank == 0) printf("INFO   : max sigmaG of iteration %d is %15.10f with sync rate of %d\n", iteration, max_sg, sync_rate);

            if (max_sg > 1.0) {
                if (sync_rate == 0) {
                    if (rank == 0) {
                        printf("CRITICAL: detected task with sigmaG = %15.10f and sync rate of %d\n", max_sg, sync_rate); 
                        printf("          => desperate situation, aborting...\n");
                    }
                    MPI_Abort(MPI_COMM_WORLD, 1);
                } else {
                    if (rank == 0)
                        printf("          => will do an attempt with setting sync_rate to 0\n");
                    replay_it = true;
                    continue;
                }
            }

            //if ( m0 > 1.2 * previt_m0 || m0 < 0.8 * previt_m0) {
            //    printf("CRITICAL: divergence detected! Will cancel iteration and set up a lower sync rate\n");
            //}
        }
        */

        // For the fixed effects
        // ---------------------
        if (opt.covariates) {

            if (rank == 0) {

                std::shuffle(xI.begin(), xI.end(), dist.rng);
                double gamma_old, num_f, denom_f;
                double sigE_sigF = sigmaE / sigmaF;
                //cout << "data.X.cols " << data.X.cols() << endl;
                for (int i=0; i<data.X.cols(); i++) {
                    gamma_old = gamma(xI[i]);
                    num_f     = 0.0;
                    denom_f   = 0.0;
                    
                    for (int k=0; k<Ntot; k++)
                        num_f += data.X(k, xI[i]) * (epsilon[k] + gamma_old * data.X(k, xI[i]));
                    denom_f = dNtotm1 + sigE_sigF;
                    gamma(i) = dist.norm_rng(num_f/denom_f, sigmaE/denom_f);
                    
                    for (int k = 0; k<Ntot ; k++) {
                        epsilon[k] = epsilon[k] + (gamma_old - gamma(xI[i])) * data.X(k, xI[i]);
                        //cout << "adding " << (gamma_old - gamma(xI[i])) * data.X(k, xI[i]) << endl;
                    }
                }
                //the next line should be uncommented if we want to use ridge for the other covariates.
                //sigmaF = inv_scaled_chisq_rng(0.001 + F, (gamma.squaredNorm() + 0.001)/(0.001+F));
                sigmaF = s02F;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        if (opt.interleave) {
            for (int i=0; i<NT; i++) {
                e_sqn8[i] = 0.0;
                for (int j=0; j<Ntot; j++) {
                    e_sqn8[i] += epsilon[j*NT + i] * epsilon[j*NT + i] * data.phenosNanMasks(i, j);
                }
            }
        } else {
            for (int i=0; i<NT; i++) {
                e_sqn8[i] = 0.0;
                for (int j=0; j<Ntot; j++) {
                    const int index = i * Ntot + j;
                    if (nanMask[index] >= 1) {
                        e_sqn8[i] += epsilon[index] * epsilon[index];
                    }
                }
            }
        }

        //printf("e_sqn8 = [%15.10f, %15.10f]\n", e_sqn8[0], e_sqn8[7]);

        for (int i=0; i<NT; i++)
            sigmaE8[i] = dist8[i].inv_scaled_chisq_rng(v0E+dNtot-double(data.phenosNanNum(i)), (e_sqn8[i] + v0E*s02E) / (v0E + dNtot - double(data.phenosNanNum(i))));
        //printf("sigmaE8 = [%15.10f, %15.10f]\n", sigmaE8[0], sigmaE8[7]);

        //printf("%d epssqn = %15.10f %15.10f %15.10f %6d => %15.10f\n", iteration, e_sqn, v0E, s02E, Ntot, sigmaE);
        if (rank%10==0) {
            for (int i=0; i<2; i++) {
                printf("RESULT : it %4d, rank %4d, pheno %d: sigmaG(%15.10f, %15.10f) = %15.10f, sigmaE = %15.10f, betasq = %15.10f, m0 = %d\n", iteration, rank, i, v0G + double(m0), (beta_squaredNorm8[i] * double(m0) + v0G * s02G) /(v0G + double(m0)), sigmaG8[i], sigmaE8[i], beta_squaredNorm8[i], m0);
            }
            fflush(stdout);
        }

        for (int i=0; i<NT; i++)
            pi8.row(i) =  dist8[i].dirichlet_rng(cass8.row(i).array() + 1);
        //cout << "pi8= " << pi8 << endl;


        //EO DISABLE FOR NOW (remove 1 == 2)
        
        // Write output files
        // ------------------
        if (iteration%opt.thin == 0 && 1 == 2) {
            
            left = snprintf(buff, LENBUF, "%5d, %4d, %20.15f, %20.15f, %20.15f, %20.15f, %7d, %2d",
                            iteration, rank, mu, sigmaG, sigmaE, sigmaG/(sigmaE+sigmaG), m0, int(pi.size()));
            assert(left > 0);

            for (int ii=0; ii<pi.size(); ++ii) {
                left = snprintf(&buff[strlen(buff)], LENBUF-strlen(buff), ", %20.15f", pi(ii));
                assert(left > 0);
            }
            left = snprintf(&buff[strlen(buff)], LENBUF-strlen(buff), "\n");
            assert(left > 0);

            offset = (size_t(n_thinned_saved) * size_t(nranks) + size_t(rank)) * strlen(buff);
            check_mpi(MPI_File_write_at_all(outfh, offset, &buff, strlen(buff), MPI_CHAR, &status), __LINE__, __FILE__);

            // Write iteration number
            if (rank == 0) {
                betoff = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double));
                check_mpi(MPI_File_write_at(betfh, betoff, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
                check_mpi(MPI_File_write_at(acufh, betoff, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
                cpnoff = sizeof(uint) + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int));
                check_mpi(MPI_File_write_at(cpnfh, cpnoff, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            }
            
            betoff = sizeof(uint) + sizeof(uint) 
                + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(double))
                + size_t(MrankS[rank]) * sizeof(double);
            check_mpi(MPI_File_write_at_all(betfh, betoff, Beta.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at_all(acufh, betoff, Acum.data(), M, MPI_DOUBLE, &status), __LINE__, __FILE__);

            cpnoff = sizeof(uint) + sizeof(uint)
                + size_t(n_thinned_saved) * (sizeof(uint) + size_t(Mtot) * sizeof(int))
                + size_t(MrankS[rank]) * sizeof(int);
            check_mpi(MPI_File_write_at_all(cpnfh, cpnoff, components.data(), M, MPI_INTEGER, &status), __LINE__, __FILE__);

            //if (iteration == 0) {
            //    printf("rank %d dumping bet: %15.10f %15.10f\n", rank, Beta[0], Beta[MrankL[rank]-1]);
            //    printf("rank %d dumping cpn: %d %d\n", rank, components[0], components[MrankL[rank]-1]);
            //}

            n_thinned_saved += 1;

            //EO: to remove once MPI version fully validated; use the check_marker utility to retrieve
            //    the corresponding values from .bet file
            //    Print a sub-set of non-zero betas, one per rank for validation of the .bet file
            for (int i=0; i<M; ++i) {
                if (Beta(i) != 0.0) {
                    //printf("%4d/%4d global Beta[%8d] = %15.10f, components[%8d] = %2d\n", iteration, rank, MrankS[rank]+i, Beta(i), MrankS[rank]+i, components(i));
                    break;
                }
            }
        }

        // Dump the epsilon vector and the marker indexing one
        // Note: single line overwritten at each saving iteration
        // .eps format: uint, uint, double[0, N-1] (it, Ntot, [eps])
        // .mrk format: uint, uint, int[0, M-1]    (it, M,    <mrk>)
        // ------------------------------------------------------
        if (iteration%opt.save == 0 && 1 == 2) {

            cout << "SAVE" << endl;

            // Each task writes its own rng file
            dist.write_rng_state_to_file(rngfp);

            epsoff  = size_t(0);
            check_mpi(MPI_File_write_at(epsfh, epsoff, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(mrkfh, epsoff, &iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

            epsoff += sizeof(uint);
            check_mpi(MPI_File_write_at(epsfh, epsoff, &Ntot,      1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(mrkfh, epsoff, &M,         1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

            epsoff = sizeof(uint) + sizeof(uint);
            check_mpi(MPI_File_write_at(epsfh, epsoff, epsilon,        Ntot,           MPI_DOUBLE, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_write_at(mrkfh, epsoff, markerI.data(), markerI.size(), MPI_INT,    &status), __LINE__, __FILE__);
            //if (iteration == 0) {
            //    printf("rank %d dumping eps: %15.10f %15.10f\n", rank, epsilon[0], epsilon[Ntot-1]);
            //}
            //EO: to remove once MPI version fully validated; use the check_epsilon utility to retrieve
            //    the corresponding values from the .eps file
            //    Print only first and last value handled by each task
            //printf("%4d/%4d epsilon[%5d] = %15.10f, epsilon[%5d] = %15.10f\n", iteration, rank, IrankS[rank], epsilon[IrankS[rank]], IrankS[rank]+IrankL[rank]-1, epsilon[IrankS[rank]+IrankL[rank]-1]);
        }

        double end_it = MPI_Wtime();
        if (rank == 0) printf("TIME_IT: Iteration %5d on rank %4d took %10.3f seconds\n", iteration, rank, end_it-start_it);

        MPI_Barrier(MPI_COMM_WORLD);
    }


    printf(">.CHECK.< int count__sum_vectors_f64_v2 = %d\n", count__sum_vectors_f64_v2);


    // Close output files
    check_mpi(MPI_File_close(&outfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&betfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&epsfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&cpnfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&acufh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&mrkfh), __LINE__, __FILE__);


    // Release memory
    _mm_free(y);
    _mm_free(epsilon);
    _mm_free(nanMask);
    _mm_free(tmpEps);
    //free(previt_eps);
    _mm_free(deltaEps);
    _mm_free(dEpsSum);
    _mm_free(deltaSum);
    _mm_free(Beta8);
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
        printf("INFO   : rank %4d, time to process the data: %.3f sec.\n", rank, du3 / double(1000.0));

    return 0;
}

