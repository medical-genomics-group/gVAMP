#include "data.hpp"
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <iterator>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <mm_malloc.h>
#include "utils.hpp"
#include "gadgets.hpp"

using namespace Eigen;
using namespace std;

Data::Data() {}



uint Data::set_Ntot(const int rank, const Options opt) {

    uint Ntot = opt.numberIndividuals;

    if (Ntot == 0) {
        printf("FATAL  : opt.numberIndividuals is zero! Set it via --number-individuals in call.");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (Ntot != numInds - numNAs) {
        if (rank == 0)
            printf("WARNING: opt.numberIndividuals set to %d but will be adjusted to %d - %d = %d due to NAs in phenotype file.\n", Ntot, numInds, numNAs, numInds - numNAs);
    }

    return Ntot;
}


uint Data::set_Mtot(const int rank, Options opt) {

    uint Mtot = opt.numberMarkers;

    if (Mtot == 0) throw("FATAL  : opt.numberMarkers is zero! Set it via --number-markers in call.");

    // Block marker definition has precedence over requested number of markers
    if (opt.markerBlocksFile != "" && opt.numberMarkers > 0) {
        opt.numberMarkers = 0;
        if (rank == 0)
            printf("WARNING: --number-markers option ignored, a marker block definition file was passed!\n");
    }

    if (opt.numberMarkers > 0 && opt.numberMarkers < Mtot) {
        Mtot = opt.numberMarkers;
        if (rank == 0)
            printf("INFO   : Option passed to process only %d markers!\n", Mtot);
    }

    return Mtot;
}


void Data::print_restart_banner(const string mcmcOut, const uint iteration_restart, 
                                const uint iteration_start) {
    printf("INFO   : %s\n", string(100, '*').c_str());
    printf("INFO   : RESTART DETECTED\n");
    printf("INFO   : restarting from: %s.* files\n", mcmcOut.c_str());
    printf("INFO   : last saved iteration:        %d\n", iteration_restart);
    printf("INFO   : will restart from iteration: %d\n", iteration_start);
    printf("INFO   : %s\n", string(100, '*').c_str());
}


void Data::read_mcmc_output_idx_file(const string      mcmcOut,
                                     const string      ext,
                                     const uint        length,
                                     const uint        iteration_to_restart_from,
                                     const string      bayesType,
                                     std::vector<int>& markerI)  {
    
    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status status;
    MPI_File   fh;

    // No rank dependency for bayesW
    string fp = mcmcOut + "." + ext + "." + std::to_string(rank);
    check_mpi(MPI_File_open(MPI_COMM_SELF, fp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), __LINE__, __FILE__);

    // 1. get and validate iteration number that we are about to read
    MPI_Offset off = size_t(0);
    uint iteration_ = UINT_MAX;
    check_mpi(MPI_File_read_at(fh, off, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_to_restart_from) {
        printf("Mismatch between expected and read mrk iteration: %d vs %d\n", iteration_to_restart_from, iteration_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. get and validate M (against size of markerI)
    uint M_ = 0;
    off = sizeof(uint);
    check_mpi(MPI_File_read_at(fh, off, &M_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    uint M = markerI.size();
    if (M_ != M) {
        printf("Mismatch between expected and read mrk M: %d vs %d\n", M, M_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 3. read the M_ coefficients
    off = sizeof(uint) + sizeof(uint);
    check_mpi(MPI_File_read_at(fh, off, markerI.data(), M_, MPI_INT, &status), __LINE__, __FILE__);


    check_mpi(MPI_File_close(&fh), __LINE__, __FILE__);
}


//EO: .gam files only contain a dump of last saved iteration (no history)
//  : format is: iter, length, vector
void Data::read_mcmc_output_gam_file(const string mcmcOut, const int gamma_length, const uint iteration_to_restart_from,
                                     VectorXd& gamma) {

    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    const string gamfp = mcmcOut + ".gam." + std::to_string(rank);

    MPI_Status status;
    
    MPI_File gamfh;
    check_mpi(MPI_File_open(MPI_COMM_SELF, gamfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &gamfh), __LINE__, __FILE__);


    // 1. get and validate iteration number that we are about to read
    MPI_Offset gamoff = size_t(0);
    uint iteration_ = UINT_MAX;
    check_mpi(MPI_File_read_at_all(gamfh, gamoff, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_to_restart_from) {
        printf("Mismatch between expected and read gamma iteration: %d vs %d\n", iteration_to_restart_from, iteration_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. get and validate gamma_length
    uint gamma_length_ = 0;
    gamoff = sizeof(uint);
    check_mpi(MPI_File_read_at_all(gamfh, gamoff, &gamma_length_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (gamma_length_ != gamma_length) {
        printf("Mismatch between expected and read length of gamma vector: %d vs %d\n", gamma_length, gamma_length_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 3. read the gamma_length_ coefficients
    gamoff = sizeof(uint) + sizeof(uint);
    check_mpi(MPI_File_read_at_all(gamfh, gamoff, gamma.data(), gamma_length_, MPI_DOUBLE, &status), __LINE__, __FILE__);

    //if (rank%50==0)
    //    printf("rank %d reading back gam: %15.10f %15.10f\n", rank, gamma[0], gamma[gamma_length_-1]);

    check_mpi(MPI_File_close(&gamfh), __LINE__, __FILE__);
}


//EO: .eps files only contain a dump of last saved iteration (no history)
//
void Data::read_mcmc_output_eps_file(const string mcmcOut,
                                     const uint   Ntotc,
                                     const uint   iteration_to_restart_from,
                                     VectorXd&    epsilon) {

    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    const string epsfp = mcmcOut + ".eps." + std::to_string(rank);
    //cout << "epsfp = " << epsfp << endl;

    MPI_Status status;
    
    MPI_File epsfh;
    check_mpi(MPI_File_open(MPI_COMM_SELF, epsfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &epsfh), __LINE__, __FILE__);


    // 1. get and validate iteration number that we are about to read
    MPI_Offset epsoff = size_t(0);
    uint iteration_   = UINT_MAX;
    check_mpi(MPI_File_read_at_all(epsfh, epsoff, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_to_restart_from) {
        printf("Mismatch between expected and read eps iteration: %d vs %d\n", iteration_to_restart_from, iteration_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. get and validate Ntot_ (against size of epsilon, adjusted for NAs)
    uint Ntot_ = 0;
    epsoff = sizeof(uint);
    check_mpi(MPI_File_read_at_all(epsfh, epsoff, &Ntot_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    uint Ntot = epsilon.size();
    assert(Ntot == Ntotc);
    if (Ntot_ != Ntot) {
        printf("Mismatch between expected and read eps Ntot: %d vs %d\n", Ntot, Ntot_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 3. read the Ntot_ coefficients
    epsoff = sizeof(uint) + sizeof(uint);
    check_mpi(MPI_File_read_at_all(epsfh, epsoff, epsilon.data(), Ntot_, MPI_DOUBLE, &status), __LINE__, __FILE__);

    //if (rank%50==0)
    //    printf("rank %d reading back eps: %15.10f %15.10f\n", rank, epsilon[0], epsilon[Ntot_-1]);

    check_mpi(MPI_File_close(&epsfh), __LINE__, __FILE__);
}


//EO: Watch out the saving frequency of the betas (--thin)
//    Each line contains simply the iteration (uint) and the value of mu (double)
void Data::read_mcmc_output_mus_file(const string mcmcOut,
                                     const uint   iteration_to_restart_from,
                                     const uint   first_thinned_iteration,
                                     const int    opt_thin,
                                     double&      mu) {

    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const string musfp = mcmcOut + ".mus." + std::to_string(rank);
    //cout << "musfp = " << musfp << endl;

    MPI_Status status;

    MPI_File musfh;
    check_mpi(MPI_File_open(MPI_COMM_WORLD, musfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &musfh), __LINE__, __FILE__);

    MPI_Offset musoff = size_t(0);

    // 1. get and validate iteration number that we are about to read
    int n_skip = iteration_to_restart_from - first_thinned_iteration;
    assert(n_skip >= 0);
    assert(n_skip % opt_thin == 0);
    n_skip /= opt_thin;
    //cout << "number of lines to skip  = " << n_skip << endl;

    musoff = size_t(n_skip) * (sizeof(uint) + sizeof(double));
    uint iteration_ = UINT_MAX;
    check_mpi(MPI_File_read_at_all(musfh, musoff, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_to_restart_from) {
        printf("Mismatch between expected and read mus iteration: %d vs %d\n", iteration_to_restart_from, iteration_);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. read the value of mu
    musoff += sizeof(uint);
    check_mpi(MPI_File_read_at(musfh, musoff, &mu, 1, MPI_DOUBLE, &status), __LINE__, __FILE__);
    //printf("reading back mu = %15.10f at iteration %d\n", mu, iteration_);

    check_mpi(MPI_File_close(&musfh), __LINE__, __FILE__);
}




//EO: .cpn, as per .bet contains full history with iteration saved at --opt.thin
//
void Data::read_mcmc_output_cpn_file(const string mcmcOut, const uint Mtot, 
                                     const uint   iteration_to_restart_from,
                                     const uint   first_thinned_iteration,
                                     const int    opt_thin,
                                     const int*   MrankS,  const int* MrankL,
                                     const bool   use_xfiles,
                                     VectorXi&    components) {

    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    string cpnfp = mcmcOut + ".cpn";
    if (use_xfiles) cpnfp = mcmcOut + ".xcpn";
    if (rank == 0) {
        if (use_xfiles) {
            printf("RESTART: reading cpn info back from last iteration xfile %s\n", cpnfp.c_str());
        } else {
            printf("RESTART: reading cpn info back from full history file %s\n", cpnfp.c_str());
        }
    }

    MPI_Status status;

    MPI_File cpnfh;
    check_mpi(MPI_File_open(MPI_COMM_WORLD, cpnfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &cpnfh), __LINE__, __FILE__);

    // 1. first element of the .bet, .cpn and .acu files is the total number of processed markers
    uint Mtot_ = 0;
    MPI_Offset cpnoff = size_t(0);
    check_mpi(MPI_File_read_at_all(cpnfh, cpnoff, &Mtot_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (Mtot_ != Mtot) {
        printf("Mismatch between expected and read cpn Mtot: %d vs %d\n", Mtot, Mtot_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. get and validate iteration number that we are about to read
    int n_skip = iteration_to_restart_from - first_thinned_iteration;
    assert(n_skip >= 0);
    assert(n_skip % opt_thin == 0);
    n_skip /= opt_thin;
    //cout << "number of lines to skip  = " << n_skip << endl;

    cpnoff = sizeof(uint) + size_t(n_skip) * (sizeof(uint) + size_t(Mtot_) * sizeof(int));
    if (use_xfiles) cpnoff = sizeof(uint);

    uint iteration_ = UINT_MAX;
    check_mpi(MPI_File_read_at_all(cpnfh, cpnoff, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_to_restart_from) {
        printf("Mismatch between expected and read cpn iteration: %d vs %d\n", iteration_to_restart_from, iteration_);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 3. read the Mtot_ coefficients
    cpnoff = sizeof(uint) + sizeof(uint) 
        + size_t(n_skip) * (sizeof(uint) + size_t(Mtot_) * sizeof(int))
        + size_t(MrankS[rank]) * sizeof(int);
    if (use_xfiles) {
        cpnoff = sizeof(uint) + sizeof(uint) + size_t(MrankS[rank]) * sizeof(int);
    }
    check_mpi(MPI_File_read_at_all(cpnfh, cpnoff, components.data(), MrankL[rank], MPI_INTEGER, &status), __LINE__, __FILE__);
    //cout << "cpn = " << components << endl;
    //printf("reading back cpn: %d %d\n", components[0], components[MrankL[rank]-1]);

    check_mpi(MPI_File_close(&cpnfh), __LINE__, __FILE__);
}


//EO: Watch out the saving frequency of the betas (--thin)
void Data::read_mcmc_output_bet_file(const string mcmcOut,           const uint Mtot,
                                     const uint   iteration_to_restart_from,
                                     const uint   first_thinned_iteration,
                                     const int    opt_thin,
                                     const int*   MrankS,  const int* MrankL,
                                     const bool   use_xfiles,
                                     VectorXd& Beta) {

    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    string betfp = mcmcOut + ".bet";
    if (use_xfiles) betfp = mcmcOut + ".xbet";

    if (rank == 0) {
        if (use_xfiles) {
            printf("RESTART: reading bet info back from last iteration xfile %s\n", betfp.c_str());
        } else {
            printf("RESTART: reading bet info back from full history file %s\n", betfp.c_str());
        }
    }

    MPI_Status status;

    MPI_File betfh;

    check_mpi(MPI_File_open(MPI_COMM_WORLD, betfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &betfh), __LINE__, __FILE__);

    // 1. first element of the .bet, .cpn and .acu files is the total number of processed markers
    uint Mtot_ = 0;
    MPI_Offset betoff = size_t(0);
    check_mpi(MPI_File_read_at_all(betfh, betoff, &Mtot_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (Mtot_ != Mtot) {
        printf("Mismatch between expected and read bet Mtot: %d vs %d\n", Mtot, Mtot_);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 2. get and validate iteration number that we are about to read
    int n_skip = iteration_to_restart_from - first_thinned_iteration;
    assert(n_skip >= 0);
    assert(n_skip % opt_thin == 0);
    n_skip /= opt_thin;

    betoff = sizeof(uint) + size_t(n_skip) * (sizeof(uint) + size_t(Mtot_) * sizeof(double));
    if (use_xfiles) betoff = sizeof(uint);

    uint iteration_ = UINT_MAX;
    check_mpi(MPI_File_read_at_all(betfh, betoff, &iteration_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (iteration_ != iteration_to_restart_from) {
        printf("Mismatch between expected and read bet iteration: %d vs %d\n", iteration_to_restart_from, iteration_);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // 3. read the Mtot_ coefficients
    betoff = sizeof(uint) + sizeof(uint) 
        + size_t(n_skip) * (sizeof(uint) + size_t(Mtot_) * sizeof(double))
        + size_t(MrankS[rank]) * sizeof(double);
    if (use_xfiles) { 
        betoff =  sizeof(uint) + sizeof(uint) + size_t(MrankS[rank]) * sizeof(double);
    }

    check_mpi(MPI_File_read_at_all(betfh, betoff, Beta.data(), MrankL[rank], MPI_DOUBLE, &status), __LINE__, __FILE__);

    //printf("rank %d reading back bet: %15.10f %15.10f\n", rank, Beta[0], Beta[MrankL[rank]-1]);
        

    check_mpi(MPI_File_close(&betfh), __LINE__, __FILE__);
}


void Data::read_mcmc_output_out_file(const string mcmcOut,
                                     const uint optThin, const uint optSave,
                                     const int K, VectorXd& sigmaG, double& sigmaE, MatrixXd& pi,
                                     uint& iteration_to_restart_from,
                                     uint& first_thinned_iteration,
                                     uint& first_saved_iteration) {
    
    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Status status;
    MPI_File fh;

    string fp = mcmcOut + ".out";

    size_t recl = sizeof(uint) + (size_t)(1 + 3) * sizeof(int) + (size_t)(sigmaG.size() + 2 + pi.size()) * sizeof(double);

    check_mpi(MPI_File_open(MPI_COMM_WORLD, fp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), __LINE__, __FILE__);

    MPI_Offset fs = 0;
    check_mpi(MPI_File_get_size(fh, &fs), __LINE__, __FILE__);

    if (fs == 0) {
        if (rank == 0)
            printf("*FATAL*: empty out file %s of record length %lu)\n", fp.c_str(), fs, recl);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (fs % recl == 0) {
        if (rank == 0)
            printf("INFO   : File %s contains %lu iteration records.\n",  fp.c_str(), fs / recl);
    } else {
        printf("*FATAL*: Something wrong with size of file %s (%lu Bytes, not a multiple of record length %lu)\n", fp.c_str(), fs, recl);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    // Read first dumped iteration as can be anything in case of (successive) restarts
    check_mpi(MPI_File_read_at_all(fh, 0, &first_thinned_iteration, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);

    // Get first saved iteration
    if (first_thinned_iteration == 0) {
        first_saved_iteration = optSave;
    } else {
        if (first_thinned_iteration % optSave == 0) {
            first_saved_iteration = (first_thinned_iteration / optSave) * optSave;
        } else {
            first_saved_iteration = (first_thinned_iteration / optSave + 1) * optSave;
        }
    }

    // Read latest saved iteration (saved at modulo opt.save but not saved at 0)
    int nrec = fs / recl;

    iteration_to_restart_from = (((nrec - 1) * optThin) / optSave) * optSave;

    if (iteration_to_restart_from == 0) {
        if (rank == 0) {
            printf("\n");
            printf("FATAL  : No saved iteration at modulo opt.save could be found when reading %s! (iteration_to_restart_from = %d with --save=%d)!\n",
                   fp.c_str(), iteration_to_restart_from, optSave);
            fflush(stdout);
        } 
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    MPI_Offset offset = (iteration_to_restart_from / optThin) * recl;

    int it_ = -1;
    check_mpi(MPI_File_read_at_all(fh, offset, &it_, 1, MPI_UNSIGNED, &status), __LINE__, __FILE__);
    if (it_ != iteration_to_restart_from) {
        if (rank == 0) {
            printf("\n");
            printf("FATAL  : Expected to read last saved iteration %d but read %d!\n", iteration_to_restart_from, it_);
            fflush(stdout);
        } 
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    offset += sizeof(uint);

    int sigg_size = 0;
    check_mpi(MPI_File_read_at_all(fh, offset, &sigg_size, 1, MPI_INTEGER, &status), __LINE__, __FILE__);
    offset += sizeof(int);

    check_mpi(MPI_File_read_at_all(fh, offset, sigmaG.data(), sigg_size, MPI_DOUBLE, &status), __LINE__, __FILE__);
    offset += sigg_size * sizeof(double);

    check_mpi(MPI_File_read_at_all(fh, offset, &sigmaE, 1, MPI_DOUBLE, &status), __LINE__, __FILE__);
    offset += sizeof(double);

    offset += sizeof(double); // skipping stuff not needed in restart
    offset += sizeof(int);

    int pi_rows_ = 0, pi_cols_ = 0;
    check_mpi(MPI_File_read_at_all(fh, offset, &pi_rows_, 1, MPI_INTEGER, &status), __LINE__, __FILE__);
    offset += sizeof(int);
    check_mpi(MPI_File_read_at_all(fh, offset, &pi_cols_, 1, MPI_INTEGER, &status), __LINE__, __FILE__);
    offset += sizeof(int);

    int pi_size_ = pi_rows_ * pi_cols_;
    assert(pi_size_ == pi.size());

    check_mpi(MPI_File_read_at_all(fh, offset, pi.data(), pi.size(), MPI_DOUBLE, &status), __LINE__, __FILE__);


    check_mpi(MPI_File_close(&fh), __LINE__, __FILE__);
}



//EO: consider moving the csv output file from ASCII to BIN
// Note: .csv and .out are history files written at modulo optThin (with optSave % optThin = 0)
//
void Data::read_mcmc_output_csv_file(const string mcmcOut,
                                     const uint optThin, const uint optSave,
                                     const int K, VectorXd& sigmaG, double& sigmaE, MatrixXd& pi,
                                     uint& iteration_to_restart_from,
                                     uint& first_thinned_iteration,
                                     uint& first_saved_iteration) {
    
    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    string        csv = mcmcOut + ".csv";
    std::ifstream file(csv);
    int           it_ = 1E9, rank_ = -1, m0_ = -1, pirows_ = -1, picols_ = -1, nchar = 0, ngrp_ = -1;
    int           nSavedIt = 0, nThinnedIt = 0;
    double        mu_, rat_, tmp;
    VectorXd      sigg_(numGroups);
    MatrixXd      pipi(numGroups,K);
    
    if (file.is_open()) {

        std::string str;

        while (std::getline(file, str)) {

            if (str.length() > 0) {

                int nread = sscanf(str.c_str(), "%5d, %4d", &it_, &ngrp_);
                //printf("it %5d | ngrp %4d with optSave = %d\n", it_, ngrp_, optSave);

                assert(it_ % optThin == 0);
                nThinnedIt += 1;
                if (nThinnedIt == 1) {
                    first_thinned_iteration = it_;
                    printf("#######: iteration %d is first_thinned_iteration\n", first_thinned_iteration);
                }


                if (it_ % optSave == 0) {

                    nSavedIt += 1;
                    
                    //EO: in case of multiple restarts, we need to know where
                    //    we restarted from last time
                    if (nSavedIt == 1) {
                        first_saved_iteration = it_;
                        printf("#######: iteration %d is first_saved_iteration\n", first_saved_iteration);
                    }
                    
                    iteration_to_restart_from = it_;

                    char cstr[str.length()+1];
                    strcpy(cstr, str.c_str());
                    nread = sscanf(cstr, "%5d, %4d, %n", &it_, &ngrp_, &nchar);
                    string remain_s = str.substr(nchar, str.length() - nchar);
                    char   remain_c[remain_s.length() + 1];
                    strcpy(remain_c, remain_s.c_str());

                    // remove commas
                    for (int i=0; i<remain_s.length(); ++i) {
                        if (remain_c[i] == ',') remain_c[i] = ' ';
                    }

                    char* prc = remain_c;
                    for (int i=0; i<numGroups; i++) {
                        sigmaG[i] = strtod(prc, &prc);
                        //printf("sigmaG[%d] = %20.15f\n", i, sigmaG[i]); 
                    }

                    sigmaE = strtod(prc, &prc);
                    //printf("sigmaE = %20.15f\n", sigmaE);

                    rat_ = strtod(prc, &prc);
                    //printf("rat_ = %20.15f\n", rat_);
                    
                    m0_ = std::strtol(prc, &prc, 10);
                    //printf("m0_ = %d\n", m0_);

                    pirows_ = std::strtol(prc, &prc, 10);
                    //printf("pirows_ = %d\n", pirows_);
                    assert(pirows_ == ngrp_);
                    assert(pi.rows() == pirows_);

                    picols_ = std::strtol(prc, &prc, 10);
                    //printf("picols_ = %d\n", picols_);
                    assert(pi.cols() == picols_);


                    for (int i=0; i<pirows_; i++) {
                        for (int j=0; j<picols_; j++) {
                            pi(i, j) = strtod(prc, &prc);                            
                        }
                    }
                }
            }
        }
    } else {
        printf("*FATAL*: failed to open csv file %s!\n", csv.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    //EO: if no saved iteration is found in the .csv file, we kill the process
    if (nSavedIt == 0) {
        if (rank == 0) {
            printf("\n");
            printf("FATAL  : No saved iteration could be found when reading %s!\n", csv.c_str());
            printf("       : with first read iteration = %d, last read iteration %d, and --thin = %d and --save = %d\n\n", first_thinned_iteration, it_, optThin, optSave);
            fflush(stdout);
        } 
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}


// SO: Currently basically the copy of the previous version.
//     Consider using adding reading mu for bR so the same function could be used
void Data::read_mcmc_output_csv_file_bW(const string mcmcOut, const uint optThin, const uint optSave, const int K, double& mu,
                                        VectorXd& sigmaG, double& sigmaE, MatrixXd& pi, 
                                        uint& iteration_to_restart_from, uint& first_thinned_iteration, uint& first_saved_iteration){
    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    string csv = mcmcOut + ".csv";
    std::ifstream file(csv);
   
    int it_ = 1E9, rank_ = -1, m0_ = -1, pirows_ = -1, picols_ = -1, nchar = 0;
    int nSavedIt = 0, nThinnedIt = 0;
    double mu_, sige_, rat_, siggSum_;

    VectorXd sigg_(numGroups);

    if (file.is_open()) {
        std::string str;
        while (std::getline(file, str)) {
            if (str.length() > 0) {
                int nread = sscanf(str.c_str(), "%5d", &it_);
 		assert(it_ % optThin == 0);
                nThinnedIt += 1;
                if (nThinnedIt == 1) {
                    first_thinned_iteration = it_;
                    printf("#######: iteration %d is first_thinned_iteration\n", first_thinned_iteration);
                }

                if (it_%optSave == 0) {
                    nSavedIt += 1;
                    
                    //EO: in case of multiple restarts, we need to know where
                    //    we restarted from last time
                    if (nSavedIt == 1) {
                        first_saved_iteration = it_;
                        printf("#######: iteration %d is first_saved_iteration\n", first_saved_iteration);
                    }
                    
                    iteration_to_restart_from = it_;                    

		    char cstr[str.length()+1];
                    strcpy(cstr, str.c_str());
                    nread = sscanf(cstr, "%5d, %lf, %lf, %lf, %lf, %7d, %7d, %2d, %n",
                                   &it_, &mu_, &siggSum_, &sige_, &rat_, &m0_, &pirows_, &picols_, &nchar);
	
                    assert(pi.rows() == pirows_);
                    assert(pi.cols() == picols_);
	
                    string remain_s = str.substr(nchar, str.length() - nchar);
                    char   remain_c[remain_s.length() + 1];
                    strcpy(remain_c, remain_s.c_str());

                    // remove commas
                    for (int i=0; i < remain_s.length(); ++i) {
                        if (remain_c[i] == ',') remain_c[i] = ' ';
                    }

                    char* prc = remain_c;
                    for (int i=0; i < numGroups; i++) {
                        sigmaG[i] = strtod(prc, &prc);
                        //printf("sigmaG[%d] = %20.15f\n", i, sigmaG[i]); 
                    }              


                    for (int i=0; i < pirows_; i++) {
                        for (int j=0; j < picols_; j++) {
                            pi(i, j) = strtod(prc, &prc);
                        }
                    }
 
		 }
            }
        }
    } else {
        printf("*FATAL*: failed to open csv file %s!\n", csv.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Assign
    mu		      = mu_;
    sigmaE            = sige_;

    //EO: if no saved iteration is found in the .csv file, we kill the process
    if (nSavedIt == 0) {
        if (rank == 0) {
            printf("\n");
            printf("FATAL  : No saved iteration could be found when reading %s!\n", csv.c_str());
            printf("       : with first read iteration = %d, last read iteration %d, and --thin = %d and --save = %d\n\n", first_thinned_iteration, it_, optThin, optSave);
            fflush(stdout);
        } 
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

}


// SO: Function to read fixed covariate effects from csv format file. Consider using adding reading mu for bR so the same function could be used
void Data::read_mcmc_output_gam_file_bW(const string mcmcOut, const uint optSave, const int gamma_length, 
                                     VectorXd& gamma) {

    int nranks, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    string csv = mcmcOut + ".gam";
    std::ifstream file(csv);
    int it_ = 1E9, nchar = 0;
    double gamma_[gamma_length];

    if (file.is_open()) {
        std::string str;
        while (std::getline(file, str)) {
            if (str.length() > 0) {
                int nread = sscanf(str.c_str(), "%5d", &it_);
                if (it_%optSave == 0) {
                    char cstr[str.length()+1];
                    strcpy(cstr, str.c_str());
                    nread = sscanf(cstr, "%5d, %n",
                                   &it_, &nchar);
	    string pis = str.substr(nchar, str.length()-nchar);
	    char   pic[pis.length()+1];
                    strcpy(pic, pis.c_str());
                    for (int i=0; i<pis.length(); ++i) {
                        if (pic[i] == ',') pic[i] = ' ';
                    }
                    //cout << "pic = " << pic << endl;
                    char* ppic = pic;
                    for (int i = 0; i < gamma_length; i++) {
                        gamma_[i] = strtod(ppic, &ppic);
                    }
                }
            }
        }
    } else {
        printf("*FATAL*: failed to open csv file %s!\n", csv.c_str());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Assign
    for (int i = 0; i < gamma_length; i++)
        gamma[i] = gamma_[i];
}


//EO: load data from a bed file
// TODO: clean up API + code
// ----------------------------
void Data::load_data_from_bed_file(const string bedfp_noext, const uint Ntot, const int M,
                                   const int rank, const int start,
                                   size_t* N1S, size_t* N1L, uint*& I1,
                                   size_t* N2S, size_t* N2L, uint*& I2,
                                   size_t* NMS, size_t* NML, uint*& IM,
                                   size_t& taskBytes) {

    MPI_File   bedfh;
    MPI_Offset offset;

    string bedfp = bedfp_noext + ".bed";
    check_mpi(MPI_File_open(MPI_COMM_WORLD, bedfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &bedfh),  __LINE__, __FILE__);

    // Length of a "column" in bytes
    const size_t snpLenByt = (Ntot % 4) ? Ntot / 4 + 1 : Ntot / 4;
    if (rank==0) printf("INFO   : marker length in bytes (snpLenByt) = %zu bytes.\n", snpLenByt);


    // Alloc memory for raw BED data
    // -----------------------------
    const size_t rawdata_n = size_t(M) * size_t(snpLenByt) * sizeof(char);
    taskBytes = rawdata_n;
    char* rawdata = (char*)_mm_malloc(rawdata_n, 64);  check_malloc(rawdata, __LINE__, __FILE__);
    //printf("rank %d allocation %zu bytes (%.3f GB) for the raw data.\n", rank, rawdata_n, double(rawdata_n/1E9));


    // Compute the offset of the section to read from the BED file
    // -----------------------------------------------------------
    //offset = size_t(3) + size_t(MrankS[rank]) * size_t(snpLenByt) * sizeof(char);
    offset = size_t(3) + size_t(start) * size_t(snpLenByt) * sizeof(char);


    // Read the BED file
    // -----------------
    MPI_Barrier(MPI_COMM_WORLD);
    //const auto st1 = std::chrono::high_resolution_clock::now();

    // Gather the sizes to determine common number of reads
    size_t rawdata_n_max = 0;
    check_mpi(MPI_Allreduce(&rawdata_n, &rawdata_n_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);

    int NREADS = check_int_overflow(size_t(ceil(double(rawdata_n_max)/double(INT_MAX/2))), __LINE__, __FILE__);
    size_t bytes = 0;
    mpi_file_read_at_all <char*> (rawdata_n, offset, bedfh, MPI_CHAR, NREADS, rawdata, bytes);

    MPI_Barrier(MPI_COMM_WORLD);

    // Close BED file
    check_mpi(MPI_File_close(&bedfh), __LINE__, __FILE__);
   

    size_t N1 = 0, N2 = 0, NM = 0;

    //EO: reading from bed file, so NAs are not considered yet
    sparse_data_get_sizes_from_raw(rawdata, M, snpLenByt, 0, N1, N2, NM);
    //printf("read from bed: N1 = %lu, N2 = %lu, NM = %lu\n", N1, N2, NM);

    // Alloc and build sparse structure
    I1 = (uint*)_mm_malloc(N1 * sizeof(uint), 64);  check_malloc(I1, __LINE__, __FILE__);
    I2 = (uint*)_mm_malloc(N2 * sizeof(uint), 64);  check_malloc(I2, __LINE__, __FILE__);
    IM = (uint*)_mm_malloc(NM * sizeof(uint), 64);  check_malloc(IM, __LINE__, __FILE__);
    
    sparse_data_fill_indices(rawdata, M, snpLenByt, 0,
                             N1S, N1L, I1,
                             N2S, N2L, I2,
                             NMS, NML, IM);

    _mm_free(rawdata);        
}


void Data::load_data_from_sparse_files(const int rank,           const int nranks,  const int M,
                                       const int* MrankS,        const int* MrankL,
                                       const string sparseOut,
                                       size_t* N1S, size_t* N1L, uint*& I1,
                                       size_t* N2S, size_t* N2L, uint*& I2,
                                       size_t* NMS, size_t* NML, uint*& IM,
                                       size_t& taskBytes) {

    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int  processor_name_len;
    MPI_Get_processor_name(processor_name, &processor_name_len);

    // Get sizes to alloc for the task
    size_t N1 = get_number_of_elements_from_sparse_files(sparseOut, "1", MrankS, MrankL, N1S, N1L);
    size_t N2 = get_number_of_elements_from_sparse_files(sparseOut, "2", MrankS, MrankL, N2S, N2L);
    size_t NM = get_number_of_elements_from_sparse_files(sparseOut, "m", MrankS, MrankL, NMS, NML);


    // EO: Disable this, not sure whether this is really useful information
    /*
    size_t N1max = 0, N2max = 0, NMmax = 0;
    check_mpi(MPI_Allreduce(&N1, &N1max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allreduce(&N2, &N2max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allreduce(&NM, &NMmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
    
    size_t N1tot = 0, N2tot = 0, NMtot = 0;
    check_mpi(MPI_Allreduce(&N1, &N1tot, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allreduce(&N2, &N2tot, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allreduce(&NM, &NMtot, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);

    totalBytes =  (N1tot + N2tot + NMtot) * sizeof(uint);

    if (rank % 10 == 0) {
        printf("INFO   : rank %3d/%3d  N1max = %15lu, N2max = %15lu, NMmax = %15lu\n", rank, nranks, N1max, N2max, NMmax);
        printf("INFO   : rank %3d/%3d  N1tot = %15lu, N2tot = %15lu, NMtot = %15lu\n", rank, nranks, N1tot, N2tot, NMtot);
        printf("INFO   : RAM for task %3d/%3d on node %s: %7.3f GB\n", rank, nranks, processor_name, (N1 + N2 + NM) * sizeof(uint) / 1E9);
    }
    */

    taskBytes = (N1 + N2 + NM) * sizeof(uint);

    /*
    if (rank == 0) 
        printf("INFO   : Total RAM for storing sparse indices %.3f GB\n", double(totalBytes) * 1E-9);
    fflush(stdout);
    */

    I1 = (uint*)_mm_malloc(N1 * sizeof(uint), 64);  check_malloc(I1, __LINE__, __FILE__);
    I2 = (uint*)_mm_malloc(N2 * sizeof(uint), 64);  check_malloc(I2, __LINE__, __FILE__);
    IM = (uint*)_mm_malloc(NM * sizeof(uint), 64);  check_malloc(IM, __LINE__, __FILE__);

    //EO: base the number of read calls on a max buffer size of 2 GiB
    //    rather than count be lower that MAX_INT/2
    //----------------------------------------------------------------------
    int NREADS1 = int(ceil(double(N1 * sizeof(uint)) / double(2147483648)));
    int NREADS2 = int(ceil(double(N2 * sizeof(uint)) / double(2147483648)));
    int NREADSM = int(ceil(double(NM * sizeof(uint)) / double(2147483648)));

    //printf("INFO   : rank %d, number of calls to read the sparse files: NREADS1 = %d, NREADS2 = %d, NREADSM = %d\n", rank, NREADS1, NREADS2, NREADSM);
    //fflush(stdout);
   
    //EO: to keep the read_at_all in, we need to get the max nreads
    int MAX_NREADS1 = 0, MAX_NREADS2 = 0, MAX_NREADSM = 0;
    check_mpi(MPI_Allreduce(&NREADS1, &MAX_NREADS1, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allreduce(&NREADS2, &MAX_NREADS2, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allreduce(&NREADSM, &MAX_NREADSM, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);

    if (rank % 10 ==0) {
        printf("INFO   : rank %3d, number of calls to read the sparse files: NREADS{1,2,M} = %3d, %3d, %3d vs MAX_NREADS1,2,M = %3d, %3d, %3d\n",
               rank, NREADS1, NREADS2, NREADSM, MAX_NREADS1, MAX_NREADS2, MAX_NREADSM);
        fflush(stdout);
    }

    read_sparse_data_file(sparseOut + ".si1", N1, N1S[0], MAX_NREADS1, I1);
    read_sparse_data_file(sparseOut + ".si2", N2, N2S[0], MAX_NREADS2, I2);
    read_sparse_data_file(sparseOut + ".sim", NM, NMS[0], MAX_NREADSM, IM);    

    // Make starts relative to start of block in each task
    const size_t n1soff = N1S[0];  for (int i=0; i<M; ++i) { N1S[i] -= n1soff; }
    const size_t n2soff = N2S[0];  for (int i=0; i<M; ++i) { N2S[i] -= n2soff; }
    const size_t nmsoff = NMS[0];  for (int i=0; i<M; ++i) { NMS[i] -= nmsoff; }
}


void Data::get_bed_marker_from_sparse(char* bdat, 
                                      const int    Ntot,
                                      const size_t S1, const size_t L1, const uint* I1,
                                      const size_t S2, const size_t L2, const uint* I2,
                                      const size_t SM, const size_t LM, const uint* IM) const {


    // Length of a marker in [byte] in BED representation (1 ind is 2 bits, so 1 byte is 4 inds)
    //cout << "ntot = " << Ntot << endl;
    //printf("L1,2,M = %lu, %lu, %lu\n", L1,L2,LM);
    const size_t snpLenByt  = (Ntot %  4) ? Ntot /  4 + 1 : Ntot /  4;

    // Step 1: Set all bits to 1
    memset(bdat, 0b11111111, snpLenByt);
    

    // Step 2: Set the 1s
    int ibyt = 0, iind = 0;
    for (int j=0; j<L1; j++) {
        ibyt = I1[j] / 4;          // 4 inds per byte in BED format
        iind = I1[j] - ibyt * 4;   // 2 bits per ind in byte, position in byte from right to left
        bdat[ibyt] ^= (0b00000001 << iind * 2);
    }
    
    // Step 3: Set the 2s
    ibyt = 0, iind = 0;
    for (int j=0; j<L2; j++) {
        ibyt = I2[j] / 4;
        iind = I2[j] - ibyt * 4;
        bdat[ibyt] ^= (0b00000011 << iind * 2);
    }
    
    // Step 4: Set the Ms
    ibyt = 0, iind = 0;
    for (int j=0; j<LM; j++) {
        ibyt = IM[j] / 4;
        iind = IM[j] - ibyt * 4;
        bdat[ibyt] ^= (0b00000010 << iind * 2);
    }
}



// EO Mixed-type representation: BED + SPARSE
// Based on a criterion (threshold_fnz) giving the fraction of non-zero elements in 
// the genotype, some markers will be handled as BED data, the others as SPARSE.
// The BED information can be either (hard-coded switch):
//   1) read directly from BED file
//   2) read from SPARSE files and converted to BED (default)
//
// The structures holding the 1s are use to hold the BED information, the others
// being not used and lengths made 0.
//
// For BED format description: http://zzz.bwh.harvard.edu/plink/binary.shtml
// Byte read backwards: from right to left: 33221100
// BUT:
// !!!   WE USE AN INVERTED BED FORMAT REPRESENTATION   !!!
// So zero is coded as 11, and so on...
// ---------------------------------------------------------------------------------

void Data::load_data_from_mixed_representations(const string bedfp_noext,   const string sparseOut,
                                                const int    rank,          const int    nranks,
                                                const int    Ntot,          const int    M,
                                                const int*   MrankS,        const int*   MrankL,
                                                size_t* N1S,  size_t* N1L,  uint*& I1,
                                                size_t* N2S,  size_t* N2L,  uint*& I2,
                                                size_t* NMS,  size_t* NML,  uint*& IM,
                                                const double threshold_fnz, bool* USEBED,
                                                size_t& taskBytes) {


    // Get N.S and N.L arrays filled from sparse files
    size_t N1c = get_number_of_elements_from_sparse_files(sparseOut, "1", MrankS, MrankL, N1S, N1L);
    size_t N2c = get_number_of_elements_from_sparse_files(sparseOut, "2", MrankS, MrankL, N2S, N2L);
    size_t NMc = get_number_of_elements_from_sparse_files(sparseOut, "m", MrankS, MrankL, NMS, NML);


    // Length of a marker in [byte] in BED representation (1 ind is 2 bits, so 1 byte is 4 inds)
    const size_t snpLenByt  = (Ntot %  4) ? Ntot /  4 + 1 : Ntot /  4;

    // Length of a marker in [uint] in BED representation (1 ind is 2 bits, so 1 uint is 16 inds)
    const size_t snpLenUint = (Ntot % 16) ? Ntot / 16 + 1 : Ntot / 16;


    // Make a copy of the original Length arrays
    //
    size_t *sparse_N1S = (size_t*)_mm_malloc(M * sizeof(size_t), 64);  check_malloc(sparse_N1S, __LINE__, __FILE__);
    size_t *sparse_N2S = (size_t*)_mm_malloc(M * sizeof(size_t), 64);  check_malloc(sparse_N2S, __LINE__, __FILE__);
    size_t *sparse_NMS = (size_t*)_mm_malloc(M * sizeof(size_t), 64);  check_malloc(sparse_NMS, __LINE__, __FILE__);

    for (int i=0; i<M; i++) {
        sparse_N1S[i] = N1S[i];
        sparse_N2S[i] = N2S[i];
        sparse_NMS[i] = NMS[i];
    }


    // Compute size of arrays to allocate for the mixed representation
    // 2 & M as usual
    // 1 to hold BED data only, that is snpLenUint uints
    // ---------------------------------------------------------------
    size_t N1 = 0, N2 = 0, NM = 0;

    for (int i=0; i<M; i++) {
        
        double mfnz = double(N1L[i] + N2L[i] + NML[i]) / double(Ntot);
        USEBED[i] = (mfnz > threshold_fnz) ? true : false;

        N1S[i] = N1;
        N2S[i] = N2;
        NMS[i] = NM;

        if (USEBED[i]) {
            //printf("rank %2d, marker %2d: WILL USE BED\n", rank, i);
            N1 += snpLenUint; //EO: store all BED in N1
            //N2 += 0; NM += 0;
        } else {
            N1 += N1L[i]; 
            N2 += N2L[i];
            NM += NML[i];
        }
        //printf("rank %02d, marker %7d: 1,2,M lengths = %6lu, %6lu, %6lu => %d\n", rank, i, N1L[i], N2L[i], NML[i], USEBED[i]);
    }

    // Alloc as uint
    taskBytes = 0;
    I1 = (uint*)_mm_malloc(N1 * sizeof(uint), 64);  check_malloc(I1, __LINE__, __FILE__);
    I2 = (uint*)_mm_malloc(N2 * sizeof(uint), 64);  check_malloc(I2, __LINE__, __FILE__);
    IM = (uint*)_mm_malloc(NM * sizeof(uint), 64);  check_malloc(IM, __LINE__, __FILE__);

    taskBytes += (N1 + N2 + NM) * sizeof(uint);

    MPI_File   bedfh, si1fh, si2fh, simfh;
    MPI_Offset bedoff;
    MPI_Status status;

    //string bedfp = bedfp_noext + ".bed";
    string si1fp = sparseOut   + ".si1";
    string si2fp = sparseOut   + ".si2";
    string simfp = sparseOut   + ".sim";

    //check_mpi(MPI_File_open(MPI_COMM_WORLD, bedfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &bedfh),  __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, si1fp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &si1fh),  __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, si2fp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &si2fh),  __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, simfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &simfh),  __LINE__, __FILE__);

    int dtsize = 0;
    MPI_Type_size(MPI_UNSIGNED, &dtsize);


    const bool   FAKE_USEBED[1] = {false};
    const size_t FAKE_NS[1]     = {0};


#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<M; i++) {

        // For markers elected for BED representation we either read from BED file or
        // from SPARSE files and convert to BED
        // By default: read from SPARSE

        if (USEBED[i]) {

            if (1 == 0) {

                //bedoff = size_t(3) + (size_t(MrankS[rank]) + i) * size_t(snpLenByt) * sizeof(char);
                //check_mpi(MPI_File_read_at(bedfh, bedoff, &I1[N1S[i]], snpLenByt, MPI_CHAR, &status), __LINE__, __FILE__);

            } else {

                // Step 0: Cast allocated uint memory of size snpLenUint to char to hold BED data
                char* bdat = reinterpret_cast<char*>(&I1[N1S[i]]);

                // Step 1: Set all to 1 
                memset(bdat, 0b11111111, snpLenUint * 4);

                // Step 2: Set the 1s
                uint* tmp = (uint*)_mm_malloc(N1L[i] * size_t(dtsize), 64);  check_malloc(tmp, __LINE__, __FILE__);
                check_mpi(MPI_File_read_at(si1fh, sparse_N1S[i] * size_t(dtsize), tmp, N1L[i], MPI_UNSIGNED, &status), __LINE__, __FILE__);
                sparse_data_correct_for_missing_phenotype(FAKE_NS, &N1L[i], tmp, 1, FAKE_USEBED);
                fflush(stdout);
                int ibyt = 0, iind = 0;
                for (int j=0; j<N1L[i]; j++) {
                    ibyt = tmp[j] / 4;          // 4 inds per byte in BED format
                    iind = tmp[j] - ibyt * 4;   // 2 bits per ind in byte, position in byte from right to left
                    bdat[ibyt] ^= (0b00000001 << iind * 2);
                }
                _mm_free(tmp);
                
                // Step 3: Set the 2s
                tmp = (uint*)_mm_malloc(N2L[i] * size_t(dtsize), 64);  check_malloc(tmp, __LINE__, __FILE__);
                check_mpi(MPI_File_read_at(si2fh, sparse_N2S[i] * size_t(dtsize), tmp, N2L[i], MPI_UNSIGNED, &status), __LINE__, __FILE__);
                sparse_data_correct_for_missing_phenotype(FAKE_NS, &N2L[i], tmp, 1, FAKE_USEBED);
                ibyt = 0, iind = 0;
                for (int j=0; j<N2L[i]; j++) {
                    ibyt = tmp[j] / 4;
                    iind = tmp[j] - ibyt * 4;
                    bdat[ibyt] ^= (0b00000011 << iind * 2);
                }
                _mm_free(tmp);

                // Step 4: Set the Ms
                tmp = (uint*)_mm_malloc(NML[i] * size_t(dtsize), 64);  check_malloc(tmp, __LINE__, __FILE__);
                check_mpi(MPI_File_read_at(simfh, sparse_NMS[i] * size_t(dtsize), tmp, NML[i], MPI_UNSIGNED, &status), __LINE__, __FILE__);
                sparse_data_correct_for_missing_phenotype(FAKE_NS, &NML[i], tmp, 1, FAKE_USEBED);
                ibyt = 0, iind = 0;
                for (int j=0; j<NML[i]; j++) {
                    ibyt = tmp[j] / 4;
                    iind = tmp[j] - ibyt * 4;
                    bdat[ibyt] ^= (0b00000010 << iind * 2);
                }
                _mm_free(tmp);
           }

            // I1 now holds the raw BED data;
            N1L[i] = snpLenUint;
            N2L[i] = 0;
            NML[i] = 0;

        } else {         // read SPARSE files

            check_mpi(MPI_File_read_at(si1fh, sparse_N1S[i] * size_t(dtsize), &I1[N1S[i]], N1L[i], MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_read_at(si2fh, sparse_N2S[i] * size_t(dtsize), &I2[N2S[i]], N2L[i], MPI_UNSIGNED, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_read_at(simfh, sparse_NMS[i] * size_t(dtsize), &IM[NMS[i]], NML[i], MPI_UNSIGNED, &status), __LINE__, __FILE__);
        }

        //printf("----> out rank %2d, marker %2d (S): %lu, %lu, %lu (L): %lu, %lu, %lu\n", rank, i, N1S[i], N2S[i], NMS[i], N1L[i], N2L[i], NML[i]);
    }

    _mm_free(sparse_N1S);
    _mm_free(sparse_N2S);
    _mm_free(sparse_NMS);


    //MPI_Barrier(MPI_COMM_WORLD);

    // Close BED and SPARSE files
    //check_mpi(MPI_File_close(&bedfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&si1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&si2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&simfh), __LINE__, __FILE__);
}


size_t Data::get_number_of_elements_from_sparse_files(const std::string basename, const std::string id, const int* MrankS, const int* MrankL,
                                                      size_t* S, size_t* L) {

    MPI_Status status;
    MPI_File   ssfh, slfh;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Number of markers in block handled by task
    const uint M = MrankL[rank];

    const std::string sl = basename + ".sl" + id;
    const std::string ss = basename + ".ss" + id;

    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &ssfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &slfh), __LINE__, __FILE__);

    // Compute the lengths of ones and twos vectors for all markers in the block
    MPI_Offset offset =  MrankS[rank] * sizeof(size_t);
    check_mpi(MPI_File_read_at_all(ssfh, offset, S, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);
    check_mpi(MPI_File_read_at_all(slfh, offset, L, M, MPI_UNSIGNED_LONG_LONG, &status), __LINE__, __FILE__);

    // Close sparse files
    check_mpi(MPI_File_close(&ssfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&slfh), __LINE__, __FILE__);


    // Absolute offsets in 0s, 1s, and 2s
    const size_t nsoff = S[0];

    size_t N = S[M-1] + L[M-1] - nsoff;

    return N;
}


// EO: Apply corrections to the sparse structures (1,2,m)
//     Watch out that NAs have to be considered globally accross the structures
// ---------------------------------------------------------------------------
void Data::sparse_data_correct_for_missing_phenotype(const size_t* NS, size_t* NL, uint* I, const int M, const bool* USEBED) {

#ifdef _OPENMP
#pragma omp parallel for
#endif
    for (int i=0; i<M; ++i) {

        // Skip for markers in BED representation (NA correction handled separately)
        if (USEBED[i]) continue;

        uint* tmp = (uint*)_mm_malloc(NL[i] * sizeof(uint), 64);  check_malloc(tmp, __LINE__, __FILE__);

        const size_t beg = NS[i], len = NL[i];

        size_t k   = 0;
        uint   nas = 0;

        if (len > 0) {

            // Make a tmp copy of the original data
            for (size_t iii=beg; iii<beg+len; ++iii) {
                tmp[iii-beg] = I[iii];
                //if (iii<3)  cout << "tmp[iii-beg] = " << tmp[iii-beg] << endl;
            }

            for (size_t iii=beg; iii<beg+len; ++iii) {
                bool isna   = false;
                uint allnas = 0;
                for (int ii=0; ii<numNAs; ++ii) {
                    if (NAsInds[ii] > tmp[iii-beg]) break;
                    if (NAsInds[ii] <= tmp[iii-beg]) allnas += 1;
                    if (tmp[iii-beg] == NAsInds[ii]) { // NA found
                        //cout << "found NA at " << tmp[iii-beg]  << endl;
                        isna = true;
                        nas += 1;
                        break;
                    }
                }
                if (isna) continue;
                I[beg+k] = tmp[iii-beg]  - allnas;
                k += 1;
            }
        }
        NL[i] -= nas;
        _mm_free(tmp);
    }
}


// EO: needs also NA as parameter, the number of already applied NA corrections
//     in the raw data.
//     -> 0 when reading from a bed file
//     -> data.numNAs when handling an buffered marker stored in BED format
//
void Data::sparse_data_get_sizes_from_raw(const char* rawdata, 
                                          const uint  NC,
                                          const uint  NB,
                                          const uint  NA,
                                          size_t& N1, size_t& N2, size_t& NM) const {

    assert(numInds - NA <= NB * 4);

    // temporary array used for translation
    int8_t *tmpi = (int8_t*)_mm_malloc(NB * 4 * sizeof(char), 64);  check_malloc(tmpi, __LINE__, __FILE__);

    N1 = 0;
    N2 = 0;
    NM = 0;

    size_t N0 = 0;

    for (uint i=0; i<NC; ++i) {

        uint c0 = 0, c1 = 0, c2 = 0, cm = 0;

        char* locraw = (char*)&rawdata[size_t(i)*size_t(NB)];

        for (int ii=0; ii<NB; ++ii) {
            for (int iii=0; iii<4; ++iii) {
                tmpi[ii*4 + iii] = (locraw[ii] >> 2*iii) & 0b11;
            }
        }
        
        for (int ii=0; ii<numInds-NA; ++ii) {
            if (tmpi[ii] == 1) {
                tmpi[ii] = -1;
            } else {
                tmpi[ii] =  2 - ((tmpi[ii] & 0b1) + ((tmpi[ii] >> 1) & 0b1));
            }
        }

        for (int ii=0; ii<numInds-NA; ++ii) {
            if      (tmpi[ii] <  0) { cm += 1;  NM += 1; }
            else if (tmpi[ii] == 0) { c0 += 1;  N0 += 1; }
            else if (tmpi[ii] == 1) { c1 += 1;  N1 += 1; }
            else if (tmpi[ii] == 2) { c2 += 1;  N2 += 1; }
        }
        //printf("N0, N1, N2, NM = %lu, %lu, %lu, %lu\n", N0, N1, N2, NM);
        //printf("%d - %d = %d\n", numInds, NA, numInds - NA);
        assert(cm+c0+c1+c2 == numInds - NA);        
        //printf("N0, N1, N2, NM = %lu, %lu, %lu, %lu\n", N0, N1, N2, NM);
    }

    _mm_free(tmpi);
}


// Screen the raw BED data and count and register number of ones, twos and missing information
// N*S: store the "S"tart of each marker representation 
// N*L: store the "L"ength of each marker representation
// I* : store the indices of the elements
// -------------------------------------------------------------------------------------------
void Data::sparse_data_fill_indices(const char* rawdata,
                                    const uint  NC,
                                    const uint  NB,
                                    const uint  NA,
                                    size_t* N1S, size_t* N1L, uint* I1,
                                    size_t* N2S, size_t* N2L, uint* I2,
                                    size_t* NMS, size_t* NML, uint* IM) const {
    
    assert(numInds - NA <= NB * 4);


    // temporary array used for translation
    int8_t *tmpi = (int8_t*)_mm_malloc(NB * 4 * sizeof(int8_t), 64);  check_malloc(tmpi, __LINE__, __FILE__);

    size_t i1 = 0, i2 = 0, im = 0;
    size_t N1 = 0, N2 = 0, NM = 0;
      
    for (int i=0; i<NC; ++i) {
        
        char* locraw = (char*)&rawdata[size_t(i) * size_t(NB)];
        
        for (int ii=0; ii<NB; ++ii) {
            for (int iii=0; iii<4; ++iii) {
                tmpi[ii*4 + iii] = (locraw[ii] >> 2*iii) & 0b11;
            }
        }
        
        for (int ii=0; ii<numInds-NA; ++ii) {
            if (tmpi[ii] == 1) {
                tmpi[ii] = -1;
            } else {
                tmpi[ii] =  2 - ((tmpi[ii] & 0b1) + ((tmpi[ii] >> 1) & 0b1));
            }
        }
        
        size_t n0 = 0, n1 = 0, n2 = 0, nm = 0;
        
        for (uint ii=0; ii<numInds-NA; ++ii) {
            if (tmpi[ii] < 0) {
                IM[im] = ii;
                im += 1;
                nm += -tmpi[ii];
            } else {
                if        (tmpi[ii] == 0) {
                    n0 += 1;
                } else if (tmpi[ii] == 1) {
                    I1[i1] = ii;
                    i1 += 1;
                    n1 += 1;
                } else if (tmpi[ii] == 2) {
                    I2[i2] = ii;
                    i2 += 1;
                    n2 += 1;
                }
            }
        }
        
        assert(nm + n0 + n1 + n2 == numInds - NA);
        
        N1S[i] = N1;  N1L[i] = n1;  N1 += n1;
        N2S[i] = N2;  N2L[i] = n2;  N2 += n2;
        NMS[i] = NM;  NML[i] = nm;  NM += nm;
    }

    _mm_free(tmpi);
}


void Data::read_sparse_data_file(const std::string filename, const size_t N, const size_t OFF, const int NREADS, uint* out) {

    MPI_Offset offset;
    MPI_Status status;
    MPI_File   fh;

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    check_mpi(MPI_File_open(MPI_COMM_WORLD, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh), __LINE__, __FILE__);

    offset = OFF * sizeof(uint);
    size_t bytes = 0;

    mpi_file_read_at_all<uint*>(N, offset, fh, MPI_UNSIGNED, NREADS, out, bytes);

    //EO: above call not collective anymore...
    MPI_Barrier(MPI_COMM_WORLD);

    check_mpi(MPI_File_close(&fh), __LINE__, __FILE__);
}


// Read raw data loaded in memory to preprocess them (center, scale and cast to double)
void Data::preprocess_data(const char* rawdata, const uint NC, const uint NB, double* ppdata, const int rank) {

    assert(numInds<=NB*4);

    // temporary array used for translation
    int8_t *tmpi = (int8_t*)_mm_malloc(NB * 4 * sizeof(int8_t), 64);  check_malloc(tmpi, __LINE__, __FILE__);

    for (int i=0; i<NC; ++i) {

        char* locraw = (char*)&rawdata[size_t(i) * size_t(NB)];

        for (int ii=0; ii<NB; ++ii) {
            for (int iii=0; iii<4; ++iii) {
                tmpi[ii*4 + iii] = (locraw[ii] >> 2*iii) & 0b11;
            }
        }
        
        for (int ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] == 1) {
                tmpi[ii] = -1;
            } else {
                tmpi[ii] =  2 - ((tmpi[ii] & 0b1) + ((tmpi[ii] >> 1) & 0b1));
            }
        }

        int sum = 0, nmiss = 0;
        //#pragma omp simd reduction(+:sum) reduction(+:nmiss)
        for (int ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] < 0) {
                nmiss += tmpi[ii];
            } else {
                sum   += tmpi[ii];
            }
        }

        double mean = double(sum) / double(numInds + nmiss); //EO: nmiss is neg
        //printf("rank %d snpInd %2d: sum = %6d, N = %6d, nmiss = %6d, mean = %20.15f\n",
        //       rank, rank*NC+i, sum, numKeptInds, nmiss, mean);

        size_t ppdata_i = size_t(i) * size_t(numInds);
        double *locpp = (double*)&ppdata[ppdata_i];

        for (size_t ii=0; ii<numInds; ++ii) {
            if (tmpi[ii] < 0) {
                locpp[ii] = 0.0;
            } else {
                locpp[ii] = double(tmpi[ii]) - mean;
            }
        }

        double sqn  = 0.0;
        for (size_t ii=0; ii<numInds; ++ii) {
            sqn += locpp[ii] * locpp[ii];
        }

        double std_ = sqrt(double(numInds - 1) / sqn);

        for (size_t ii=0; ii<numInds; ++ii) {
            locpp[ii] *= std_;
        }
    }

    _mm_free(tmpi);
}



// EO
// Read marker blocks definition file
// Line format:  "%d %d\n"; Nothing else is accepted.
// Gaps are allowed; Overlaps are forbiden
// --------------------------------------------------
void Data::readMarkerBlocksFile(const string &markerBlocksFile) {
    
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ifstream in(markerBlocksFile.c_str());
    if (!in) throw ("Error: can not open the file [" + markerBlocksFile + "] to read.");
    
    blocksStarts.clear();
    blocksEnds.clear();
    std::string str;
    
    while (std::getline(in, str)) {
        std::vector<std::string> results;
        boost::split(results, str, [](char c){return c == ' ';});
        if (results.size() != 2) {
            printf("Error: line with wrong format: >>%s<<\n", str.c_str());
            printf("       => expected format \"%%d %%d\": two integers separated with a single space, with no leading or trailing space.\n");
            exit(1);
        }
        blocksStarts.push_back(stoi(results[0]));
        blocksEnds.push_back(stoi(results[1]));        
    }
    in.close();

    numBlocks = (unsigned) blocksStarts.size();
    //cout << "Found definitions for " << nbs << " marker blocks." << endl;

    // Neither gaps or overlaps are accepted
    // -------------------------------------
    for (int i=0; i<numBlocks; ++i) {

        if (blocksStarts[i] > blocksEnds[i]) {
            if (rank == 0) {
                printf("FATAL  : block starts beyond end [%d, %d].\n", blocksStarts[i], blocksEnds[i]);
                printf("         => you must correct your marker blocks definition file %s\n", markerBlocksFile.c_str());
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        int j=i+1;
        if (j < numBlocks && blocksStarts[j] != blocksEnds[i] + 1) {
            if (rank == 0) {
                printf("FATAL  : block %d ([%d, %d]) and block %d ([%d, %d]) are not contiguous!\n", i, blocksStarts[i], blocksEnds[i], j, blocksStarts[j], blocksEnds[j]);
                printf("         => you must correct your marker blocks definition file %s\n", markerBlocksFile.c_str());
            }
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
}


void Data::readFamFile(const string &famFile){
    // ignore phenotype column
    ifstream in(famFile.c_str());
    if (!in) throw ("Error: can not open the file [" + famFile + "] to read.");
    //cout << "Reading PLINK FAM file from [" + famFile + "]." << endl;
    indInfoVec.clear();
    indInfoMap.clear();
    string fid, pid, dad, mom, sex, phen;
    unsigned idx = 0;
    while (in >> fid >> pid >> dad >> mom >> sex >> phen) {
        IndInfo *ind = new IndInfo(idx++, fid, pid, dad, mom, atoi(sex.c_str()));
        indInfoVec.push_back(ind);
        if (indInfoMap.insert(pair<string, IndInfo*>(ind->catID, ind)).second == false) {
            throw ("Error: Duplicate individual ID found: \"" + fid + "\t" + pid + "\".");
        }
    }
    in.close();
    numInds = (unsigned) indInfoVec.size();
    //cout << numInds << " individuals to be included from [" + famFile + "]." << endl;
}


void Data::readBimFile(const string &bimFile) {
    // Read bim file: recombination rate is defined between SNP i and SNP i-1
    ifstream in(bimFile.c_str());
    if (!in) throw ("Error: can not open the file [" + bimFile + "] to read.");
    //cout << "Reading PLINK BIM file from [" + bimFile + "]." << endl;
    snpInfoVec.clear();
    snpInfoMap.clear();
    string id, allele1, allele2;
    unsigned chr, physPos;
    float genPos;
    unsigned idx = 0;
    while (in >> chr >> id >> genPos >> physPos >> allele1 >> allele2) {
        SnpInfo *snp = new SnpInfo(idx++, id, allele1, allele2, chr, genPos, physPos);
        snpInfoVec.push_back(snp);
        if (snpInfoMap.insert(pair<string, SnpInfo*>(id, snp)).second == false) {
            throw ("Error: Duplicate SNP ID found: \"" + id + "\".");
        }
    }
    in.close();
    numSnps = (unsigned) snpInfoVec.size();
    //cout << numSnps << " SNPs to be included from [" + bimFile + "]." << endl;
}


void Data::center_and_scale(double* __restrict__ vec, int* __restrict__ mask, const uint N, const uint nas) {

    // Compute mean
    double mean = 0.0;
    uint nonas = N - nas;

    for (int i=0; i<N; ++i)
        mean += vec[i] * mask[i];

    mean /= nonas;
    //cout << "mean = " << mean << endl;

    // Center
    for (int i=0; i<N; ++i)
        vec[i] -= mean;

    // Compute scale
    double sqn = 0.0;
    for (int i=0; i<N; ++i) {
        if (mask[i] == 1) {
            sqn += vec[i] * vec[i];
        }
    }
    sqn = sqrt(double(nonas-1) / sqn);

    // Scale
    for (int i=0; i<N; ++i) {
        if (mask[i] == 1) {
            vec[i] *= sqn;
        } else {
            vec[i] = 0.0;
        }
    }
}


// EO: overloaded function to be used when processing sparse data
//     In such case we do not read from fam file before
// --------------------------------------------------------------
void Data::readPhenotypeFiles(const vector<string> &phenFiles, const int numberIndividuals, MatrixXd& dest) {

    const int NT = phenFiles.size();
    int ninds = -1;

    for (int i=0; i<NT; i++) {
        //cout << "reading phen " << i << ": " << phenFiles[i] << endl;
        VectorXd phen;
        VectorXi mask;
        uint nas = 0;
        readPhenotypeFileAndSetNanMask(phenFiles[i], numberIndividuals, phen, mask, nas);
        cout << "read phen file of length: " << phen.size() << " with " << nas << " NAs" << endl;

        // Make sure that all phenotypes cover the same number of individuals
        if (ninds < 0) {
            ninds = phen.size();

            phenosData.resize(NT, ninds);
            phenosNanMasks.resize(NT, ninds);
            phenosNanNum.resize(NT);
            cout << "data phenosNanNum " << phenosNanNum.size() << endl; 

            phenosData.setZero();
            phenosNanMasks.setZero();
            phenosNanNum.setZero();
        }
        assert(ninds == phen.size());
        assert(ninds == mask.size());

        center_and_scale(phen.data(), mask.data(), ninds, nas);

        for (int j=0; j<ninds; j++) {
            phenosData(i, j)   = phen(j);
            phenosNanMasks(i, j) = mask(j);
        }

        phenosNanNum(i) = nas;
    }
}


void Data::readPhenotypeFileAndSetNanMask(const string &phenFile, const int numberIndividuals, VectorXd& dest, VectorXi& mask, uint& nas) {

    numInds = numberIndividuals;
    ifstream in(phenFile.c_str());
    if (!in) throw ("Error: can not open the phenotype file [" + phenFile + "] to read.");
    //cout << "Reading phenotypes from [" + phenFile + "], and setting NAn" << endl;
    uint line = 0, nonas = 0;
    Gadget::Tokenizer colData;
    string inputStr;
    string sep(" \t");
    dest.setZero(numInds);
    mask.setZero(numInds);
    while (getline(in,inputStr)) {
        colData.getTokens(inputStr, sep);
        if (colData[1+1] != "NA") {
            dest(line) = double( atof(colData[1+1].c_str()) );
            mask(line) = 1;
            nonas += 1;
        } else {
            dest(line) = 1E30; //0./0.; //EO: generate a nan
            mask(line) = 0;
            nas += 1;
        }
        line += 1;
    }
    in.close();
    assert(nonas + nas == numInds);
    assert(line == numInds);
    //printf("nonas = %d, nas = %d\n", nonas, nas);
}


//EO: combined reading of a .phen and .cov files
//    Assume .cov and .phen to be consistent with .fam and .bed!
//--------------------------------------------------------------
void Data::readPhenCovFiles(const string &phenFile, const string covFile, const int numberIndividuals, VectorXd& dest, const int rank) {

    numInds = numberIndividuals;
    dest.setZero(numInds);

    ifstream inp(phenFile.c_str());
    if (!inp)
        throw ("Error: can not open the phenotype file [" + phenFile + "] to read.");

    ifstream inc(covFile.c_str());
    if (!inc)
        throw ("Error: can not open the covariates file [" + covFile + "] to read.");

    uint line = 0, nas = 0, nonas = 0;
    string sep(" \t");
    Gadget::Tokenizer colDataP,  colDataC;
    string            inputStrP, inputStrC;
    std::vector<double> values;
    while (getline(inp, inputStrP)) {
        getline(inc, inputStrC);
        colDataP.getTokens(inputStrP, sep);
        colDataC.getTokens(inputStrC, sep);
        bool naC = false;
        for (int i=2; i<colDataC.size(); i++) {
            if (colDataC[i] == "NA") {
                naC = true;
                break;
            }
        }
 
        if (colDataP[1+1] != "NA" && naC == false) {
            dest[nonas] = double( atof(colDataP[1+1].c_str()) );
            for (int i=2; i<colDataC.size(); i++) {
                values.push_back(std::stod(colDataC[i]));
            }
            nonas += 1;
        } else {
            if (rank == 0)
                cout << "NA(s) detected on line " << line << ": naC? " << naC << ", naP? " << colDataP[1+1] << endl;
            NAsInds.push_back(line);
            nas += 1;
        }

        line += 1;
    }
    inp.close();
    inc.close();

    assert(nonas + nas == numInds);

    assert(line == numInds);

    numFixedEffects = values.size() / (line - nas);
    cout << "numFixedEffect = " << numFixedEffects << endl;

    X = Map<const Matrix<double, Dynamic, Dynamic, RowMajor>>(values.data(), (line - nas), numFixedEffects);

    numNAs = nas;

    dest.conservativeResize(numInds-nas);
}


//(S)EO: combined reading of a .phen and .cov and .fail files
//    Assume .cov and .phen to be consistent with .fam and .bed!
//--------------------------------------------------------------
void Data::readPhenFailCovFiles(const string &phenFile, const string covFile, const string &failFile, const int numberIndividuals, VectorXd& dest, VectorXd& dfail, const int rank) {

    numInds = numberIndividuals;
    dest.setZero(numInds);
    dfail.setZero(numInds);

    ifstream inp(phenFile.c_str());
    if (!inp)
        throw ("Error: can not open the phenotype file [" + phenFile + "] to read.");

    ifstream inc(covFile.c_str());
    if (!inc)
        throw ("Error: can not open the covariates file [" + covFile + "] to read.");

    ifstream inf(failFile.c_str());
    if (!inf)
        throw ("Error: can not open the failure file [" + failFile + "] to read.");

    uint line = 0, nas = 0, nonas = 0;
    string sep(" \t");
    Gadget::Tokenizer colDataP,  colDataC, colDataF;
    string            inputStrP, inputStrC, inputStrF;
    std::vector<double> values;
    while (getline(inp, inputStrP)) {
        getline(inc, inputStrC);
        getline(inf, inputStrF);
        colDataP.getTokens(inputStrP, sep);
        colDataC.getTokens(inputStrC, sep);
        colDataF.getTokens(inputStrF, sep);

        bool naC = false;
        for (int i=2; i<colDataC.size(); i++) {
            if (colDataC[i] == "NA") {
                naC = true;
                break;
            }
        }

        if (colDataP[1+1] != "NA" && naC == false && colDataF[0] != "-9") {
            dest[nonas] = double( atof(colDataP[1+1].c_str()) );
            dfail[nonas] = double( atof(colDataF[0].c_str()) );
            for (int i=2; i<colDataC.size(); i++) {
                values.push_back(std::stod(colDataC[i]));
            }
            nonas += 1;
        } else {
            if (rank == 0)
                cout << "NA(s) detected on line " << line << ": naC? " << naC << ", naP? " << colDataP[1+1] << ", naF? " << colDataF[0]  << endl;
            NAsInds.push_back(line);
            nas += 1;
        }
        
        line += 1;
    }
    inp.close();
    inc.close();
    inf.close();
    assert(nonas + nas == numInds);

    assert(line == numInds);

    numFixedEffects = values.size() / (line - nas);

    X = Map<const Matrix<double, Dynamic, Dynamic, RowMajor>>(values.data(), (line - nas), numFixedEffects);

    numNAs = nas;

    dest.conservativeResize(numInds-nas);
    dfail.conservativeResize(numInds-nas);

}


//SEO: Function to read phenotype and failure files simultaneously (without covariates)
void Data::readPhenFailFiles(const string &phenFile, const string &failFile, const int numberIndividuals, VectorXd& dest, VectorXd& dfail, const int rank) {

    numInds = numberIndividuals;
    dest.setZero(numInds);
    dfail.setZero(numInds);

    ifstream inp(phenFile.c_str());
    if (!inp)
        throw ("Error: can not open the phenotype file [" + phenFile + "] to read.");

    ifstream inf(failFile.c_str());
    if (!inf)
        throw ("Error: can not open the failure file [" + failFile + "] to read.");

    uint line = 0, nas = 0, nonas = 0;
    string sep(" \t");
    Gadget::Tokenizer colDataP,  colDataF;
    string            inputStrP, inputStrF;
    std::vector<double> values;
    while (getline(inp, inputStrP)) {
        getline(inf, inputStrF);
        colDataP.getTokens(inputStrP, sep);
        colDataF.getTokens(inputStrF, sep);

        if (colDataP[1+1] != "NA" && colDataF[0] != "-9") {
            dest[nonas] = double( atof(colDataP[1+1].c_str()) );
            dfail[nonas] = double( atof(colDataF[0].c_str()) );
            nonas += 1;
        } else {
            if (rank == 0)
                cout << "NA(s) detected on line " << line << ", naP? " << colDataP[1+1] << ", naF? " << colDataF[0]  << endl;
            NAsInds.push_back(line);
            nas += 1;
        }

        line += 1;
    }
    inp.close();
    inf.close();
    assert(nonas + nas == numInds);

    assert(line == numInds);

    numNAs = nas;

    dest.conservativeResize(numInds-nas);
    dfail.conservativeResize(numInds-nas);
}


void Data::readPhenotypeFile(const string &phenFile, const int numberIndividuals, VectorXd& dest) {
    numInds = numberIndividuals;
    ifstream in(phenFile.c_str());
    if (!in) throw ("Error: can not open the phenotype file [" + phenFile + "] to read.");
    //cout << "Reading phenotypes from [" + phenFile + "]." << endl;
    uint line = 0, nas = 0, nonas = 0;
    Gadget::Tokenizer colData;
    string inputStr;
    string sep(" \t");
    //y.setZero(numInds);
    dest.setZero(numInds);
    while (getline(in,inputStr)) {
        colData.getTokens(inputStr, sep);
        if (colData[1+1] != "NA") {
            //y[nonas] = double( atof(colData[1+1].c_str()) );
            dest[nonas] = double( atof(colData[1+1].c_str()) );
            //if (nonas < 30) printf("read no na on line %d, nonas %d = %15.10f\n", line, nonas, y(nonas));
            nonas += 1;
        } else {
            //cout << "WARNING: found NA on line/individual " << line << endl;
            NAsInds.push_back(line);
            nas += 1;
        }
        line += 1;
    }
    in.close();
    assert(nonas + nas == numInds);

    numNAs = nas;
    //y.conservativeResize(numInds-nas);
    dest.conservativeResize(numInds-nas);
}

void Data::readPhenotypeFile(const string &phenFile) {
    // NA: missing phenotype
    ifstream in(phenFile.c_str());
    if (!in) throw ("Error: can not open the phenotype file [" + phenFile + "] to read.");
    //cout << "Reading phenotypes from [" + phenFile + "]." << endl;
    map<string, IndInfo*>::iterator it, end=indInfoMap.end();
    IndInfo *ind = NULL;
    Gadget::Tokenizer colData;
    string inputStr;
    string sep(" \t");
    string id;
    double tmp = 0.0;
    //correct loop to go through numInds
    y.setZero(numInds);
    uint line = 0, nas = 0, nonas = 0;

    while (getline(in,inputStr)) {
        colData.getTokens(inputStr, sep);
        id = colData[0] + ":" + colData[1];
        it = indInfoMap.find(id);
        // First one corresponded to mphen variable (1+1)
        if (it != end) {
            ind = it->second;
            if (colData[1+1] != "NA") {
                tmp = double(atof(colData[1+1].c_str()));
                ind->phenotype = tmp;
                y[nonas]       = tmp;
                nonas += 1;
            } else {
                //cout << "WARNING; found NA on line/individual " << line << endl;
                ind->kept = false;
                NAsInds.push_back(line);
                nas += 1;
            }
            line += 1;
        }
    }
    in.close();
    //printf("nonas = %d + nas = %d = numInds = %d\n", nonas, nas, numInds);
    assert(nonas + nas == numInds);

    numNAs = nas;
    y.conservativeResize(numInds-nas);
}


template<typename M>
M Data::readCSVFile (const string &path) {
    std::ifstream indata;
    indata.open(path);

    std::string line;

    std::vector<double> values;
    //cout << path << endl;

    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        //cout << cell  << endl;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    //cout << "rows = " << rows << " vs numInds " << numInds << endl; 
    if (rows != numInds)
        throw(" Error: covariate file has different number of individuals as BED file");

    numFixedEffects = values.size()/rows;

    return Map<const Matrix<typename M::Scalar, Dynamic, Dynamic, RowMajor>>(values.data(), rows, values.size()/rows);
}


//EO@@@
void Data::readFailureFile(const string &failureFile){
	ifstream input(failureFile);
	vector<double> tmp;
	int col;
	if(!input.is_open()){
		cout << "Error opening the file" << endl;
		return;
	}

	while(true){
		input >> col ;
		if(input.eof()) break;
		if(col == 0 or col == 1){
			tmp.push_back(col); // Only read the individuals who have a valid failure indicator (others are missing data)
		}
	}
	input.close();
	fail = Eigen::VectorXd::Map(tmp.data(), tmp.size());
}


void Data::readGroupFile(const string &groupFile) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    ifstream in(groupFile.c_str());
    if (!in) throw ("Error: can not open the group file [" + groupFile + "] to read. Use the --groupIndexFile option!");

    if (rank == 0)
        cout << "INFO   : Reading groups from [" + groupFile + "]." << endl;

    std::istream_iterator<int> start(in), end;    

    std::vector<int> numbers(start, end);

    int* ptr =(int*)&numbers[0];

    groups = Eigen::Map<Eigen::VectorXi>(ptr, numbers.size());
}


// MP: read mS (mixtures) for each group
void Data::readmSFile(const string& mSfile){

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    ifstream in(mSfile);
    if (!in) throw ("Error: can not open the mixture file [" + mSfile + "] to read. Use the --groupMixtureFile option!");

    if (rank == 0)
        cout << "INFO   : Reading mixtures from [" + mSfile + "]." << endl;

    if (!in.is_open()) throw ("Error opening the file");

    string whole_text{ istreambuf_iterator<char>(in), istreambuf_iterator<char>() };
    
    Gadget::Tokenizer strvec;
    Gadget::Tokenizer strT;
    
    strvec.getTokens(whole_text, ";"); // Size is number of groups
    strT.getTokens(strvec[0], ",");    // Size is number of non-zero mixture components

    const int n_groups = strvec.size();
    const int n_mixt_c = strT.size();

    numGroups = n_groups;

    // Add a column for the 0.0
    mS.resize(n_groups, n_mixt_c + 1);
    mS.col(0).array() = 0.0;

    for (unsigned j=0; j<n_groups; ++j) {

        strT.getTokens(strvec[j], ",");

        // Compare number of elements of each line to that of the first line
        if (strT.size() != n_mixt_c)
            throw("FATAL  : all group mixture should have the same number of components");

        for(unsigned k=0; k<strT.size(); ++k) {
            double mix = stod(strT[k]);
            if (mix <= 0.0) throw("FATAL  : mixture value can only be strictly positive");

            // Shit to col + 1, as col 0 is 0.0 by default
            mS(j, k+1) = mix;
        }
    }
}


void Data::printGroupMixtureComponents() {
    
    printf("INFO   : group mixture component data.mS has size = [%lu, %lu]\n", mS.rows(), mS.cols());
    printf("INFO   : data.mS = [");        
    for (int i=0; i<numGroups; i++) {
        if (i > 0) printf("                    ");
        for (int ii=0; ii<(int)mS.cols(); ii++) {
            printf("%6.5f", mS(i, ii));
            if (ii < (int)mS.cols() - 1)  printf(", ");
            else
                if   (i < numGroups - 1)  printf(",");
                else                      printf("];");                      
        }
        printf("\n");
    }
}

/*
 * Reads priors v0, s0 for groups from file
 * in : path to file (expected format as "v0,s0; v0,s0; ...")
 * out: void
 */
void Data::read_group_priors(const string& file){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        ifstream in(file);
        string whole_text{ istreambuf_iterator<char>(in), istreambuf_iterator<char>() };

        Gadget::Tokenizer strvec;
        Gadget::Tokenizer strT;
        // get element sizes to instantiate result vector
        strvec.getTokens(whole_text, ";");
        strT.getTokens(strvec[0], ",");
        priors = Eigen::MatrixXd(strvec.size(), strT.size());
        numGroups = strvec.size();
        //cout << "numGroups = " << numGroups << endl;
        for (unsigned j=0; j<strvec.size(); ++j) {
            strT.getTokens(strvec[j], ",");
            for (unsigned k=0; k<2; ++k) {
                priors(j, k) = stod(strT[k]);
            }
        }
    } catch (const ifstream::failure& e) {
        cout<<"Error opening the file"<< endl;
    }
    if (rank == 0) {
        cout << "Mixtures read from file" << endl;
    }
}


/*
 * Reads parameters for Dirichlet distribution from file to member variable
 * in : path to file (expected format as "x,y,z; a,b,c; ..." when k=3)
 * out: void
 */
void Data::read_dirichlet_priors(const string& file){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        ifstream in(file);
        string whole_text{ istreambuf_iterator<char>(in), istreambuf_iterator<char>() };

        Gadget::Tokenizer strvec;
        Gadget::Tokenizer strT;
        // get element sizes to instantiate result vector
        strvec.getTokens(whole_text, ";");
        strT.getTokens(strvec[0], ",");
        dPriors = Eigen::MatrixXd(strvec.size(), strT.size());
        numGroups = strvec.size();
        //cout << "numGroups = " << numGroups << endl;
        for (unsigned j=0; j<strvec.size(); ++j) {
            strT.getTokens(strvec[j], ",");
            for (unsigned k=0; k<strT.size(); ++k) {
                dPriors(j, k) = stod(strT[k]);
            }
        }
    } catch (const ifstream::failure& e) {
        cout<<"Error opening the file"<< endl;
    }
    if (rank == 0) {
        cout << "Dirichlet parameters read from file" << endl;
    }
}


/*
 * Reads priors  for sigmaE/alpha, mu, fixed effects from file
 * in : path to file (expected format as "par1,par2; par1; par1")
 * out: void
 */
void Data::read_hyperparameter_priors(const string& file){
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    try {
        ifstream in(file);
        string whole_text{ istreambuf_iterator<char>(in), istreambuf_iterator<char>() };

        Gadget::Tokenizer strvec;
        Gadget::Tokenizer strT;
        // get element sizes to instantiate result vector
        strvec.getTokens(whole_text, ";");

        sigmaEPriors = Eigen::VectorXd(2);  // We need to pass two values
        for (unsigned j=0; j<strvec.size(); ++j) {
            if(j == 0){
	        strT.getTokens(strvec[j], ",");
                for (unsigned k=0; k<2; ++k) {
                    sigmaEPriors(k) = stod(strT[k]);
                }
	    }else if(j == 1){
		strT.getTokens(strvec[j], ",");
	        muPrior = stod(strT[0]);
	    }else if(j == 2){
 	        strT.getTokens(strvec[j], ",");
                covarPrior = stod(strT[0]);
	    }
        }
    } catch (const ifstream::failure& e) {
        cout<<"Error opening the file"<< endl;
    }
    if (rank == 0) {
        cout << "Hyperparameters for intercept and error variance read from file" << endl;
    }
}



