#ifndef HYDRA_DATA_H_
#define HYDRA_DATA_H_

#include <Eigen/Eigen>
#include <mpi.h>
#include "utils.hpp"

using namespace std;
using namespace Eigen;


class SnpInfo {

public:
    const string ID;
    const string a1; // the referece allele
    const string a2; // the coded allele
    const int chrom;
    const float genPos;
    const int physPos;

    int index;
    int window;
    int windStart;  // for window surrounding the SNP
    int windSize;   // for window surrounding the SNP
    float af;       // allele frequency
    bool included;  // flag for inclusion in panel
    bool isQTL;     // for simulation

    SnpInfo(const int idx, const string &id, const string &allele1, const string &allele2,
            const int chr, const float gpos, const int ppos)
    : ID(id), index(idx), a1(allele1), a2(allele2), chrom(chr), genPos(gpos), physPos(ppos) {
        window = 0;
        windStart = -1;
        windSize  = 0;
        af = -1;
        included = true;
        isQTL = false;
    };

    void resetWindow(void) {windStart = -1; windSize = 0;};
};


class IndInfo {

public:
    const string famID;
    const string indID;
    const string catID;    // catenated family and individual ID
    const string fatherID;
    const string motherID;
    const int famFileOrder; // original fam file order
    const int sex;  // 1: male, 2: female

    int index;
    bool kept;

    float phenotype;

    VectorXf covariates;  // covariates for fixed effects

    IndInfo(const int idx, const string &fid, const string &pid, const string &dad, const string &mom, const int sex)
    : famID(fid), indID(pid), catID(fid+":"+pid), fatherID(dad), motherID(mom), index(idx), famFileOrder(idx), sex(sex) {
        phenotype = -9;
        kept = true;
    }
};


// An entry for the index to the compressed preprocessed bed file
struct IndexEntry {
    long pos;
    long size;
};


class Data {

public:
    Data();

    // Original data
    MatrixXd X;              // coefficient matrix for fixed effects
    MatrixXd Z;
    VectorXf D;              // 2pqn

    VectorXd y;              // phenotypes
    MatrixXd phenosData;     // multiple phenotypes
    MatrixXi phenosNanMasks; // masks for NAs in phenotypes phenos
    VectorXi phenosNanNum;   // number of NAs in phenotypes phenos

    // marion : vector for annotation file and matrix for mS
    VectorXi groups;     	 // groups
    MatrixXd mS;			 // mixtures in groups
    MatrixXd priors;         // group priors v0, s0
    MatrixXd dPriors;        // group priors of dirichlet distribution
    VectorXd sigmaEPriors;   // Priors for sigmaE or alpha
    double muPrior = 100;    // Prior for mu
    double covarPrior = 100; // Prior for the covariates

    VectorXd fail;           // Failure indicator

    vector<SnpInfo*> snpInfoVec;
    vector<IndInfo*> indInfoVec;

    map<string, SnpInfo*> snpInfoMap;
    map<string, IndInfo*> indInfoMap;
    
    unsigned numFixedEffects = 0;

    unsigned numSnps = 0;
    unsigned numInds = 0;
    unsigned numNAs  = 0;

    vector<uint> NAsInds;
    vector<int>  blocksStarts;
    vector<int>  blocksEnds;
    uint         numBlocks = 0;


    uint set_Ntot(const int rank, const Options opt);

    uint set_Mtot(const int rank, Options opt);

    void center_and_scale(double* __restrict__ vec, int* __restrict__ mask, const uint N, const uint nas);

    void print_restart_banner(const string mcmcOut, const uint iteration_restart, const uint iteration_start);

    
    void read_mcmc_output_idx_file(const string mcmcOut, const string ext, const uint length, const uint iteration_restart, const string bayesType,
                                   std::vector<int>& markerI);
    
    void read_mcmc_output_gam_file(const string mcmcOut, const int gamma_length, const uint iteration_restart,
                                   VectorXd& gamma);

    void read_mcmc_output_eps_file(const string mcmcOut, const uint Ntot, const uint iteration_restart,
                                   VectorXd&    epsilon);

    void read_mcmc_output_cpn_file(const string mcmcOut, const uint Mtot,
                                   const uint   iteration_restart, const uint first_saved_it_restart, const int thin,
                                   const int*   MrankS,  const int* MrankL, const bool use_xfiles,
                                   VectorXi&    components);

    void read_mcmc_output_bet_file(const string mcmcOut, const uint Mtot,
                                   const uint   iteration_restart, const uint first_saved_it_restart, const int thin,
                                   const int*   MrankS,  const int* MrankL, const bool use_xfiles,
                                   VectorXd&    Beta);

    // ASCII
    void read_mcmc_output_csv_file(const string mcmcOut,
                                   const uint optThin, const uint optSave,
                                   const int K, VectorXd& sigmaG, double& sigmaE, MatrixXd& pi,
                                   uint& iteration_to_restart_from,
                                   uint& first_thinned_iteration,
                                   uint& first_saved_iteration);
    //BIN
    void read_mcmc_output_out_file(const string mcmcOut,
                                   const uint optThin, const uint optSave,
                                   const int K, VectorXd& sigmaG, double& sigmaE, MatrixXd& pi,
                                   uint& iteration_to_restart_from,
                                   uint& first_thinned_iteration,
                                   uint& first_saved_iteration);

    // Three functions tailored for bW output. Consider using same format in bR
    void read_mcmc_output_csv_file_bW(const string mcmcOut, const uint optThin, const uint optSave, const int K, double& mu,
                                        VectorXd& sigmaG, double& sigmaE, MatrixXd& pi,
                                        uint& iteration_to_restart_from, uint& first_thinned_iteration, uint& first_saved_iteration);

    void read_mcmc_output_gam_file_bW(const string mcmcOut, const uint optSave, const int gamma_length,
                                     VectorXd& gamma);

    void read_mcmc_output_mus_file(const string mcmcOut,
                                   const uint  iteration_restart, const uint first_saved_it_restart, const int thin,
                                   double& mu);

    void load_data_from_bed_file(const string bedfp, const uint Ntot, const int M,
                                 const int    rank,  const int start,
                                 size_t* N1S, size_t* N1L, uint*& I1,
                                 size_t* N2S, size_t* N2L, uint*& I2,
                                 size_t* NMS, size_t* NML, uint*& IM,
                                 size_t& taskBytes);


    void load_data_from_sparse_files(const int rank, const int nranks, const int M,
                                     const int* MrankS, const int* MrankL,
                                     const string sparseOut,
                                     size_t* N1S, size_t* N1L, uint*& I1,
                                     size_t* N2S, size_t* N2L, uint*& I2,
                                     size_t* NMS, size_t* NML, uint*& IM,
                                     size_t& taskBytes);


    void get_bed_marker_from_sparse(char* bed,
                                    const int Ntot,
                                    const size_t S1, const size_t L1, const uint* I1,
                                    const size_t S2, const size_t L2, const uint* I2,
                                    const size_t SM, const size_t LM, const uint* IM) const;

 
    void load_data_from_mixed_representations(const string bedfp,         const string sparseOut,
                                              const int    rank,          const int nranks,
                                              const int    Ntot,          const int M,
                                              const int*   MrankS,        const int* MrankL,
                                              size_t* N1S,  size_t* N1L,  uint*& I1,
                                              size_t* N2S,  size_t* N2L,  uint*& I2,
                                              size_t* NMS,  size_t* NML,  uint*& IM,
                                              const double threshold_fnz, bool* USEBED,
                                              size_t& taskBytes);
    
    void sparse_data_correct_for_missing_phenotype(const size_t* NS, size_t* NL, uint* I, const int M, const bool* USEBED);

    void sparse_data_get_sizes_from_raw(const char* rawdata,
                                        const uint  NC,
                                        const uint  NB,
                                        const uint  NA,
                                        size_t& N1,    size_t& N2,    size_t& NM) const;

    void sparse_data_fill_indices(const char* rawdata,
                                  const uint  NC,
                                  const uint  NB,
                                  const uint  NA,
                                  size_t* N1S, size_t* N1L, uint* I1,
                                  size_t* N2S, size_t* N2L, uint* I2,
                                  size_t* NMS, size_t* NML, uint* IM) const;

    size_t get_number_of_elements_from_sparse_files(const std::string basename, const std::string id, const int* MrankS, const int* MrankL,
                                                    size_t* S, size_t* L);
    
    void read_sparse_data_file(const std::string filename, const size_t N, const size_t OFF, const int NREAD, uint* out);



    // MPI_File_read_at_all handling count argument larger than INT_MAX
    //
    template <typename T>
    void mpi_file_read_at_all(const size_t N, const MPI_Offset offset, const MPI_File fh, const MPI_Datatype MPI_DT, const int NREADS, T buffer, size_t &bytes) const {

        int rank, dtsize;
        MPI_Status status;
        
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Type_size(MPI_DT, &dtsize);
        //printf("dtsize = %d vs %lu\n", dtsize, sizeof(buffer[0]));
        //fflush(stdout);
        assert(dtsize == sizeof(buffer[0]));
        
        if (NREADS == 0) return;
        
        int SPLIT_ON = check_int_overflow(size_t(ceil(double(N)/double(NREADS))), __LINE__, __FILE__);

        int count = SPLIT_ON;
        
        double totime = 0.0;
        bytes = 0;
        
        for (uint i=0; i<NREADS; ++i) {
            
            double t1 = -mysecond();

            const size_t iim = size_t(i) * size_t(SPLIT_ON);

            // Last iteration takes only the leftover
            if (i == NREADS-1) count = check_int_overflow(N - iim, __LINE__, __FILE__);
            
            //printf("read %d with count = %d x %lu = %lu Bytes to read\n", i, count, sizeof(buffer[0]), sizeof(buffer[0]) * size_t(count));
            //fflush(stdout);
            
            //check_mpi(MPI_File_read_at_all(fh, offset + iim * size_t(dtsize), &buffer[iim], count, MPI_DT, &status), __LINE__, __FILE__);
            check_mpi(MPI_File_read_at(fh, offset + iim * size_t(dtsize), &buffer[iim], count, MPI_DT, &status), __LINE__, __FILE__);
            t1 += mysecond();
            totime += t1;
            bytes += size_t(count) * size_t(dtsize);
            if (rank % 10 == 0) {
                printf("INFO   : rank %3d cumulated read time at %2d/%2d: %7.3f sec, avg time %7.3f, BW = %7.3f GB/s\n",
                       rank, i+1, NREADS, totime, totime / (i + 1), double(bytes) / totime / 1E9 );
                fflush(stdout);
            }

            //MPI_Barrier(MPI_COMM_WORLD);
        }

        //MPI_Barrier(MPI_COMM_WORLD);
    }


    // MPI_File_write_at_all handling count argument larger than INT_MAX
    //
    template <typename T>
    void mpi_file_write_at_all(const size_t N, MPI_Offset offset, MPI_File fh, MPI_Datatype MPI_DT, const int NWRITES, T buffer) const {

        int rank, dtsize;
        MPI_Status status;

        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Type_size(MPI_DT, &dtsize);
        assert(dtsize == sizeof(buffer[0]));

        if (NWRITES == 0) return;

        int SPLIT_ON = check_int_overflow(size_t(ceil(double(N)/double(NWRITES))), __LINE__, __FILE__);
        int count = SPLIT_ON;

        for (uint i=0; i<NWRITES; ++i) {

            const size_t iim = size_t(i) * size_t(SPLIT_ON);

            // Last iteration takes only the leftover
            if (i == NWRITES-1) count = check_int_overflow(N - iim, __LINE__, __FILE__);

            check_mpi(MPI_File_write_at_all(fh, offset + iim * size_t(dtsize), &buffer[iim], count, MPI_DT, &status), __LINE__, __FILE__);
        }
    }

    void preprocess_data(const char* data, const uint NC, const uint NB, double* ppdata, const int rank);

    //EO to read definitions of blocks of markers to process
    void readMarkerBlocksFile(const string &markerBlocksFile);

    void readFamFile(const string &famFile);

    void readBimFile(const string &bimFile);

    void readPhenotypeFile(const string &phenFile);
    void readPhenotypeFile(const string &phenFile, const int numberIndividuals, VectorXd& dest);

    void readPhenotypeFileAndSetNanMask(const string &phenFile, const int numberIndividuals, VectorXd& phen, VectorXi& mask, uint& nas);

    void readPhenotypeFiles(const vector<string> &phenFile, const int numberIndividuals, MatrixXd& dest);

    void readPhenCovFiles(const string &phenFile, const string covFile, const int numberIndividuals, VectorXd& dest, const int rank);


    // Functions to read for bayesW
    //
    void readPhenFailCovFiles(const string &phenFile, const string covFile, const string &failFile, const int numberIndividuals, VectorXd& dest, VectorXd& dfail, const int rank);

    void readPhenFailFiles(const string &phenFile, const string &failFile, const int numberIndividuals, VectorXd& dest, VectorXd& dfail, const int rank);

    template<typename M>
    M readCSVFile(const string &covariateFile);

    // marion : annotation variables
    unsigned numGroups = 1;	// number of annotations
    void readGroupFile(const string &groupFile);
    void readmSFile(const string& mSfile);
    void printGroupMixtureComponents();
    
    // bW var
    void readFailureFile(const string &failureFile);
    
    // prior reading files
    void read_group_priors(const string& file);
    void read_dirichlet_priors(const string& file);
    void read_hyperparameter_priors(const string& file);
};

#endif /* data_hpp */
