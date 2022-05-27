#ifndef options_hpp
#define options_hpp

#include <iostream>
#include <sstream>
#include <fstream>
#include <stdio.h>
#include <cstring>
#include <string>
#include <limits.h>
#include <boost/format.hpp>
#include "gadgets.hpp"
#include <Eigen/Eigen>

using namespace std;
using namespace boost;

const unsigned Megabase = 1e6;

class Options {
public:

    unsigned shuffleMarkers      = 1;
    string   groupIndexFile      = "";
    string   groupMixtureFile    = "";
    bool     restart             = false;
    bool     useXfilesInRestart  = true;
    bool     sparseSync          = false;
    bool     bedSync             = false;
    unsigned syncRate            = 1;
    bool     bedToSparse         = false;
    bool     readFromBedFile     = false;
    bool     readFromSparseFiles = false;
    bool     mixedRepresentation = false;
    unsigned blocksPerRank       = 1;     //EO: for bed -> sparse conversion, to split blocks if too large
    bool     checkRam            = false;
    unsigned checkRamTasks       = 0;
    unsigned checkRamTpn         = 0;
    unsigned numberMarkers       = 0;
    unsigned numberIndividuals   = 0;
    unsigned chainLength;
    unsigned burnin;
    unsigned seed;
    unsigned thin;                       // save every this th sampled value in MCMC
    unsigned save;                       // sampling rate of the epsilon vector
    vector<double> S;                    // variance components

    unsigned verbosity           = 0;
 
    //marion : include annotation variables
    unsigned int numGroups;
    Eigen::MatrixXd mS;
    string groupFile;
    string mSfile;
    string failureFile;
    string bayesW_version;
    string quad_points;
    string priorsFile;
    string dPriorsFile;
    string hypPriorsFile;

    string title;
    string bayesType;

    string phenotypeFile;
    bool   multi_phen = false;
    vector<string> phenotypeFiles;
    bool   interleave = false;

    string markerBlocksFile;
    string bedFile;
    string mcmcOutDir  = "";
    string mcmcOutNam  = "default_output_name";
    string mcmcOut     = "default_output_name";
    string sparseDir   = "";
    string sparseBsn   = "";
    string optionFile;
    string covariatesFile;         // for extra covariates.
    bool   covariates  = false;    // for extra covatiates.
    bool   compress    = false;
    bool   deltaUpdate = true;     // Use the delta epsilon to pass the message in mpi

    // Use BED representation over SPARSE if fraction of non-zero elements (fnz)
    // is greater than this threshold:
    double threshold_fnz = 0.06;

    // FH related
    double betaA = 0.0;
    double betaB = 0.0;
    double tau0  = 1.0;  // global hyperparameter
    double s02c  = 1.0;  // scale regularising slab
    //EO: should be of type int. const as well? 
    double v0c   = 3;    // degrees of freedom regularising slab
    double v0L   = 3;    // degrees of freedom local parameters
    double v0t   = 3;    // degrees of freedom global paramaeters


    string options_s;

    Options(){
        chainLength      = 10000;
        burnin           = 5000;
        seed             = static_cast<unsigned int>(std::time(0));
        thin             = 5;
        save             = 10;
        S.resize(3);
        S[0]             = 0.01;
        S[1]             = 0.001;
        S[2]             = 0.0001;
        title            = "brr";
        bayesType        = "C";
        phenotypeFile    = "";
        markerBlocksFile = "";
        bedFile          = "";
        sparseDir        = "";
        sparseBsn        = "";
        optionFile       = "";
        numGroups		 = 1;
        groupFile        = "";
        priorsFile       = "";
        dPriorsFile      = "";
        mSfile           = "";
        betaA            = 1.0;
        betaB            = 1.0;
    }

    void inputOptions(const int argc, const char* argv[]);

    void printBanner(void);

    void printProcessingOptions(void);

    std::string get_sparse_output_filebase(const int rank) const;


private:
    void readFile(const string &file);
    
    void makeTitle(void);

};

#endif /* options_hpp */
