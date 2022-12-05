#pragma once
#include <string>
#include <vector>
#include <memory>
#include <immintrin.h> // contains definition of _mm_malloc


// class data contains genotypic and phenotypic data and auxiliary functions related
// to the loading of genotypic and phenotypic data as well as computing marker statistics 
// and efficient dot product between a marker and a phenotype vector.

class data {

private:

    int Mtotal;     // total number of markers
    int N;          // number of individuals 
    int M;          // number of markers attributed to the node
    int S;          // marker starting index 
    int rank;       // rank of MPI process

    std::string phenfp;       // filepath to phenotypes
    std::string bedfp;          // filepath to .bed file
    std::vector<double> phen_data;   // vector of phenotype data

    int nonas;
    int nas;
    int m4;
    int im4 = 0;
    int perm = 0;
    std::vector<int> perm_idxs;
    std::vector < double* > p_luts;
    std::vector < double* > p_dotp_luts;
    std::vector<unsigned char> mask4;
    double* mave     = nullptr;
    double* msig     = nullptr;
    unsigned char* bed_data = nullptr;
    size_t mbytes;
    double sigma_max = 1e8;
    double intercept = 0;
    double scale = 1; 
    int normal_data;
    std::vector<double> xL, xR;

public:

    void read_phen();   // reading phenotype file
    std::vector<double> get_phen(){ return phen_data; };
    // int get_phen_size(){ return phen_data.size(); };
    double get_intercept() { return intercept; };
    double get_scale() { return scale; };
    double get_sigma_max() { return sigma_max; };
    size_t get_mbytes(){ return mbytes; };
    void set_phen( std::vector<double> new_data ) { phen_data = new_data; };
    std::vector<unsigned char>& get_mask4() { return mask4; }
    double* get_mave()        { return mave; }
    double* get_msig()        { return msig; }
    int     get_im4()   const { return im4; }
    int     get_nonas()       { return nonas; }
    int     get_S()     const { return S; }
    int     get_normal_data()   const {return normal_data; }

    // constructor and destructor for class data
    data(std::string fp, std::string bedfp, const int N, const int M, const int Mt, const int S, const int normal, const int rank);
    // data(std::string fp, std::string bedfp, const int N, const int M, const int Mt, const int S, const int rank, const int perm);
    ~data() {
        if (mave     != nullptr)  _mm_free(mave);
        if (msig     != nullptr)  _mm_free(msig);
    }
    void compute_markers_statistics();

    double dot_product(const int mloc, double* __restrict__ phen, const double mu, const double sigma_inv, int normal);

    void read_genotype_data();

    std::vector<double> Ax(double* __restrict__ phen, int normal);
    
    std::vector<double> ATx(double* __restrict__ phen, int normal);

    //std::vector < double* > perm_luts();

    //std::vector < double* > perm_dotp_luts();

    void SinkhornKnopp(std::vector<double> &xL, std::vector<double> &xR, double err_thr, int SK_max_iter);

    //double generate_mixture_gaussians(int K_grp, std::vector<double> eta, std::vector<double> pi);

    //std::vector<double> simulate(int M, std::vector<double> eta, std::vector<double> pi);

    //void store_vec_to_file(std::string filepath, std::vector<double> vec);

    //double noise_prec_calc( double SNR, std::vector<double> vars, std::vector<double> probs );

};


