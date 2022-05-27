#pragma once
#include <string>
#include <vector>
#include <memory>
#include <immintrin.h>
#include <mpi.h>
#include "options.hpp"
#include "distributions.hpp"
#include "utilities.hpp"


class Phenotype {

private:
    Distributions dist_m; // for shuffling the markers
    Distributions dist_d; // for sampling the distributions
    std::string filepath;
    std::string inbet_fp;
    std::string outmlma_fp;
    std::string outbet_fp;
    std::string outcpn_fp;
    std::string outcsv_fp;
    MPI_File inbet_fh;
    MPI_File outmlma_fh;
    MPI_File outbet_fh;
    MPI_File outcpn_fh;
    MPI_File outcsv_fh;
    int nonas = 0;
    int nas   = 0;
    int im4   = 0;
    const unsigned int M = 0;
    const unsigned int N = 0;
    const unsigned int G = 0;
    const unsigned int K = 0;
    std::vector<double> betas;
    std::vector<double> data;
    std::vector<unsigned char> mask4;
    std::vector<int> midx;
    std::vector<double> denom;
    std::vector<double> muk;
    std::vector<double> logl;
    std::vector<double> acum;
    std::vector<int> comp;
    std::vector<double> beta_sqn;
    std::vector<int> m0;
    std::vector<std::vector<double>> pi_est;
    std::vector<double> dirich;
    double* mave     = nullptr;
    double* msig     = nullptr;
    double* epsilon_ = nullptr;
    int*    cass     = nullptr;
    double epssum = 0.0;
    double sigmae_ = 0.0;
    std::vector<double> sigmag;
    double mu     = 0.0;
    void read_file(const Options& opt);
    void set_output_filenames(const std::string out_dir);

public:
    Phenotype(std::string fp, const Options& opt, const int N, const int M);

    Phenotype(const Phenotype& rhs);

    ~Phenotype() {
        //std::cout << "\\\\\\ calling Phenotype dtor on rank " << rank << &outbet_fh << std::endl;
        if (mave     != nullptr)  _mm_free(mave);
        if (msig     != nullptr)  _mm_free(msig);
        if (epsilon_ != nullptr)  _mm_free(epsilon_);
        if (cass     != nullptr)  _mm_free(cass);
    }

    std::string get_filepath()   const { return filepath; }
    std::string get_inbet_fp()   const { return inbet_fp; }
    std::string get_outmlma_fp() const { return outmlma_fp; }
    std::string get_outbet_fp()  const { return outbet_fp; }
    std::string get_outcpn_fp()  const { return outcpn_fp; }
    std::string get_outcsv_fp()  const { return outcsv_fp; }
    MPI_File* get_inbet_fh()   { return &inbet_fh; }
    MPI_File* get_outmlma_fh() { return &outmlma_fh; }
    MPI_File* get_outbet_fh()  { return &outbet_fh; }
    MPI_File* get_outcpn_fh()  { return &outcpn_fh; }
    MPI_File* get_outcsv_fh()  { return &outcsv_fh; }
    void print_info() const;
    std::vector<unsigned char>& get_mask4() { return mask4; }
    std::vector<int>&           get_midx()  { return midx; }
    std::vector<double>&        get_denom() { return denom; }
    std::vector<double>&        get_muk()   { return muk; }
    std::vector<double>&        get_logl()  { return logl; }
    std::vector<double>&        get_acum()  { return acum; }
    std::vector<double>&        get_betas() { return betas; }
    std::vector<int>&           get_comp()  { return comp; }
    //std::vector<std::vector<double>>&  get_pi_est() { return pi_est; }
    std::vector<std::vector<double>>*  get_pi_est() { return &pi_est; }
    int* get_cass()     { return cass; }

    int     get_im4()   const { return im4; }
    int     get_nas()   const { return nas; }
    int     get_nonas() const { return nonas; }
    double* get_mave()        { return mave; }
    double* get_msig()        { return msig; }
    double* get_epsilon()     { return epsilon_; }
    double  get_epsilon_sum() { return epssum; }
    double  get_sigmae()      { return sigmae_; }
    void    set_sigmae(const double val) { sigmae_ = val; }
    std::vector<double>* get_sigmag()    { return &sigmag; }
    double  get_mu()          { return mu; }

    void offset_epsilon(const double);
    //void update_epsilon_sum();
    void update_epsilon_sigma();

    void set_prng_m(const unsigned int);
    void set_prng_d(const unsigned int);
    void set_midx();
    void shuffle_midx(const bool mimic_hydra);
    double sample_norm_rng();
    double sample_norm_rng(const double a, const double b);
    double sample_beta_rng();
    double sample_beta_rng(const double a, const double b);
    double sample_unif_rng();
    double sample_inv_scaled_chisq_rng(const double a, const double b);
    void   sample_for_free(const int n);

    unsigned int get_random_int() { return dist_d.get_random_number(); }

    void   set_pi_est(const std::vector<std::vector<double>> val) { pi_est = val; }
    void   set_pi_est(const int group, const int k, const double val) { pi_est[group][k] = val; }
    double get_pi_est(const int group, const int k) { return pi_est[group][k]; }

    void set_comp(const int mloc, const int val) { comp[mloc] = val; }

    void print_pi_est() {
        for (int i=0; i<pi_est.size(); i++) {
            for(int j=0; j<pi_est.at(0).size(); j++) {
                printf("%d %d %20.15f\n", i, j, pi_est[i][j]);
            }
        }
    }

    //void set_sigmag(const double val) { sigmag = val; }
    void   set_mu(const double val) { mu = val; }

    void   set_marker_acum(const int idx, const double val) { acum[idx] = val; }
    double get_marker_acum(const int idx) { return acum[idx]; }

    void    set_marker_beta(const int idx, const double val) { betas[idx] = val; }
    double  get_marker_beta(const int idx) { return betas[idx]; }

    int    get_marker_local_index(const int shuff_idx);
    double get_marker_ave(const int idx) { return mave[idx]; }
    double get_marker_sig(const int idx) { return msig[idx]; }

    void   update_epsilon(const double* dbeta, const unsigned char* bed);
    double epsilon_sumsqr();
    double epsilon_sum();

    void reset_beta_sqn_to_zero() {
        std::fill(beta_sqn.begin(), beta_sqn.end(), 0.0);
    }
    void increment_beta_sqn(const int group, const double val);
    double get_beta_sqn_for_group(const int group) { return beta_sqn.at(group); }
    std::vector<double>* get_beta_sqn ()     { return &beta_sqn; }

    double get_sigmag_sum() {
        double sum = 0.0;
        for (int i=0; i<G; i++)
            sum += get_sigmag_for_group(i);
        return sum;
    }

    void set_beta_sqn(const double* in) {
        for (int i=0; i<G; i++)
            beta_sqn[i] = in[i];
    }

    void reset_cass() {
        for (int i=0; i<G*K; i++)
            cass[i] = 0;
    }

    void set_cass(const int* in) {
        for (int i=0; i<G; i++) {
            for (int j=0; j<K; j++) {
                cass[i * K + j] = in[i * K + j];
            }
        }
    }

    int get_cass_for_group(const int g, const int k) {
        return cass[g * K + k];
    }
    int get_cass_sum_for_group(const int g) {
        int cass_sum = 0;
        for (int i=0; i<K; i++)
            cass_sum += get_cass_for_group(g, i);
        return cass_sum;
    }

    void increment_cass(const int g, const int k, const int val) {
        cass[g * K + k] += val;
    }

    void print_cass();
    void print_cass(const std::vector<int>& mtotgrp);


    void reset_m0() { std::fill(m0.begin(), m0.end(), 0); }
    void set_m0_for_group(const int group, const int val) { m0.at(group) = val; }
    int  get_m0_for_group(const int group) { return m0.at(group); }
    int  get_m0_sum();
    void   set_sigmag_for_group(const int group, const double val) { sigmag.at(group) = val; }
    double get_sigmag_for_group(const int group) { return sigmag.at(group); }

    void update_pi_est_dirichlet(const int group);

    void open_prediction_files();
    void close_prediction_files();
    void delete_output_prediction_files();

    void open_output_files();
    void close_output_files();
    void delete_output_files();

    void set_prediction_filenames(const std::string out_dir);

    void set_nas_to_zero(double* y, const int N);

    //EO: trick for --prediction, assumed epsilon unchanged!!
    void get_centered_and_scaled_y(double* y_k) {
        for (int i=0; i<N; i++) {
            //if (i<10)
            //    printf("epsilon_[%d] = %20.15f\n", i, epsilon_[i]);
            y_k[i] = epsilon_[i];
        }
        /*
        double sum = 0.0;
        for (int j=0; j<im4; j++) {
            for (int k=0; k<4; k++) {
                sum += data[j*4 + k] * mask4[j*4 + k];
            }
        }
        double ave = sum / double(nonas);
        double sig = 0.0;
        for (int j=0; j<im4; j++) {
            for (int k=0; k<4; k++) {
                sig += (data[j*4 + k] - ave) * (data[j*4 + k] - ave) * mask4[j*4 + k];
            }
        }
        sig /= (nonas - 1);
        sig  = sqrt(sig);
        printf("y sum = %20.15f, N = %d, nonas = %d, ave = %20.15f, sig = %20.15f\n", sum, N, nonas, ave, sig);

        for (int j=0; j<im4; j++) {
            for (int k=0; k<4; k++) {
                y_k[j*4 + k] = (data[j*4 + k] - ave) / sig * mask4[j*4 + k];
            }
        }
        */
    }

    friend class PhenMgr;
};


class PhenMgr {

public:
    PhenMgr() = default;
    PhenMgr(const Options& opt, const int N, const int M) {
        //std::cout << "+++Calling PhenMgr ctor" << std::endl;
        read_phen_files(opt, N, M);
        //std::cout << "---PhenMgr ctor done" << std::endl;
    }
    //~PhenMgr() {
    //    std::cout << "/!\\ Calling PhenMgr dtor" << std::endl;
    //}
    void print_info();
    void compute_markers_statistics(const unsigned char* bed_data, const int N, const int M, const int mbytes);
    std::vector<Phenotype>& get_phens() { return phens; }
    void display_markers_statistics(const int n);
    void read_phen_files(const Options& opt, const int N, const int M);

private:
    std::vector<Phenotype> phens;
};

