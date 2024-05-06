#pragma once
#include <vector> 
#include <tuple>
#include "data.hpp"
#include "options.hpp"

class vamp{

private:
    int N, M, Mt, C, max_iter, rank, nranks;
    double gam1, gam2, gam_before, eta1, eta2;  // linear model precisions
    std::vector<double> gam1s, gam2s, R2trains;
    double tau1, tau2;                          // probit model precisions
    double alpha1, alpha2;                      // Onsager corrections
    double rho;                          // damping factor
    double gamw;                                // linear model noise precision 

    std::vector<double> x1_hat, x2_hat, true_signal;
    std::vector<double> z1_hat, z2_hat;
    std::vector<double> y;                              // phenotype vector
    std::vector<double> z1;                             // z1 = A * x1_hat 
    std::vector<double> r1, r2, r2_prev;
    std::vector<double> p1, p2;
    std::vector<double> cov_eff;                        // covariates in a probit model
    std::vector<double> mu_CG_last;                     // last LMMSE estimate
    

    std::vector<double> probs, probs_before;
    std::vector<double> vars, vars_before;

    double gamma_min = 1e-11;
    double gamma_max = 1e11;
    double probit_var; // hardcoded
    int EM_max_iter; // = 1e5;
    double EM_err_thr; // = 1e-4;
    int CG_max_iter; // = 10;
    int auto_var_max_iter  = 5;// = 50;
    int calc_state_evo = 0;
    int learn_vars;
    int init_est;
    long unsigned int seed;
    double damp_max = 1;
    double damp_min = 0.05;
    double stop_criteria_thr; // = 1e-5;
    double gamma_damp;

    std::string model;
    std::string out_dir;
    std::string out_name;
    std::vector<double> bern_vec;
    std::vector<double> invQ_bern_vec;

    int store_pvals = 1;                                // if .bim file is present we also perform LOCO p-value estimation
    double total_comp_time=0;
    int reverse = 1;
    int use_lmmse_damp = 0;
    int use_freeze=0;

    // cross-validation parameters
    int SBglob, LBglob, redglob;
    int use_cross_val = 0, SB_cross;

    double deltaH;

    // restart variables
    double gam1_init;
    double gamw_init;
    std::string r1_init_file;
    std::string x1_hat_init_file;

    std::string estimate_file;
    std::string freeze_index_file;

public:

    //******************
    //  CONSTRUCTORS 
    //******************
    vamp(int N, int M,  int Mt, double gam1, double gamw, int max_iter, double rho, std::vector<double> vars,  std::vector<double> probs, std::vector<double> true_signal, int rank, std::string out_dir, std::string out_name, std::string model, Options opt = Options());
    vamp(int M, double gam1, double gamw, std::vector<double> true_signal, int rank, Options opt);


    //**********************
    // INFERENCE PROCEDURES
    //**********************
    std::vector<double> infere(data* dataset);
    std::vector<double> infere_linear(data* dataset);
    std::vector<double> infere_bin_class(data* dataset);

    std::vector<double> infere_robust(data* dataset);

    //std::vector<double> predict(std::vector<double> est, data* dataset);


    //********************************************
    // DENOISING PROCEDURES & ONSAGER CALCULATION
    //********************************************
    double g1(double x, double gam1);
    double g1_bin_class(double p, double tau1, double y, double m_cov);
    double g1d(double x, double gam1);
    double g1d_bin_class(double p, double tau1, double y, double m_cov);
    double g2d_onsager(double gam2, double tau, data* dataset);
    double g2d_onsagerAAT(double gam2, double tau, data* dataset);

    double g1_Huber(double p1, double tau1, double deltaH, double y);
    double g1d_Huber(double p1, double tau1, double deltaH, double y);
    double g1d_Huber_der(double p1, double tau1, double deltaH, double y);


    //************************
    // HYPERPARAMETERS UPDATE
    //************************
    void updatePrior(int verbose);
    void updateNoisePrec(data* dataset);
    void updateNoisePrecAAT(data* dataset);

    double Huber_loss(double z, double deltaH, double y);
    double E_MC_eval_ind(double p1, double tau1, double deltaH, double y, int num_MC_steps);
    double E_MC_eval(std::vector<double> p1, double tau1, double deltaH, std::vector<double> y, int num_MC_steps);
    double M_deltaH_update(std::vector<double> p1, double tau1, double deltaH, std::vector<double> y, int num_MC_steps, std::vector<double> grid);
    double EM_deltaH(std::vector<double> p1, double tau1, double deltaH, std::vector<double> y, int num_MC_steps, std::vector<double> grid, int num_EM_steps);

    std::vector<double> lmmse_mult(std::vector<double> v, double tau, data* dataset, int red = 0);
    std::vector<double> lmmse_multAAT(std::vector<double> u, double tau, data* dataset);
    std::vector<double> lmmse_denoiserAAT(std::vector<double> r, std::vector<double> mu_CG_AAT_last, data* dataset);

    std::vector<double> CG_solverAAT(std::vector<double> v, std::vector<double> mu_start, double tau, int save, data* dataset);
    std::vector<double> precondCG_solver(std::vector<double> v, double tau, int denoiser, data* dataset, int red = 0);
    std::vector<double> precondCG_solver(std::vector<double> v, std::vector<double> mu_start, double tau, int denoiser, data* dataset, int red = 0);    

    void err_measures(data * dataset, int ind);
    void probit_err_measures(data *dataset, int sync, std::vector<double> true_signal, std::vector<double> est, std::string var_name);
    void robust_err_measures(data *dataset, int sync, std::vector<double> true_signal, std::vector<double> est, std::string var_name);

    std::tuple<double, double, double> state_evo(int ind, double gam_prev, double gam_before, std::vector<double> probs_before, std::vector<double> vars_before, data* dataset);

    double probit_var_EM_deriv(double v, std::vector<double> z, std::vector<double> y); 
    double expe_probit_var_EM_deriv(double v, double eta, std::vector<double> z, std::vector<double> y);
    double update_probit_var(double v, double eta, std::vector<double> z_hat, std::vector<double> y);
    std::vector<double> grad_cov(std::vector<double> y, std::vector<double> gg, double probit_var, std::vector< std::vector<double> > Z, std::vector<double> eta);
    double mlogL_probit(std::vector<double> y, std::vector<double> gg, double probit_var, std::vector< std::vector<double> > Z, std::vector<double> eta);
    std::vector<double> grad_desc_step_cov(std::vector<double> y, std::vector<double> gg, double probit_var, std::vector< std::vector<double> > Z, std::vector<double> eta, double* grad_norm);
    std::vector<double> grad_desc_cov(std::vector<double> y, std::vector<double> gg, double probit_var, std::vector< std::vector<double> > Z, std::vector<double> eta);
    std::vector<double> Newton_method_cov(std::vector<double> y, std::vector<double> gg, std::vector< std::vector<double> > Z, std::vector<double> eta);


    void set_SBglob(int SB) { SBglob = SB; }
    void set_LBglob(int LB) { LBglob = LB; }
    void set_gam2 (double gam) { gam2 = gam; }
    std::vector<double> get_cov_eff() const {return cov_eff;}
    
   };
