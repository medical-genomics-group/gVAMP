#pragma once
#include <vector> 
#include <tuple>
#include "data.hpp"
#include "options.hpp"

class vamp{

private:
    int N, M, Mt, max_iter, rank, nranks;
    std::vector<double> x1_hat, x2_hat, true_signal, z1_hat, z2_hat;
    std::vector<double> y;
    std::vector<double> obj_fun_vals = std::vector<double> (1, std::numeric_limits<double>::min());
    std::vector<double> z1;
    std::vector<double> r1, r2;
    std::vector<double> r2_prev;
    std::vector<double> p1, p2;
    std::vector<double> p_CG_last, mu_CG_last;
    double gam1, gam2, eta1, eta2, rho, gamw, gam_before, tau1, tau2, alpha1, alpha2;
    std::vector<double> probs, probs_before;
    std::vector<double> vars, vars_before;
    double gamma_min = 1e-11;
    double gamma_max = 1e11;
    double probit_var = 1e-2; // hardcoded
    int EM_max_iter; // = 1e5;
    double EM_err_thr; // = 1e-4;
    int CG_max_iter; // = 10;
    int use_adap_damp = 0;
    int calc_state_evo = 0;
    int numb_adap_damp_hist = 3;
    int poly_deg_apr_vamp_obj = 7;
    double damp_inc_fact = 1.10;
    double damp_dec_fact = 0.90;
    double damp_max = 1;
    double damp_min = 0.05;
    double stop_criteria_thr; // = 1e-5;
    std::string model;
    std::string out_dir;
    std::string out_name;
    std::vector<double> bern_vec;
    std::vector<double> invQ_bern_vec;
    int normal = 1;
    int store_pvals = 1;
    double largest_sing_val2;
    double total_comp_time=0;
    int reverse = 1;
    int use_lmmse_damp = 0;

public:

    vamp(int N, int M,  int Mt, double gam1, double gamw, int max_iter, double rho, std::vector<double> vars,  std::vector<double> probs, std::vector<double> true_signal, int rank, std::string out_dir, std::string out_name, std::string model);
    vamp(int M, double gam1, double gamw, std::vector<double> true_signal, int rank, Options opt);
    std::vector<double> infere(data* dataset);
    std::vector<double> infere_linear(data* dataset);
    std::vector<double> infere_bin_class(data* dataset);
    //std::vector<double> predict(std::vector<double> est, data* dataset);
    double g1(double x, double gam1);
    double g1_bin_class(double p, double tau1, double y);
    double g1d(double x, double gam1);
    double g1d_bin_class(double p, double tau1, double y);
    double g2d_onsager(double gam2, double tau, data* dataset);
    double g2d_onsagerAAT(double gam2, double tau, data* dataset);
    void updatePrior();
    void updateNoisePrec(data* dataset);
    void updateNoisePrecAAT(data* dataset);
    std::vector<double> lmmse_mult(std::vector<double> v, double tau, data* dataset);
    std::vector<double> lmmse_multAAT(std::vector<double> u, double tau, data* dataset);
    std::vector<double> lmmse_denoiserAAT(std::vector<double> r, std::vector<double> mu_CG_AAT_last, data* dataset);
    std::vector<double> CG_solver(std::vector<double> v, double tau, data* dataset);
    std::vector<double> CG_solver(std::vector<double> v, std::vector<double> mu_start, std::vector<double> p_start, double tau, data* dataset);
    std::vector<double> CG_solverAAT(std::vector<double> v, std::vector<double> mu_start, double tau, int save, data* dataset);
    double vamp_obj_func(double eta, double gam, std::vector<double> invQu, std::vector<double> bernu, std::vector<double> vars, std::vector<double> pi, data* dataset);
    void err_measures(data * dataset, int ind);
    std::tuple<double, double, double> state_evo(int ind, double gam_prev, double gam_before, std::vector<double> probs_before, std::vector<double> vars_before, data* dataset);
    double probit_var_EM_deriv(double v, std::vector<double> y);
    double update_probit_var(double v, std::vector<double> y);
    std::vector<double> precondCG_solver(std::vector<double> v, double tau, int save, data* dataset);
    std::vector<double> precondCG_solver(std::vector<double> v, std::vector<double> mu_start, double tau, int save, data* dataset);    
};