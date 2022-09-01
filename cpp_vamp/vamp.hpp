#pragma once
#include <vector> 
#include <tuple>
#include "data.hpp"
#include "options.hpp"

class vamp{

private:
    int N, M, Mt, max_iter, rank, nranks;
    std::vector<double> x1_hat, x2_hat, true_signal;
    std::vector<double> obj_fun_vals;
    std::vector<double> z1;
    std::vector<double> r1, r2;
    double gam1, gam2, eta1, eta2, rho, gamw;
    std::vector<double> probs;
    std::vector<double> vars;
    double gamma_min = 1e-11;
    double gamma_max = 1e11;
    int EM_max_iter; // = 1e5;
    double EM_err_thr; // = 1e-4;
    int CG_max_iter; // = 10;
    int use_adap_damp = 0;
    int numb_adap_damp_hist = 2;
    double damp_inc_fact = 1.05;
    double damp_dec_fact = 0.95;
    double damp_max = 1;
    double stop_criteria_thr; // = 1e-5;
    std::string out_dir;
    std::string out_name;
    std::vector<double> bern_vec;
    std::vector<double> invQ_bern_vec;

public:

    vamp(int N, int M,  int Mt, double gam1, double gamw, int max_iter, double rho, std::vector<double> vars,  std::vector<double> probs, std::vector<double> true_signal, int rank, std::string out_dir, std::string out_name);
    vamp(int M, double gam1, double gamw, std::vector<double> true_signal, int rank, Options opt);
    std::vector<double> infere(data* dataset);
    //std::vector<double> predict(std::vector<double> est, data* dataset);
    double g1(double x, double gam1);
    double g1d(double x, double gam1);
    double g2d_onsager(double gam2, data* dataset);
    void updatePrior();
    void updateNoisePrec(data* dataset);
    std::vector<double> lmmse_mult(std::vector<double> v, data* dataset);
    std::vector<double> CG_solver(std::vector<double> v, data* dataset);
    double vamp_obj_func(double eta, double gam, std::vector<double> invQu, std::vector<double> u, std::vector<double> vars, std::vector<double> pi, data* dataset);
    void err_measures(data * dataset, int ind);
    std::tuple<double, double, double> state_evo(int ind, double gam_prev, double gam_before, std::vector<double> probs_before, std::vector<double> vars_before, data* dataset);
};