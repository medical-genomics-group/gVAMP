#ifndef HYDRA_GAUSS_HERMITE_H
#define HYDRA_GAUSS_HERMITE_H


//
double gauss_hermite_adaptive_integral(double C_k,
                                       double sigma,
                                       string n,
                                       double vi_sum,
                                       double vi_2,
                                       double vi_1,
                                       double vi_0,
                                       double mean,
                                       double sd,
                                       double mean_sd_ratio,
                                       const pars_beta_sparse used_data_beta);


double gauss_hermite_adaptive_integral_temp(double C_k,
                                       double sigma,
                                       string n,
                                       double vi_sum,
                                       double vi_2,
                                       double vi_1,
                                       double vi_0,
                                       double mean,
                                       double sd,
                                       double mean_sd_ratio,
                                       const pars_beta_sparse used_data_beta);


#endif
