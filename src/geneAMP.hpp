/*
 * BayesRRm.h
 *
 *  Created on: 5 Sep 2018
 *      Author: admin
 */
 
#ifndef SRC_BAYESW_HPP_
#define SRC_BAYESW_HPP_

#include "BayesRRm.h"
#include "data.hpp"
#include "options.hpp"
#include "distributions_boost.hpp"
#include "ars.hpp"
#include <Eigen/Eigen>



class geneAMP : public BayesRRm
{
public:
    const double	alpha_0      = 0.01;
    const double	kappa_0      = 0.01;
    const double      sigma_mu     = 100;
    const double      sigma_covariate = 100;

    double    alpha_sigma  = 1;
    double    beta_sigma   = 0.0001;
    const string 	quad_points  = opt.quad_points;   // Number of Gaussian quadrature points
    unsigned int 	K            = opt.S.size() + 1;  // Number of mixtures + 0 class
    unsigned int    km1          = opt.S.size();      // Number of mixtures 
    const size_t    LENBUF_gamma = 3500;              // Not more than 160 "fixed effects can be used at the moment 

	// The ARS structures
	struct pars used_data;
	struct pars_beta_sparse used_data_beta;
	struct pars_alpha used_data_alpha;

	// Component variables
	MatrixXd pi_L;                   // mixture probabilities
	VectorXd marginal_likelihoods;   // likelihood for each mixture component (for specific group)
    VectorXd marginal_likelihood_0;  // 0th likelihood for each group component
    
    int numGroups;
    VectorXi groups;

    // Linear model variables
    VectorXd vi;
    

    //AMP algorithm structures and variables
    std::default_random_engine generator;
    double sigma = 0;
    double muk = 1;
    int iterNumb = 100;
    int iterAMP = 1;
    struct prior_info{
        int K_groups;
        VectorXd pi;
        VectorXd eta;
    };
    int K_groups = 3;
    VectorXd eta =VectorXd::Zero(K_groups);
    VectorXd pi =VectorXd::Zero(K_groups);
    VectorXd eta_EM;
    VectorXd pi_EM;

    geneAMP(Data &data, Options &opt) : BayesRRm(data, opt)
	{
	};

    virtual ~geneAMP();

    int runMpiGibbs_bW();
    
private:
    void init(unsigned int individualCount, unsigned int Mtot, unsigned int fixedCount);

    // double fk(double y);

    // double fkd(double y);

    // VectorXd xbeta_mult(VectorXd beta, int Ntot, int M, size_t *N1S, size_t *N1L, size_t *N2S, size_t *N2L, double *mstd, double *mave, uint *I1, uint *I2, double scale);
    // VectorXd xtr_mult(VectorXd r, int Ntot, int M, size_t *N1S, size_t *N1L, size_t *N2S, size_t *N2L, double *mstd, double *mave, uint *I1, uint *I2, double scale);

    // VectorXd KPM(VectorXd points, int num_points, int M_deg, size_t *N1S, size_t *N1L, size_t *N2S, size_t *N2L, double *mstd, double *mave, uint *I1, uint *I2, int Ntot, int M, double scaling);

    double generating_mixture_gaussians(int K, VectorXd eta, VectorXd pi);

};

#endif /* SRC_BAYESW_HPP_ */
