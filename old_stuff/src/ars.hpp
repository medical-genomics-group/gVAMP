#ifndef HYDRA_ARS_HPP_
#define HYDRA_ARS_HPP_

// Three structures for ARS


struct pars {

	// Common parameters for the densities

    // epsilon per subject (before each sampling, need to remove 
    // the effect of the sampled parameter and then carry on
    Eigen::VectorXd epsilon;			

	//Probably unnecessary to store mixture classes
	//VectorXd mixture_classes; // Vector to store mixture component C_k values
	//int used_mixture; //Write the index of the mixture we decide to use

	// Store the current variables
	double alpha;

	// Beta_j - specific variables
    Eigen::VectorXd X_j;

	//  of sum(X_j*failure)
	double sum_failure;

	// Mu-specific variables
	double sigma_mu;

	// Covariate-specific variables
	double sigma_covariate;

	// Number of events (sum of failure indicators)
	double d;
};


struct pars_beta_sparse {

	// Common parameters for the densities
    
	// Instead of storing the vector of mixtures and the corresponding 
    // index, we keep only the mixture value in the structure
	double mixture_value; 

	// Store the current variables
	double alpha, sigmaG;

	// Beta_j - specific variables
	double vi_0, vi_1, vi_2; // Sums of vi elements

	// Mean, std dev and their ratio for snp j
	double mean, sd, mean_sd_ratio;

	//  of sum(X_j*failure)
	double sum_failure;

	// Number of events (sum of failure indicators)
	double d;      
};


struct pars_alpha {

    Eigen::VectorXd failure_vector;

    // epsilon per subject (before each sampling, need to remove
    // the effect of the sampled parameter and then carry on
    Eigen::VectorXd epsilon;			

	// Alpha-specific variable; prior parameters
	double alpha_0, kappa_0;

	// Number of events (sum of failure indicators)
	double d;
};


#endif
