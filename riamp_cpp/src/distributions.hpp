#pragma once

#include <boost/random.hpp>

class Distributions {

private:
    boost::mt19937 rng;

public:

    void set_prng(unsigned int seed) {
        rng = boost::mt19937(seed);
    }

    unsigned int get_random_number() {
        return rng();
    }
 
    boost::mt19937& get_rng() {
        return rng;
    }

    double inv_scaled_chisq_rng(const double a, const double b) {
        return inv_gamma_rng(0.5 * a, 0.5 * a * b);
    }

    double inv_gamma_rng(const double a, const double b) {
        return 1.0 / rgamma(a, 1.0 / b);
    }

    double rgamma(const double a, const double b) {
        boost::random::gamma_distribution<double> myGamma(a, b);
        boost::random::variate_generator<boost::mt19937&, boost::random::gamma_distribution<> > rand_gamma(rng, myGamma);
        double val = rand_gamma();
        return val;
    }

    double beta_rng(const double a, const double b) {
        //std::cout << "@@ beta_rng " << a << ", " << b << std::endl;
        boost::random::beta_distribution<double> mybeta(a, b);
        boost::random::variate_generator<boost::mt19937&, boost::random::beta_distribution<> > rand_beta(rng, mybeta);
        double val = rand_beta();
        //std::cout << "@@ beta_rng val = " << val << std::endl;
        return val;
    }

    double norm_rng(double mean, double sigma2) {
        //std::cout << "@@ norm_rng on " << mean << ", " << sigma2 << std::endl;
        boost::random::normal_distribution<double> nd(mean, std::sqrt(sigma2));
        boost::random::variate_generator< boost::mt19937&, boost::normal_distribution<> > var_nor(rng, nd);
        return var_nor();
    }

    double unif_rng() {
        boost::random::uniform_real_distribution<double> myU(0,1);
        boost::random::variate_generator<boost::mt19937&, boost::random::uniform_real_distribution<> > real_variate_generator(rng, myU);
        return real_variate_generator();
    }

};


