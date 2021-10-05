/*
 * distributions_boost.cpp
 *
 *  Created on: 7 Sep 2018
 *      Author: admin
 */
#include <Eigen/Eigen>
#include <math.h>
#include "distributions_boost.hpp"
#include <boost/random/gamma_distribution.hpp>
#include "boost/random.hpp"
#include "boost/generator_iterator.hpp"
#include <boost/random/normal_distribution.hpp>
#include <boost/random/beta_distribution.hpp>
#include <boost/random/uniform_01.hpp>
#include <iostream>
#include <fstream>

using namespace std;

/*
Distributions_boost::Distributions_boost(unsigned int pseed):seed(pseed){
    std::cout << "Initiating dist.rng with pseed = " << pseed << std::endl;
    rng = boost::mt19937(seed);
}
*/

Distributions_boost::Distributions_boost() {
    rng = boost::mt19937(0);
}

Distributions_boost::~Distributions_boost() {}

void Distributions_boost::reset_rng(unsigned int seed) {
    rng = boost::mt19937(seed);
}

void Distributions_boost::write_rng_state_to_file(const std::string rngfp) {
    //cout << "Dumping rng state to file " << rngfp << endl;
    ofstream file;
	file.open(rngfp.c_str(), ios::out | ios::trunc | ios::binary);
	file << rng;
	file.close();
}

void Distributions_boost::read_rng_state_from_file(const std::string rngfp) {
    //cout << "Reading rng state from file " << rngfp << endl;
    ifstream file(rngfp.c_str(), ios::in | ios::binary);
    if (file.is_open()) {
        file >> rng;
    } else {
        cout << "*FATAL*: Unable to read from file " << rngfp << endl;
        exit(1);
    }
}

double Distributions_boost::rgamma(double shape, double scale){
    boost::random::gamma_distribution<double> myGamma(shape, scale);
    boost::random::variate_generator<boost::mt19937&, boost::random::gamma_distribution<> > rand_gamma(rng, myGamma);
    return rand_gamma();
}

double Distributions_boost::unif_rng(){
    boost::random::uniform_real_distribution<double> myU(0,1);
    boost::random::variate_generator<boost::mt19937&, boost::random::uniform_real_distribution<> > real_variate_generator(rng, myU);
    return real_variate_generator();
}

Eigen::VectorXd Distributions_boost::dirichlet_rng(Eigen::VectorXi alpha) {
    int len;
    len=alpha.size();
    Eigen::VectorXd result(len);
    for(int i=0;i<len;i++)
        result[i]=rgamma((double)alpha[i], 1.0);
    result/=result.sum();
    return result;
}

Eigen::VectorXd Distributions_boost::dirichlet_rng(Eigen::VectorXd alpha) {
    int len;
    len=alpha.size();
    Eigen::VectorXd result(len);
    for(int i=0; i<len; i++)
        result[i]=rgamma(alpha[i],(double)1.0);
    result/=result.sum();
    return result;
}

double Distributions_boost::inv_gamma_rng(double shape,double scale){
    return ((double)1.0 / rgamma(shape, 1.0/scale));
}

double Distributions_boost::gamma_rng(double shape,double scale){
    return rgamma(shape, scale);
}

double Distributions_boost::inv_gamma_rate_rng(double shape,double rate){
    return (double)1.0 / gamma_rate_rng(shape, rate);
}

double Distributions_boost::gamma_rate_rng(double shape,double rate){
    return rgamma(shape,(double)1.0/rate);
}

double Distributions_boost::inv_scaled_chisq_rng(double dof,double scale){
    return inv_gamma_rng((double)0.5*dof, (double)0.5*dof*scale);
}

double Distributions_boost::norm_rng(double mean,double sigma2){
    boost::normal_distribution<double> nd(mean, std::sqrt((double)sigma2));
    boost::variate_generator<boost::mt19937&,boost::normal_distribution<> > var_nor(rng, nd);
    return var_nor();
}

inline double runif(unsigned int seed){
    boost::mt19937 rng(seed);
    static boost::uniform_01<boost::mt19937> zeroone(rng);
    return zeroone();
}

double Distributions_boost::component_probs(double b2,Eigen::VectorXd pi){

    double sum;
    double p;
    p   = runif(seed);
    sum = pi[0]*exp((-0.5*b2)/(5e-2))/sqrt(5e-2)+pi[1]*exp((-0.5*b2));

    if (p<=(pi[0]*exp((-0.5*b2)/(5e-2))/sqrt(5e-2))/sum)
        return 5e-2;
    else
        return 1;
}

double Distributions_boost::beta_rng(double a,double b){
    boost::random::beta_distribution<double> mybeta(a, b);
    boost::random::variate_generator<boost::mt19937&, boost::random::beta_distribution<> > rand_beta(rng, mybeta);
    return rand_beta();
}
