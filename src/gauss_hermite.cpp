#include <iostream>
#include <math.h>
#include <string>
#include <Eigen/Eigen>
#include "ars.hpp"



// The function for integration
//inline 
double gh_integrand_adaptive(double s,
                             double alpha,
                             double dj,
                             double sqrt_2Ck_sigmaG,
                             double vi_sum,
                             double vi_2,
                             double vi_1,
                             double vi_0,
                             double mean,
                             double sd,
                             double mean_sd_ratio){
	//vi is a vector of exp(vi)
	double temp = -alpha * s * dj * sqrt_2Ck_sigmaG +
        vi_sum - exp(alpha*mean_sd_ratio * s * sqrt_2Ck_sigmaG) *
        (vi_0 + vi_1 * exp(-alpha * s * sqrt_2Ck_sigmaG / sd) + vi_2 * exp(-2.0 * alpha * s * sqrt_2Ck_sigmaG / sd))
        - pow(s, 2.0);

	return exp(temp);
}


// Calculate the value of the integral using Adaptive Gauss-Hermite quadrature
// Let's assume that mu is always 0 for speed
double gauss_hermite_adaptive_integral(double C_k,
                                       double sigma,
                                       std::string n,
                                       double vi_sum,
                                       double vi_2,
                                       double vi_1,
                                       double vi_0,
                                       double mean,
                                       double sd,
                                       double mean_sd_ratio,
                                       const pars_beta_sparse used_data_beta) {

	double temp = 0.0;

	double sqrt_2ck_sigma = sqrt(2.0 * C_k * used_data_beta.sigmaG);

	if (n == "3") {

		double x1 = 1.2247448713916;
		double x2 = -x1;

		const double w1 = 1.3239311752136;
		const double w2 = w1;
		const double w3 = 1.1816359006037;

		x1 = sigma * x1;
		x2 = sigma * x2;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                           vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3;

	} else if (n == "5") {

		double x1,x2,x3,x4;//x5;
		double w1,w2,w3,w4,w5; //These are adjusted weights

		x1 = 2.0201828704561;
		x2 = -x1;
		w1 = 1.181488625536;
		w2 = w1;

		x3 = 0.95857246461382;
		x4 = -x3;
		w3 = 0.98658099675143;
		w4 = w3;

		//	x5 = 0.0;
		w5 = 0.94530872048294;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		//x5 = sigma*x5;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                           vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 ;//* gh_integrand_adaptive(x5,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j); // This part is just 1

	} else if (n == "7") {

		double x1,x2,x3,x4,x5,x6;
		double w1,w2,w3,w4,w5,w6,w7; //These are adjusted weights

		x1 = 2.6519613568352;
		x2 = -x1;
		w1 = 1.1013307296103;
		w2 = w1;

		x3 = 1.6735516287675;
		x4 = -x3;
		w3 = 0.8971846002252;
		w4 = w3;

		x5 = 0.81628788285897;
		x6 = -x5;
		w5 = 0.8286873032836;
		w6 = w5;

		w7 = 0.81026461755681;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                           vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,
                                       vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7;

	} else if(n == "9") {

		double x1,x2,x3,x4,x5,x6,x7,x8,x9;//,x11;
		double w1,w2,w3,w4,w5,w6,w7,w8,w9; //These are adjusted weights

		x1 = 3.1909932017815;
		x2 = -x1;
		w1 = 1.0470035809767;
		w2 = w1;

		x3 = 2.2665805845318;
		x4 = -x3;
		w3 = 0.84175270147867;
		w4 = w3;

		x5 = 1.4685532892167;
		x6 = -x5;
		w5 = 0.7646081250946;
		w6 = w5;

		x7 = 0.72355101875284;
		x8 = -x7;
		w7 = 0.73030245274509;
		w8 = w7;

        //	x9 = 0;
		w9 = 0.72023521560605;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;
		x7 = sigma*x7;
		x8 = sigma*x8;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w9 ;//* gh_integrand_adaptive(x9,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j);

	} else if (n == "11") {

		double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10;//,x11;
		double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11; //These are adjusted weights

		x1 = 3.6684708465596;
		x2 = -x1;
		w1 = 1.0065267861724;
		w2 = w1;

		x3 = 2.7832900997817;
		x4 = -x3;
		w3 = 0.802516868851;
		w4 = w3;

		x5 = 2.0259480158258;
		x6 = -x3;
		w5 = 0.721953624728;
		w6 = w5;

		x7 = 1.3265570844949;
		x8 = -x7;
		w7 = 0.6812118810667;
		w8 = w7;

		x9 = 0.6568095668821;
		x10 = -x9;
		w9 = 0.66096041944096;
		w10 = w9;

		//x11 = 0.0;
		w11 = 0.65475928691459;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;
		x7 = sigma*x7;
		x8 = sigma*x8;
		x9 = sigma*x9;
		x10 = sigma*x10;
		//	x11 = sigma*x11;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w11 ;//* gh_integrand_adaptive(x11,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j);

	} else if (n == "13") {

		double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12;
		double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13; //These are adjusted weights

		x1 = 4.1013375961786;
		x2 = -x1;
		w1 = 0.97458039564;
		w2 = w1;

		x3 = 3.2466089783724;
		x4 = -x3;
		w3 = 0.7725808233517;
		w4 = w3;

		x5 = 2.5197356856782;
		x6 = -x3;
		w5 = 0.6906180348378;
		w6 = w5;

		x7 = 1.8531076516015;
		x8 = -x7;
		w7 = 0.6467594633158;
		w8 = w7;

		x9 = 1.2200550365908;
		x10 = -x9;
		w9 = 0.6217160552868;
		w10 = w9;

		x11 = 0.60576387917106;
		x12 = -x11;
		w11 = 0.60852958370332;
		w12 = w11;

		//x13 = 0.0;
		w13 = 0.60439318792116;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;
		x7 = sigma*x7;
		x8 = sigma*x8;
		x9 = sigma*x9;
		x10 = sigma*x10;
		x11 = sigma*x11;
		x12 = sigma*x12;


		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w11 * gh_integrand_adaptive(x11,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w12 * gh_integrand_adaptive(x12,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w13 ;//* gh_integrand_adaptive(x11,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j);

	} else if (n == "15") {

		double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14;//,x11;
		double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15; //These are adjusted weights

		x1 = 4.4999907073094;
		x2 = -x1;
		w1 = 0.94836897082761;
		w2 = w1;

		x3 = 3.6699503734045;
		x4 = -x3;
		w3 = 0.7486073660169;
		w4 = w3;

		x5 = 2.9671669279056;
		x6 = -x3;
		w5 = 0.666166005109;
		w6 = w5;

		x7 = 2.3257324861739;
		x8 = -x7;
		w7 = 0.620662603527;
		w8 = w7;

		x9 = 1.7199925751865;
		x10 = -x9;
		w9 = 0.5930274497642;
		w10 = w9;

		x11 = 1.1361155852109;
		x12 = -x11;
		w11 = 0.5761933502835;
		w12 = w11;

		x13 = 0.5650695832556;
		x14 = -x13;
		w13 = 0.5670211534466;
		w14 = w13;

		//x15 = 0.0;
		w15 = 0.56410030872642;

		x1 = sigma*x1;
		x2 = sigma*x2;
		x3 = sigma*x3;
		x4 = sigma*x4;
		x5 = sigma*x5;
		x6 = sigma*x6;
		x7 = sigma*x7;
		x8 = sigma*x8;
		x9 = sigma*x9;
		x10 = sigma*x10;
		x11 = sigma*x11;
		x12 = sigma*x12;
		x13 = sigma*x13;
		x14 = sigma*x14;

		temp = 	w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w11 * gh_integrand_adaptive(x11,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w12 * gh_integrand_adaptive(x12,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w13 * gh_integrand_adaptive(x13,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w14 * gh_integrand_adaptive(x14,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w15 ;//* gh_integrand_adaptive(x11,p.alpha,p.sum_failure,sqrt_2ck_sigma,vi,p.X_j);

	} else if (n == "17") {

        double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16;//,x17;
        double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17; //These are adjusted weights

        x1 = 4.8713451936744;
        x2 = -x1;
        w1 = 0.92625413999;
        w2 = w1;

        x3 = 4.0619466758755;
        x4 = -x3;
        w3 = 0.728748370587;
        w4 = w3;

        x5 = 3.3789320911415;
        x6 = -x3;
        w5 = 0.6462917002129;
        w6 = w5;

        x7 = 2.7577629157039;
        x8 = -x7;
        w7 = 0.5998927326678;
        w8 = w7;

        x9 = 2.1735028266666;
        x10 = -x9;
        w9 = 0.5707392941245;
        w10 = w9;

        x11 = 1.6129243142212;
        x12 = -x11;
        w11 = 0.55177735307817;
        w12 = w11;

        x13 = 1.0676487257435;
        x14 = -x13;
        w13 = 0.5397631139085;
        w14 = w13;

        x15 = 0.53163300134266;
        x16 = -x15;
        w15 = 0.5330706545736;
        w16 = w15;

        w17 = 0.53091793762486;

        x1 = sigma*x1;
        x2 = sigma*x2;
        x3 = sigma*x3;
        x4 = sigma*x4;
        x5 = sigma*x5;
        x6 = sigma*x6;
        x7 = sigma*x7;
        x8 = sigma*x8;
        x9 = sigma*x9;
        x10 = sigma*x10;
        x11 = sigma*x11;
        x12 = sigma*x12;
        x13 = sigma*x13;
        x14 = sigma*x14;
        x15 = sigma*x15;
        x16 = sigma*x16;

        temp =  w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w11 * gh_integrand_adaptive(x11,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w12 * gh_integrand_adaptive(x12,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w13 * gh_integrand_adaptive(x13,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w14 * gh_integrand_adaptive(x14,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w15 * gh_integrand_adaptive(x15,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w16 * gh_integrand_adaptive(x16,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w17 ;//* gh_integrand_adaptive(0,...)= 1

    } else if (n == "25") {

        double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24;
        double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25; //These are adjusted weights

        x1 = 6.1642724340525;
        x2 = -x1;
        w1 = 0.862401988731;
        w2 = w1;

        x3 = 5.41363635528;
        x4 = -x3;
        w3 = 0.673022290222;
        w4 = w3;

        x5 = 4.7853203673522;
        x6 = -x3;
        w5 = 0.5920816930865;
        w6 = w5;
        x7 = 4.2186094443866;
        x8 = -x7;
        w7 = 0.5449177721944;
        w8 = w7;

        x9 = 3.690282876998;
        x10 = -x9;
        w9 = 0.513655789775;
        w10 = w9;

        x11 = 3.1882949244251;
        x12 = -x11;
        w11 = 0.4915068818876;
        w12 = w11;

        x13 = 2.705320237173;
        x14 = -x13;
        w13 = 0.4752497380022;
        w14 = w13;

        x15 = 2.2364201302673;
        x16 = -x15;
        w15 = 0.463141046575;
        w16 = w15;

        x17 = 1.7780011243372;
        x18 = -x17;
        w17 = 0.45415588552762;
        w18 = w17;

        x19 = 1.3272807020731;
        x20 = -x19;
        w19 = 0.4476612565874;
        w20 = w19;

        x21 = 0.88198275621382;
        x22 = -x21;
        w21 = 0.44325918925185;
        w22 = w21;

        x23 = 0.44014729864531;
        x24 = -x23;
        w23 = 0.44070582891206;
        w24 = w23;
        //x25 = 0.0;
        w25 = 0.43986872216949;

        x1 = sigma*x1;
        x2 = sigma*x2;
        x3 = sigma*x3;
        x4 = sigma*x4;
        x5 = sigma*x5;
        x6 = sigma*x6;
        x7 = sigma*x7;
        x8 = sigma*x8;
        x9 = sigma*x9;
        x10 = sigma*x10;
        x11 = sigma*x11;
        x12 = sigma*x12;
        x13 = sigma*x13;
        x14 = sigma*x14;
        x15 = sigma*x15;
        x16 = sigma*x16;
        x17 = sigma*x17;
        x18 = sigma*x18;
        x19 = sigma*x19;
        x20 = sigma*x20;
        x21 = sigma*x21;
        x22 = sigma*x22;
        x23 = sigma*x23;
        x24 = sigma*x24;

        temp =  w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w11 * gh_integrand_adaptive(x11,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w12 * gh_integrand_adaptive(x12,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w13 * gh_integrand_adaptive(x13,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w14 * gh_integrand_adaptive(x14,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w15 * gh_integrand_adaptive(x15,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w16 * gh_integrand_adaptive(x16,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w17 * gh_integrand_adaptive(x17,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w18 * gh_integrand_adaptive(x18,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w19 * gh_integrand_adaptive(x19,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w20 * gh_integrand_adaptive(x20,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w21 * gh_integrand_adaptive(x21,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w22 * gh_integrand_adaptive(x22,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w23 * gh_integrand_adaptive(x23,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w24 * gh_integrand_adaptive(x24,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w25 ;//* gh_integrand_adaptive(0,...)= 1

    } else {

        std::cout << "Possible number of quad_points = 3,5,7,9,11,13,15,17,25 but got " << n << std::endl;
		exit(1);
	}

	return sigma * temp;
}



///

double gauss_hermite_adaptive_integral_temp(double C_k,
                                       double sigma,
                                       std::string n,
                                       double vi_sum,
                                       double vi_2,
                                       double vi_1,
                                       double vi_0,
                                       double mean,
                                       double sd,
                                       double mean_sd_ratio,
                                       const pars_beta_sparse used_data_beta) {

	double temp = 0.0;

	double sqrt_2ck_sigma = sqrt(2.0 * C_k * used_data_beta.sigmaG);
    std::cout <<"sqrt_2ck_sigma " << sqrt_2ck_sigma << std::endl;
        std::cout <<" C_k " <<  C_k << std::endl;
        std::cout <<" used_data_beta.sigmaG " <<  used_data_beta.sigmaG << std::endl;

	if (n == "25") {

        double x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24;
        double w1,w2,w3,w4,w5,w6,w7,w8,w9,w10,w11,w12,w13,w14,w15,w16,w17,w18,w19,w20,w21,w22,w23,w24,w25; //These are adjusted weights

        x1 = 6.1642724340525;
        x2 = -x1;
        w1 = 0.862401988731;
        w2 = w1;

        x3 = 5.41363635528;
        x4 = -x3;
        w3 = 0.673022290222;
        w4 = w3;

        x5 = 4.7853203673522;
        x6 = -x3;
        w5 = 0.5920816930865;
        w6 = w5;
        x7 = 4.2186094443866;
        x8 = -x7;
        w7 = 0.5449177721944;
        w8 = w7;

        x9 = 3.690282876998;
        x10 = -x9;
        w9 = 0.513655789775;
        w10 = w9;

        x11 = 3.1882949244251;
        x12 = -x11;
        w11 = 0.4915068818876;
        w12 = w11;

        x13 = 2.705320237173;
        x14 = -x13;
        w13 = 0.4752497380022;
        w14 = w13;

        x15 = 2.2364201302673;
        x16 = -x15;
        w15 = 0.463141046575;
        w16 = w15;

        x17 = 1.7780011243372;
        x18 = -x17;
        w17 = 0.45415588552762;
        w18 = w17;

        x19 = 1.3272807020731;
        x20 = -x19;
        w19 = 0.4476612565874;
        w20 = w19;

        x21 = 0.88198275621382;
        x22 = -x21;
        w21 = 0.44325918925185;
        w22 = w21;

        x23 = 0.44014729864531;
        x24 = -x23;
        w23 = 0.44070582891206;
        w24 = w23;
        //x25 = 0.0;
        w25 = 0.43986872216949;

        x1 = sigma*x1;
        x2 = sigma*x2;
        x3 = sigma*x3;
        x4 = sigma*x4;
        x5 = sigma*x5;
        x6 = sigma*x6;
        x7 = sigma*x7;
        x8 = sigma*x8;
        x9 = sigma*x9;
        x10 = sigma*x10;
        x11 = sigma*x11;
        x12 = sigma*x12;
        x13 = sigma*x13;
        x14 = sigma*x14;
        x15 = sigma*x15;
        x16 = sigma*x16;
        x17 = sigma*x17;
        x18 = sigma*x18;
        x19 = sigma*x19;
        x20 = sigma*x20;
        x21 = sigma*x21;
        x22 = sigma*x22;
        x23 = sigma*x23;
        x24 = sigma*x24;


  /*        std::cout << gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio) << std::endl;
             std::cout <<  gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio) << std::endl;
          std::cout <<  gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio) << std::endl;
           std::cout <<  gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio) << std::endl;
     std::cout <<  gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio) << std::endl;
         std::cout <<  gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio) << std::endl;
         std::cout <<  gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
         std::cout <<  gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio) << std::endl;
         std::cout <<  gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio) << std::endl;
         std::cout <<  gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio) << std::endl;
         std::cout <<  gh_integrand_adaptive(x11,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
          std::cout <<  gh_integrand_adaptive(x12,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
          std::cout << gh_integrand_adaptive(x13,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
         std::cout <<  gh_integrand_adaptive(x14,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
          std::cout <<  gh_integrand_adaptive(x15,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
          std::cout <<  gh_integrand_adaptive(x16,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
          std::cout <<  gh_integrand_adaptive(x17,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
          std::cout <<  gh_integrand_adaptive(x18,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
          std::cout <<  gh_integrand_adaptive(x19,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
          std::cout <<  gh_integrand_adaptive(x20,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
          std::cout <<  gh_integrand_adaptive(x21,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
          std::cout <<  gh_integrand_adaptive(x22,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
          std::cout <<  gh_integrand_adaptive(x23,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
           std::cout <<  gh_integrand_adaptive(x24,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)<< std::endl;
*/

        temp =  w1 * gh_integrand_adaptive(x1,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w2 * gh_integrand_adaptive(x2,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w3 * gh_integrand_adaptive(x3,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w4 * gh_integrand_adaptive(x4,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w5 * gh_integrand_adaptive(x5,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w6 * gh_integrand_adaptive(x6,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w7 * gh_integrand_adaptive(x7,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w8 * gh_integrand_adaptive(x8,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w9 * gh_integrand_adaptive(x9,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w10 * gh_integrand_adaptive(x10,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w11 * gh_integrand_adaptive(x11,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w12 * gh_integrand_adaptive(x12,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w13 * gh_integrand_adaptive(x13,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w14 * gh_integrand_adaptive(x14,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w15 * gh_integrand_adaptive(x15,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w16 * gh_integrand_adaptive(x16,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w17 * gh_integrand_adaptive(x17,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w18 * gh_integrand_adaptive(x18,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w19 * gh_integrand_adaptive(x19,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w20 * gh_integrand_adaptive(x20,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w21 * gh_integrand_adaptive(x21,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w22 * gh_integrand_adaptive(x22,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w23 * gh_integrand_adaptive(x23,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w24 * gh_integrand_adaptive(x24,used_data_beta.alpha,used_data_beta.sum_failure,sqrt_2ck_sigma,vi_sum, vi_2, vi_1, vi_0, mean, sd, mean_sd_ratio)+
            w25 ;//* gh_integrand_adaptive(0,...)= 1

    } else {

        std::cout << "Possible number of quad_points = 3,5,7,9,11,13,15,17,25 but got " << n << std::endl;
		exit(1);
	}

	return sigma * temp;
}
