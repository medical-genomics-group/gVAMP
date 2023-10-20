#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <limits.h>
#include <algorithm>
#include <cmath>
#include "utilities.hpp"
#include<boost/math/distributions/students_t.hpp>

double round_dp(const double in) {
    return in;

    printf("in = %20.15f\n", in);
    double out = round(in * 1.0E12) / 1.0E12;
    printf("ou = %20.15f\n", out);
    return out;
}

void check_malloc(const void* ptr, const int linenumber, const char* filename) {
    if (ptr == NULL) {
        fprintf(stderr, "#FATAL#: malloc failed on line %d of %s\n", linenumber, filename);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void check_mpi(const int error, const int linenumber, const char* filename) {
    if (error != MPI_SUCCESS) {
        fprintf(stderr, "*FATAL*: MPI error %d at line %d of file %s\n", error, linenumber, filename);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

// Check whether a size_t can be casted to int or would overflow
int check_int_overflow(const size_t n, const int linenumber, const char* filename) {

    if (n > INT_MAX) {
        fprintf(stderr, "FATAL  : integer overflow detected on line %d of %s. %lu does not fit in type int.\n", linenumber, filename, n);
        fflush(stdout);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    return static_cast<int>(n);
}


// sampling from a mixture of gaussians
double generate_mixture_gaussians(int K_grp, std::vector<double> eta, std::vector<double> pi, long unsigned int seed)
{
    std::random_device rand_dev;
    //std::mt19937 generator(rand_dev());  
    std::mt19937 generator{seed};  
    std::uniform_real_distribution<double> unif(0.0,1.0);
    double u = unif(generator);
    double c_sum = 0;
    double out_val = 0;
    for (int j=0; j<K_grp; j++)
    {   
        c_sum += pi[j];
        if (u <= c_sum)
        {
            if (eta[j] != 0)
            {
                std::normal_distribution<double> gauss_beta_gen( 0.0, sqrt( eta[j] ) ); //2nd parameter is stddev
                out_val = gauss_beta_gen(generator);
            }
            else
                out_val = 0;  // spike is being set at zero
            break;
        }
    }
    return out_val;
}


// simulating signal from a mixture of gaussians
std::vector<double> simulate(int M, std::vector<double> eta, std::vector<double> pi, long unsigned int seed){

    int K_grp = eta.size();
    std::vector<double> signal(M, 0.0);
    //#ifdef _OPENMP
    //    #pragma omp parallel for // not to mess with seeds
    //#endif
    for (int i = 0; i < M; i++){
        signal[i] = generate_mixture_gaussians(K_grp, eta, pi, seed);
    }
    return signal;
 }

// calculation of noise precision based on the signal-to-noise ratio (SNR) and prior distribution of the signal
double noise_prec_calc(double SNR, std::vector<double> vars, std::vector<double> probs, int Mt, int N){

    double expe = 0;
    for(int i = 0; i < vars.size(); i++)
        expe += vars[i] * probs[i];
    //std::cout <<"expe = " << expe << std::endl;
    //double gamw = SNR * N / Mt / expe;
    double gamw = SNR / Mt / expe;
    return gamw;
 }

// reading signal vector from a file 
std::vector<double> read_vec_from_file(std::string filename, int M, int S){
    double value;
    int it = 0;
    std::vector<double> v;
    std::ifstream inFile(filename);
    std::string word;
    while (inFile >> value){
        if ( it >= S && it < S+M ){
            v.push_back( value );
        }
        else if ( it >= S+M ){
            break;
        }  
        it += 1;
    } 
    //std::cout << "it | S | M = " << it <<" " << S << " " << M << std::endl;
    //std::cout << "length( v ) = " << v.size() << std::endl;
    return v;
}


// storing vector to a file
void store_vec_to_file(std::string filepath, std::vector<double> vec){

    std::ofstream file;
    file.open(filepath);
 
    for(int i=0; i<vec.size(); i++)
        file << vec[i] << std::endl;

    file.close();
 }

 // inner product implementation
 double inner_prod(std::vector<double> const& u, std::vector<double> const& v, int sync){

    double accum = 0;
    #ifdef _OPENMP
    #pragma omp parallel for reduction(+ : accum)
    #endif
    for (int i = 0; i < u.size(); i++) {
        accum += u[i] * v[i];
    }

    double accum_total = 0;
    if (sync == 1){
        //std::cout << "accum = " << accum << ", rank = " << rank << std::endl;
        MPI_Allreduce(&accum, &accum_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    } 
    else {
        accum_total = accum;    
    }

    return accum_total;
 }

 double l2_norm2(std::vector<double> const& u, int sync){
    return inner_prod(u,u, sync);
 }

double log_mix_gauss_pdf_ratio(double x, std::vector<double> eta_nom, std::vector<double> eta_den, std::vector<double> pi_nom, std::vector<double> pi_den){

    double pdf_nom = 0, pdf_den = 0;
    double eta_nom_max = *(std::max_element(eta_nom.begin(), eta_nom.end()));
    double eta_den_max = *(std::max_element(eta_den.begin(), eta_den.end()));
    for (int i=0; i<eta_nom.size(); i++){
        // std::cout << "x = " << x << " eta_nom[i] = " << eta_nom[i] << " eta_nom_max = " << eta_nom_max << " -(pow(x, 2) / 2 * (eta_nom_max - eta_nom[i]) / eta_nom[i] / eta_nom_max ) = " << -(pow(x, 2) / 2 * (eta_nom_max - eta_nom[i]) / eta_nom[i] / eta_nom_max ) << std::endl;
        pdf_nom += pi_nom[i] / sqrt(eta_nom[i]) * exp(-(pow(x, 2) / 2 * (eta_nom_max - eta_nom[i]) / eta_nom[i] / eta_nom_max ) );
        
    }
    for (int i=0; i<eta_den.size(); i++){
        // std::cout << "x = " << x << " eta_den[i] = " << eta_den[i] << " eta_den_max = " << eta_den_max << " -(pow(x, 2) / 2 * (eta_den_max - eta_den[i]) / eta_den[i] / eta_den_max ) = " << -(pow(x, 2) / 2 * (eta_den_max - eta_den[i]) / eta_den[i] / eta_den_max ) << std::endl;
        pdf_den += pi_den[i] / sqrt(eta_den[i]) * exp(-(pow(x, 2) / 2 * (eta_den_max - eta_den[i]) / eta_den[i] / eta_den_max ) );
    }
    // pdf_nom / pdf_den * exp( - x*x / 2 * (eta_den_max - eta_nom_max) / eta_den_max / eta_nom_max );
    return  log(pdf_nom) - log(pdf_den) - ( x*x / 2 * (eta_den_max - eta_nom_max) / eta_den_max / eta_nom_max );
    
 }

 double calc_stdev(std::vector<double> vec, int sync){

    double sum = std::accumulate(vec.begin(), vec.end(), 0.0);
    double sq_sum = std::inner_product(vec.begin(), vec.end(), vec.begin(), 0.0);
    int vec_len = vec.size();
    
    if (sync == 1){
        double sum_total = 0;
        double sq_sum_total = 0;
        int vec_len_total = 0;
        MPI_Allreduce(&sum, &sum_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
        MPI_Allreduce(&sq_sum, &sq_sum_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD); 
        MPI_Allreduce(&vec_len, &vec_len_total, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD); 
        sum = sum_total;
        sq_sum = sq_sum_total; 
        vec_len = vec_len_total;
    }

    double mean = sum / vec_len;
    double stdev = std::sqrt( (sq_sum - vec_len * mean * mean) / (vec_len-1) );

    return stdev;
 }

 std::vector<double> divide_work(int Mt){

    int rank = 0;
    int nranks = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const int modu = Mt % nranks;
    const int size = Mt / nranks;

    int Mm = Mt % nranks != 0 ? size + 1 : size;

    int len[nranks], start[nranks];
    int cum = 0;
    for (int i=0; i<nranks; i++) {
        len[i]  = i < modu ? size + 1 : size;
        start[i] = cum;
        cum += len[i];
    }
    assert(cum == Mt);

    int M = len[rank];
    int S = start[rank];  // task marker start

    printf("INFO   : rank %4d has %d markers over tot Mt = %d, max Mm = %d, starting at S = %d\n", rank, M, Mt, Mm, S);

    std::vector<double> MS(3, 0.0);
    MS[0] = M;
    MS[1] = S;
    MS[2] = Mm;

    return MS;
 }

 void mpi_store_vec_to_file(std::string filepath_out, std::vector<double> vec, int S, int M){
    
    MPI_File outfh;
    MPI_Status status;
    MPI_File_open(MPI_COMM_SELF, filepath_out.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &outfh);
    MPI_File_set_view(outfh, S*sizeof(double), MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_write_at(outfh, 0, (void*) vec.data(), M, MPI_DOUBLE, &status);
    MPI_File_close(&outfh);
 }

 std::vector<double> mpi_read_vec_from_file(std::string filename, int M, int S){
        
    MPI_File outfh;
    MPI_Status status;
    //std::cout << "filename is " << filename << std::endl;
    MPI_File_open(MPI_COMM_SELF, filename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &outfh);
    std::vector<double> vec(M, 0.0);
    MPI_File_set_view(outfh, S*sizeof(double), MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_read_at(outfh, 0, (void*) vec.data(), M, MPI_DOUBLE, &status);
    //std::cout << "Received from process " << status.MPI_SOURCE << "; with tag " << status.MPI_TAG << ", MPI_ERROR = " << status.MPI_ERROR << std::endl;
    //int count;
    //MPI_Get_count(&status, MPI_DOUBLE, &count);
    //std::cout << "and assuming it contains doubles there are " << count << " elements" << std::endl;

    MPI_File_close(&outfh);  
    return vec;
 }

 double linear_reg1d_pvals(double sumx, double sumsqx, double sumxy, double sumy, double sumsqy, int n){

    double s2y = (sumsqy - sumy*sumy/n) / (n-1);
    double s2x = (sumsqx - sumx*sumx/n) / (n-1);
    double sxy = (sumxy - sumx*sumy/n) / (n-1);
    double rxy = sxy / sqrt(s2x*s2y);

    // double beta = rxy * s2y / s2x;
    double t = rxy * sqrt( (n-2) / (1-rxy*rxy) ); // t-statistics
    boost::math::students_t dist(n-2);
    double pvalue = 2.0*boost::math::cdf(boost::math::complement(dist,t>0 ? t:(0-t)));

    return pvalue;
 }

double normal_cdf(double value)
{
   return 0.5 * erfc(-value * M_SQRT1_2);
}

int sgn(double val) {
    return (0.0 < val) - (val < 0.0);
}

double erfcx (double x)
{
    double a, d, e, m, p, q, r, s, t;

    a = fmax (x, 0.0 - x); // NaN preserving absolute value computation

    /* Compute q = (a-4)/(a+4) accurately. [0,INF) -> [-1,1] */
    m = a - 4.0;
    p = a + 4.0;
    r = 1.0 / p;
    q = m * r;
    t = fma (q + 1.0, -4.0, a); 
    e = fma (q, -a, t); 
    q = fma (r, e, q); 

    /* Approximate (1+2*a)*exp(a*a)*erfc(a) as p(q)+1 for q in [-1,1] */
    p =             0x1.edcad78fc8044p-31;  //  8.9820305531190140e-10
    p = fma (p, q,  0x1.b1548f14735d1p-30); //  1.5764464777959401e-09
    p = fma (p, q, -0x1.a1ad2e6c4a7a8p-27); // -1.2155985739342269e-08
    p = fma (p, q, -0x1.1985b48f08574p-26); // -1.6386753783877791e-08
    p = fma (p, q,  0x1.c6a8093ac4f83p-24); //  1.0585794011876720e-07
    p = fma (p, q,  0x1.31c2b2b44b731p-24); //  7.1190423171700940e-08
    p = fma (p, q, -0x1.b87373facb29fp-21); // -8.2040389712752056e-07
    p = fma (p, q,  0x1.3fef1358803b7p-22); //  2.9796165315625938e-07
    p = fma (p, q,  0x1.7eec072bb0be3p-18); //  5.7059822144459833e-06
    p = fma (p, q, -0x1.78a680a741c4ap-17); // -1.1225056665965572e-05
    p = fma (p, q, -0x1.9951f39295cf4p-16); // -2.4397380523258482e-05
    p = fma (p, q,  0x1.3be1255ce180bp-13); //  1.5062307184282616e-04
    p = fma (p, q, -0x1.a1df71176b791p-13); // -1.9925728768782324e-04
    p = fma (p, q, -0x1.8d4aaa0099bc8p-11); // -7.5777369791018515e-04
    p = fma (p, q,  0x1.49c673066c831p-8);  //  5.0319701025945277e-03
    p = fma (p, q, -0x1.0962386ea02b7p-6);  // -1.6197733983519948e-02
    p = fma (p, q,  0x1.3079edf465cc3p-5);  //  3.7167515521269866e-02
    p = fma (p, q, -0x1.0fb06dfedc4ccp-4);  // -6.6330365820039094e-02
    p = fma (p, q,  0x1.7fee004e266dfp-4);  //  9.3732834999538536e-02
    p = fma (p, q, -0x1.9ddb23c3e14d2p-4);  // -1.0103906603588378e-01
    p = fma (p, q,  0x1.16ecefcfa4865p-4);  //  6.8097054254651804e-02
    p = fma (p, q,  0x1.f7f5df66fc349p-7);  //  1.5379652102610957e-02
    p = fma (p, q, -0x1.1df1ad154a27fp-3);  // -1.3962111684056208e-01
    p = fma (p, q,  0x1.dd2c8b74febf6p-3);  //  2.3299511862555250e-01

    /* Divide (1+p) by (1+2*a) ==> exp(a*a)*erfc(a) */
    d = a + 0.5;
    r = 1.0 / d;
    r = r * 0.5;
    q = fma (p, r, r); // q = (p+1)/(1+2*a)
    t = q + q;
    e = (p - q) + fma (t, -a, 1.0); // residual: (p+1)-q*(1+2*a)
    r = fma (e, r, q);

    /* Handle argument of infinity */
    if (a > 0x1.fffffffffffffp1023) r = 0.0;

    /* Handle negative arguments: erfcx(x) = 2*exp(x*x) - erfcx(|x|) */
    if (x < 0.0) {
        s = x * x;
        d = fma (x, x, -s);
        e = exp (s);
        r = e - r;
        r = fma (e, d + d, r); 
        r = r + e;
        if (e > 0x1.fffffffffffffp1023) r = e; // avoid creating NaN
    }
    return r;
}