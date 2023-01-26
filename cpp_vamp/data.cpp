#include "data.hpp"
#include <cmath> 
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <regex>
#include <filesystem>
#include <cassert> // contains assert
#include <mpi.h>
#include <omp.h>
#include "dotp_lut.hpp"
#include "na_lut.hpp"
//#include "perm_lut.hpp"
//#include "perm_dotp_lut.hpp"
#include "utilities.hpp"
#include <immintrin.h>
#include <bits/stdc++.h>



// constructors for class data
data::data(std::string fp, std::string genofp, const int N, const int M, const int Mt, const int S, const int normal, const int rank, std::string type_data) :
    phenfp(fp),
    bedfp(genofp),
    type_data(type_data),
    N(N),
    M(M),
    Mtotal(Mt),
    S(S),
    rank(rank),
    mbytes(( N % 4 ) ? (size_t) N / 4 + 1 : (size_t) N / 4),
    normal_data(normal),
    im4(N%4 == 0 ? N/4 : N/4+1) {
    mave = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(mave, __LINE__, __FILE__);
    msig = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(msig, __LINE__, __FILE__);

    if (type_data == "bed"){
        // in the constructor we load phen, read genotype data and compute marker statistics
        read_phen();
        read_genotype_data();
        compute_markers_statistics();

        if (normal == 2){
            double SK_err_thr = 1e-4;
            int SK_max_iter = 20;
            xR = std::vector<double> (M, 1.0);
            xL = std::vector<double> (4*mbytes, 1.0);
            //std::cout << "before SK!" << std::endl;
            SinkhornKnopp(xL, xR, SK_err_thr, SK_max_iter);
        }
        // perm_idxs = std::vector<int>(M, 1);
        // p_luts = perm_luts();
        // p_dotp_luts = perm_dotp_luts();
    } 
    else if(type_data == "meth"){
        methfp = genofp;
        read_phen();
        read_methylation_data();
        compute_markers_statistics();
    }
    
}

// constructors for class data
data::data(std::vector<double> y, std::string genofp, const int N, const int M, const int Mt, const int S, const int normal, const int rank, std::string type_data) :
    type_data(type_data),
    N(N),
    M(M),
    Mtotal(Mt),
    S(S),
    rank(rank),
    phen_data(y),
    mbytes(( N % 4 ) ? (size_t) N / 4 + 1 : (size_t) N / 4),
    normal_data(normal),
    im4(N%4 == 0 ? N/4 : N/4+1) {
    mave = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(mave, __LINE__, __FILE__);
    msig = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(msig, __LINE__, __FILE__);

    for (int i=0; i<N; i++){
        int m4 = i % 4;
        if (m4 == 0)  mask4.push_back(0b00001111); // we are interested only in last 4 bits -> 1111 = all values present
    }  
    // Set last bits to 0 if ninds % 4 != 0
    m4 = N % 4;
    if (m4 != 0) {
        for (int i=m4; i<4; i++) {
            mask4.at(int(N / 4)) &= ~(0b1 << i);
        }
        std::cout << "rank = " << rank << ": setting last " << 4 - m4 << " bits to NAs" << std::endl;
    }
    
    set_nonas(N);

    if (type_data == "bed"){
        bedfp = genofp;
        // we read genotype data and compute marker statistics
        read_genotype_data();
        compute_markers_statistics();

        if (normal == 2){
            double SK_err_thr = 1e-4;
            int SK_max_iter = 20;
            xR = std::vector<double> (M, 1.0);
            xL = std::vector<double> (4*mbytes, 1.0);
            //std::cout << "before SK!" << std::endl;
            SinkhornKnopp(xL, xR, SK_err_thr, SK_max_iter);
        }
        // perm_idxs = std::vector<int>(M, 1);
        // p_luts = perm_luts();
        // p_dotp_luts = perm_dotp_luts();
    }
    else if (type_data == "meth"){
        methfp = genofp;
        read_methylation_data();
        compute_markers_statistics();
    }
    
}

/*
// constructors for class data
data::data(std::string fp, std::string genofp, const int N, const int M, const int Mt, const int S, const int normal, std::string type_data, const int rank) :
    N(N),
    M(M),
    type_data(type_data),
    phenfp(fp),
    Mtotal(Mt),
    S(S),
    rank(rank),
    mbytes(( N % 4 ) ? (size_t) N / 4 + 1 : (size_t) N / 4),
    normal_data(normal),
    im4(N%4 == 0 ? N/4 : N/4+1) {
    mave = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(mave, __LINE__, __FILE__);
    msig = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(msig, __LINE__, __FILE__);

    // we read genotype data and compute marker statistics
    if (type_data == "bed"){
        bedfp = genofp;
        read_phen();
        read_genotype_data();
        compute_markers_statistics();

        if (normal == 2){
            double SK_err_thr = 1e-4;
            int SK_max_iter = 20;
            xR = std::vector<double> (M, 1.0);
            xL = std::vector<double> (4*mbytes, 1.0);
            //std::cout << "before SK!" << std::endl;
            SinkhornKnopp(xL, xR, SK_err_thr, SK_max_iter);
        }
    }
    else if (type_data == "meth"){
        methfp = genofp;
        read_phen();
        read_methylation_data();
        compute_markers_statistics();
    }
}
*/

/*
data::data(std::string fp, std::string bedfp, const int N, const int M, const int Mt, const int S, const int rank, const int perm) :
    phenfp(fp),
    bedfp(bedfp),
    N(N),
    M(M),
    S(S),
    rank(rank),
    perm(perm),
    mbytes(( N % 4 ) ? (size_t) N / 4 + 1 : (size_t) N / 4),
    im4(N%4 == 0 ? N/4 : N/4+1) {
    mave = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(mave, __LINE__, __FILE__);
    msig = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(msig, __LINE__, __FILE__);

    if (perm == 0)
        perm_idxs = std::vector<int>(M, 1);
    else if (perm == 1)
        for (int i=0; i<M; i++)
            perm_idxs.push_back( (rand() % 6) + 1);

    p_luts = perm_luts();
    p_dotp_luts = perm_dotp_luts();
}
*/

// Read phenotype file assuming PLINK format:
// Family ID, Individual ID, Phenotype; One row per individual
void data::read_phen() {

    std::ifstream infile(phenfp);
    std::string line;
    std::regex re("\\s+");

    double sum = 0.0;

    if (infile.is_open()) {
        int line_n = 0;
        nonas = 0, nas = 0;
        while (getline(infile, line)) {
            int m4 = line_n % 4;
            if (m4 == 0)  mask4.push_back(0b00001111); // we are interested only in last 4 bits -> 1111 = all values present

            std::sregex_token_iterator first{line.begin(), line.end(), re, -1}, last;
            std::vector<std::string> tokens{first, last};
            if (tokens[2] == "NA") {
                nas += 1;
                phen_data.push_back(std::numeric_limits<double>::max());
                mask4.at(int(line_n / 4)) &= ~(0b1 << m4);
            } else {
                nonas += 1;
                phen_data.push_back(atof(tokens[2].c_str()));
                sum += atof(tokens[2].c_str());
            }

            line_n += 1;
        }
        infile.close();
        assert(nas + nonas == N);

        // Set last bits to 0 if ninds % 4 != 0
        const int m4 = line_n % 4;
        if (m4 != 0) {
            for (int i=m4; i<4; i++) {
                mask4.at(int(line_n / 4)) &= ~(0b1 << i);
            }
            std::cout << "rank = " << rank << ": setting last " << 4 - m4 << " bits to NAs" << std::endl;
        }

        // Center and scale
        double avg = sum / double(nonas);

        double sqn = 0.0;
        for (int i=0; i<phen_data.size(); i++) {
            if (phen_data[i] != std::numeric_limits<double>::max()) {
                sqn += (phen_data[i] - avg) * (phen_data[i] - avg);
            }
        }
        sqn = sqrt( double(nonas-1) / sqn );
        for (int i=0; i<phen_data.size(); i++)
            phen_data[i] *= sqn;

        // saving intercept and scale term for phenotypes
        intercept = avg;
        scale = sqn; // inverse standard deviation

    } else {
        std::cout << "FATAL: could not open phenotype file: " << phenfp << std::endl;
        exit(EXIT_FAILURE);
    }
}


// Compute mean and associated standard deviation for markers
// for each of the phenotypes (stats are NA dependent)
// ! one byte of bed  contains information for 4 individuals
// ! one byte of phen contains information for 8 individuals
void data::compute_markers_statistics() {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const std::vector<unsigned char> mask4 = get_mask4();
    const int im4 = get_im4();
    double* mave = get_mave();
    double* msig = get_msig();

    double start = MPI_Wtime();

    if (type_data == "bed"){

        #ifdef MANVECT
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
                for (int i=0; i<M; i++) {
                    __m256d suma = _mm256_set1_pd(0.0);
                    __m256d sumb = _mm256_set1_pd(0.0);
                    __m256d luta, lutb, lutna;
                    size_t bedix = size_t(i) * size_t(mbytes);
                    const unsigned char* bedm = &bed_data[bedix];
                    for (int j=0; j<mbytes; j++) {
                        luta  = _mm256_load_pd(&dotp_lut_a[bedm[j] * 4]); // bedm[j] contains 1 byte, an information about 4 people, hence 2^8 = 256 possibilities
                        //luta  = _mm256_load_pd(&(p_luts[perm_idxs[i]-1][bedm[j] * 4]));
                        lutb  = _mm256_load_pd(&dotp_lut_b[bedm[j] * 4]);
                        lutna = _mm256_load_pd(&na_lut[mask4[j] * 4]);
                        luta  = _mm256_mul_pd(luta, lutna);
                        lutb  = _mm256_mul_pd(lutb, lutna);
                        suma  = _mm256_add_pd(suma, luta);
                        sumb  = _mm256_add_pd(sumb, lutb);
                    }
                    double asum = suma[0] + suma[1] + suma[2] + suma[3];
                    double bsum = sumb[0] + sumb[1] + sumb[2] + sumb[3];
                    double avg  = asum / bsum; // calculation of average value of one marker column

                    __m256d vave = _mm256_set1_pd(-avg);
                    __m256d sums = _mm256_set1_pd(0.0);
                    for (int j=0; j<mbytes; j++) {
                        luta  = _mm256_load_pd(&dotp_lut_a[bedm[j] * 4]);
                        //luta  = _mm256_load_pd(&(p_luts[perm_idxs[i]-1][bedm[j] * 4]));
                        lutb  = _mm256_load_pd(&dotp_lut_b[bedm[j] * 4]);
                        lutna = _mm256_load_pd(&na_lut[mask4[j] * 4]);
                        luta  = _mm256_add_pd(luta, vave);    // - mu
                        luta  = _mm256_mul_pd(luta, lutb);    // M -> 0.0
                        luta  = _mm256_mul_pd(luta, lutna);   // NAs
                        luta  = _mm256_mul_pd(luta, luta);    // ^2
                        sums  = _mm256_add_pd(sums, luta);    // sum
                    }
                    double sig = 1.0 / sqrt((sums[0] + sums[1] + sums[2] + sums[3]) / (double(get_nonas()) - 1.0)); // calculation of inverse standard deviation of one marker column
                    mave[i] = avg;
                    msig[i] = sig;
            }
        #else
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
                for (int i=0; i<M; i++) {
                    size_t bedix = size_t(i) * size_t(mbytes);
                    const unsigned char* bedm = &bed_data[bedix];
                    double suma = 0.0;
                    double sumb = 0.0;
                    for (int j=0; j<im4; j++) {
                        for (int k=0; k<4; k++) {
                            //suma += p_luts[perm_idxs[i]-1][bedm[j] * 4 + k] * na_lut[mask4[j] * 4 + k];
                            suma += dotp_lut_a[bedm[j] * 4 + k] * na_lut[mask4[j] * 4 + k]; 
                            sumb += dotp_lut_b[bedm[j] * 4 + k] * na_lut[mask4[j] * 4 + k];
                        }
                    }
                    mave[i] = suma / sumb;
                    double sumsqr = 0.0;
                    for (int j=0; j<im4; j++) {
                        for (int k=0; k<4; k++) {
                            double val = (dotp_lut_a[bedm[j] * 4 + k] - mave[i]) * dotp_lut_b[bedm[j] * 4 + k] * na_lut[mask4[j] * 4 + k];
                            //double val = (p_luts[perm_idxs[i]-1][bedm[j] * 4 + k] - mave[i]) * dotp_lut_b[bedm[j] * 4 + k] * na_lut[mask4[j] * 4 + k];
                            // calculate the value and filter for nans in genotype and phenotype
                            sumsqr += val * val;
                        }
                    }
                    msig[i] = 1.0 / sqrt(sumsqr / (double( get_nonas() ) - 1.0));
                }
            #endif
    }
    else if (type_data == "meth"){
        int im4m1 = im4-1;
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
                for (int i=0; i<M; i++) {
                    size_t methix = size_t(i) * size_t(N);
                    const double* methm = &meth_data[methix];
                    double suma = 0.0;

                    // calculating marker mean in 2 step process since size(phen) = 4*mbytes and size(methm) = N
                    // currently only non-missing methylation data is allowed 
                    for (int j=0; j<(im4-1); j++) {
                        #ifdef _OPENMP
                        #pragma omp simd reduction(+:suma)
                        #endif
                        for (int k=0; k<4; k++) 
                            suma += methm[4*j + k] * na_lut[mask4[j] * 4 + k];
                    }  
                    for (int k=0; k<4; k++) {
                        if (4*im4m1 + k < N){
                            suma += methm[4*im4m1 + k] * na_lut[mask4[im4m1] * 4 + k];
                        } 
                    }

                    //calculating vector of marker precision
                    mave[i] = suma / double( get_nonas() );
                    double sumsqr = 0.0;
                    for (int j=0; j<(im4-1); j++) {
                        #ifdef _OPENMP
                        #pragma omp simd reduction(+:sumsqr)
                        #endif
                        for (int k=0; k<4; k++) {
                            double val = (methm[4*j + k] - mave[i]) * na_lut[mask4[j] * 4 + k];
                            sumsqr += val * val;
                        }
                    }
                    for (int k=0; k<4; k++) {
                        if (4*im4m1 + k < N){
                            double val = (methm[4*im4m1 + k] - mave[i]) * na_lut[mask4[im4m1] * 4 + k];
                            sumsqr += val * val;
                        } 
                    }

                    /*
                    for (int j=0; j<im4; j++) {
                        for (int k=0; k<4; k++) {
                            suma += methm[4*j + k]; // currently only non-missing methylation data is allowed
                        }
                    }
                    mave[i] = suma / double( get_nonas() );
                    double sumsqr = 0.0;
                    for (int j=0; j<im4; j++) {
                        for (int k=0; k<4; k++) {
                            double val = (methm[4*j + k] - mave[i]) * na_lut[mask4[j] * 4 + k];
                            sumsqr += val * val;
                        }
                    }
                    */
                    // std::cout << "sumsqr = " << sumsqr << std::endl;
                    // std::cout << "nonas = " << get_nonas() << std::endl;
                    msig[i] = 1.0 / sqrt(sumsqr / (double( get_nonas() ) - 1.0));
                }
    }
        double end = MPI_Wtime();
        if (rank == 0)
            std::cout << "rank = " << rank << ": statistics took " << end - start << " seconds to run." << std::endl;
}

double data::dot_product(const int mloc, double* __restrict__ phen, const double mu, const double sigma_inv, int normal) {
// __restrict__ means that phen is the only pointer pointing to that data

if (type_data == "bed"){

    unsigned char* bed = &bed_data[mloc * mbytes];

    #ifdef MANVECT
        if (normal == 1){
            __m256d luta; // lutb, lutna;
            __m512d lutab, p42;
            __m256d suma = _mm256_set1_pd(0.0);
            __m512d sum42 = _mm512_set1_pd(0.0);
            #ifdef _OPENMP
            //#pragma omp parallel for schedule(static) reduction(addpd4:suma,sumb)
            #pragma omp parallel for schedule(static) private(luta,lutab,p42) reduction(addpd8:sum42)
            #endif
            for (int j=0; j<mbytes; j++){
                lutab = _mm512_load_pd(&dotp_lut_ab[bed[j] * 8]);
                // broadcasts the 4 packed double-precision (64-bit) floating-point elements
                p42 = _mm512_broadcast_f64x4(_mm256_load_pd(&phen[j * 4])); 
                p42 = _mm512_mul_pd(p42, lutab);
                sum42 = _mm512_add_pd(sum42, p42);
            }
            return sigma_inv * (sum42[0] + sum42[1] + sum42[2] + sum42[3] - mu * (sum42[4] + sum42[5] + sum42[6] + sum42[7]));
        }
        else if (normal == 0){
            __m256d luta, p4;
            __m256d suma = _mm256_set1_pd(0.0);
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) private(luta,p4) reduction(addpd4:suma)
            #endif
            for (int j=0; j<mbytes; j++){
                luta  = _mm256_load_pd(&dotp_lut_a[bed[j] * 4]);
                p4 = _mm256_load_pd(&phen[j * 4]); 
                // __m256d p4 = _mm256_load_pd(&phen[j * 4]);
                luta  = _mm256_mul_pd(luta, p4);
                suma  = _mm256_add_pd(suma, luta);
            }
            return (suma[0] + suma[1] + suma[2] + suma[3]);
        }
        else if (normal == 2){
            __m256d luta, p4, mmxL;
            __m256d suma = _mm256_set1_pd(0.0);
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) private(luta,p4,mmxL) reduction(addpd4:suma)
            #endif
            for (int j=0; j<mbytes; j++){
                luta  = _mm256_load_pd(&dotp_lut_a[bed[j] * 4]);
                p4 = _mm256_load_pd(&phen[j * 4]); 
                mmxL =  _mm256_load_pd(&xL[j * 4]);
                p4 = _mm256_mul_pd(p4, mmxL);
                luta  = _mm256_mul_pd(luta, p4);
                suma  = _mm256_add_pd(suma, luta);
            }
            return (xR[mloc] * (suma[0] + suma[1] + suma[2] + suma[3]));
        }
        // if none of the valid normalization is chosen return numeric max
        return std::numeric_limits<double>::max();

    #else

        if (normal == 1){
            double dpa = 0.0;
            double dpb = 0.0;

            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) reduction(+:dpa,dpb)
            #endif
            for (int i=0; i<mbytes; i++) {
            //double dotp_lut_a_p[2048];
            //std::copy(&p_luts[perm_idxs[mloc]-1][0], &p_luts[perm_idxs[mloc]-1][1024], dotp_lut_a_p);
            //dotp_lut_a_p = p_luts[perm_idxs[mloc]-1];
            #ifdef _OPENMP
            #pragma omp simd aligned(dotp_lut_a,dotp_lut_b,phen:32)
            //#pragma omp simd aligned(dotp_lut_a_p,dotp_lut_b,phen:32)
            #endif
                for (int j=0; j<4; j++) {
                    dpa += dotp_lut_a[bed[i] * 4 + j] * phen[i * 4 + j];
                    //dpa += dotp_lut_a_p[bed[i] * 4 + j] * phen[i * 4 + j];
                    dpb += dotp_lut_b[bed[i] * 4 + j] * phen[i * 4 + j];
                }
            }
            return sigma_inv * (dpa - mu * dpb);
        }
        else if (normal == 0){
            double dpa = 0.0;
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) reduction(+:dpa)
            #endif
            for (int i=0; i<mbytes; i++) {
            #ifdef _OPENMP
            #pragma omp simd aligned(dotp_lut_a,phen:32)
            #endif
                for (int j=0; j<4; j++) {
                    dpa += dotp_lut_a[bed[i] * 4 + j] * phen[i * 4 + j];
                }
            }
            return dpa;
        }
        else if (normal == 2){
            double dpa = 0.0;
            double* xLdata = xL.data();
            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) reduction(+:dpa)
            #endif
            for (int i=0; i<mbytes; i++) {
            #ifdef _OPENMP
            #pragma omp simd aligned(dotp_lut_a,phen,xLdata:32)
            #endif
                for (int j=0; j<4; j++) {
                    dpa += dotp_lut_a[bed[i] * 4 + j] * phen[i * 4 + j] * xLdata[i * 4 + j];
                }
            }
            return xR[mloc] * dpa;
        }
        // if none of the valid normalization is chosen return numeric max
        return std::numeric_limits<double>::max();
    #endif
}
else if (type_data == "meth"){

    double* meth = &meth_data[mloc * N];

    if (normal == 1){
        double dpa = 0.0;

        #ifdef _OPENMP
        #pragma omp parallel for simd schedule(static) reduction(+:dpa)
        #endif
        for (int i=0; i<N; i++)
            dpa += (meth[i] - mu) * phen[i];

        /*
        for (int i=0; i<mbytes; i++) {
        #ifdef _OPENMP
        #pragma omp simd
        #endif
            for (int j=0; j<4; j++) {
                // if (!std::isnan(meth[i * 4 + j]))
                dpa += (meth[i * 4 + j] - mu) * phen[i * 4 + j];
            }
        }
        */
        return sigma_inv * dpa;
    }
}
return std::numeric_limits<double>::max();
}


std::vector<double> data::ATx(double* __restrict__ phen, int normal) {

    std::vector<double> ATx(M, 0.0);
    #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
    #endif
    for (int mloc=0; mloc < M; mloc++)
         ATx[mloc] = dot_product(mloc, phen, mave[mloc], msig[mloc], normal);
    if (normal == 1)
        for (int mloc=0; mloc < M; mloc++)
            ATx[mloc] /= sqrt(N);
    return ATx;
}

std::vector<double> data::Ax(double* __restrict__ phen, int normal) {

if (type_data == "bed"){
 #ifdef MANVECT

    //if (rank == 0)
    //    std::cout << "MANCVECT is active!" << std::endl;
    double* mave = get_mave();
    double* msig = get_msig();
    std::vector<double> Ax_temp(4 * mbytes, 0.0);

    // #pragma omp declare reduction(reduction-identifier : typename-list : combiner) [initializer-clause] new-line, where:

    // 1. reduction-identifier is either an id-expression or one of the following operators: +, -, *, &, |, ^, && or ||
    // 2. typename-list is a list of type names
    // 3. combiner is an expression
    // 4. initializer-clause is initializer(initializer-expr) where initializer-expr is omp_priv initializer or function-name(argument-list)
    
    // a. The only variables allowed in the combiner are omp_in and omp_out.
    // b. The only variables allowed in the initializer-clause are omp_priv and omp_orig.

    // for more details see here: http://www.archer.ac.uk/training/course-material/2018/07/AdvOpenMP-camb/L09-OpenMP4.pdf
    
    /*
    #ifdef _OPENMP2

    if (rank == 0)
        std::cout << "_OPENMP2 is defined!" << std::endl;

    #pragma omp declare reduction(vec_add_d : std::vector<double> : std::transform(omp_in.begin(),omp_in.end(),omp_out.begin(),omp_out.begin(),std::plus<double>())) initializer (omp_priv=omp_orig)

    #pragma omp parallel
    {
        #pragma omp for nowait schedule(static) reduction(vec_add_d:Ax_temp)

    #endif
    */

    if (normal == 1){
        for (int i=0; i<M; i++){
            unsigned char* bedm = &bed_data[i * mbytes];

            __m256d luta, lutb;
            __m256d minave = _mm256_set1_pd(-mave[i]);
            __m256d sig = _mm256_set1_pd(msig[i]);
            __m256d suma = _mm256_set1_pd(0.0); // setting all 4 values to 0.0
            __m256d mmphen = _mm256_set1_pd(phen[i]);
            //#ifdef _OPENMP2

            //#pragma omp parallel private(suma, luta, lutb) shared(Ax_temp)
            //{
                //#pragma omp for nowait schedule(static) // reduction(vec_add_d:Ax_temp)
            
            //#endif
            for (int j=0; j<mbytes; j++) {
                luta = _mm256_load_pd(&dotp_lut_a[bedm[j] * 4]);
                //luta = _mm256_load_pd(&p_luts[perm_idxs[i]-1][bedm[j] * 4]);
                lutb = _mm256_load_pd(&dotp_lut_b[bedm[j] * 4]);
                luta = _mm256_add_pd(luta, minave);
                luta = _mm256_mul_pd(luta, sig);
                luta = _mm256_mul_pd(luta, lutb); // because we need to multiply - mu / sigma with 0
                suma = _mm256_load_pd(&Ax_temp[j * 4]);
                suma = _mm256_fmadd_pd(luta, mmphen, suma);
                _mm256_storeu_pd(&Ax_temp[j * 4], suma);
            }
            // #ifdef _OPENMP2
            //}
            // #endif  
        }
    }
    else if (normal == 0){
        for (int i=0; i<M; i++){
            unsigned char* bedm = &bed_data[i * mbytes];
            __m256d luta;
            __m256d suma = _mm256_set1_pd(0.0); // setting all 4 values to 0.0
            __m256d mmphen = _mm256_set1_pd(phen[i]);
            for (int j=0; j<mbytes; j++) {
                luta = _mm256_load_pd(&dotp_lut_a[bedm[j] * 4]);
                suma = _mm256_load_pd(&Ax_temp[j * 4]);
                suma = _mm256_fmadd_pd(luta, mmphen, suma);
                _mm256_storeu_pd(&Ax_temp[j * 4], suma);
            }
        }
    }
    else if (normal == 2){ // means we use normalization s.t. X = sqrt(Diag(xL)) * G * sqrt(Diag(xR))
        for (int i=0; i<M; i++){
            unsigned char* bedm = &bed_data[i * mbytes];
            __m256d luta;
            __m256d suma = _mm256_set1_pd(0.0); // setting all 4 values to 0.0
            // xR should already containt sqrt of a diag vector
            __m256d xR_mmphen = _mm256_set1_pd(xR[i] * phen[i]); 
            for (int j=0; j<mbytes; j++) {
                luta = _mm256_load_pd(&dotp_lut_a[bedm[j] * 4]);
                suma = _mm256_load_pd(&Ax_temp[j * 4]);
                suma = _mm256_fmadd_pd(luta, xR_mmphen, suma);
                _mm256_storeu_pd(&Ax_temp[j * 4], suma);
            }
        }
    }

    std::vector<double> Ax_total(4 * mbytes, 0.0);
    MPI_Allreduce(Ax_temp.data(), Ax_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (normal == 1)
        for(int i = 0; i < 4 * mbytes; i++)
            Ax_total[i] /= sqrt(N);
    else if (normal == 2){
        for(int i = 0; i < 4 * mbytes; i++)
            Ax_total[i] *= xL[i];  
    }
    return Ax_total;

 #else

    std::vector<double> Ax_temp(4 * mbytes, 0.0);
    unsigned char* bed;

    //#ifdef _OPENMP
    //#pragma omp declare reduction(vec_double_plus : std::vector<double> : \
                              std::transform(omp_out.begin(), omp_out.end(), omp_in.begin(), omp_out.begin(), std::plus<double>())) \
                    initializer(omp_priv = decltype(omp_orig)(omp_orig.size()))

    //#pragma omp parallel for reduction(vec_double_plus : Ax_temp)
    //#endif

    if (normal == 1){
        for (int i=0; i<M; i++){
            bed = &bed_data[i * mbytes];
            double ave = mave[i];
            double sig = msig[i];
            double phen_i = phen[i];
            double sig_phen_i = sig * phen_i;
            
            #ifdef _OPENMP
                #pragma omp parallel for shared(Ax_temp)
            #endif
            for (int j=0; j<mbytes; j++) {
                #ifdef _OPENMP
                #pragma omp simd aligned(dotp_lut_a,dotp_lut_b:32)
                #endif
                for (int k=0; k<4; k++) {
                    // because msig[i] is the precision of the i-th marker
                    Ax_temp[j * 4 + k] += (dotp_lut_a[bed[j] * 4 + k] - ave) * sig_phen_i * dotp_lut_b[bed[j] * 4 + k]; 
                }
            } 
        }
    }
    else if (normal == 0){
        for (int i=0; i<M; i++){
            bed = &bed_data[i * mbytes];
            double phen_i = phen[i];
            
            #ifdef _OPENMP
                #pragma omp parallel for shared(Ax_temp)
            #endif
            for (int j=0; j<mbytes; j++) {
                #ifdef _OPENMP
                #pragma omp simd aligned(dotp_lut_a:32)
                #endif
                for (int k=0; k<4; k++) {
                    Ax_temp[j * 4 + k] += dotp_lut_a[bed[j] * 4 + k] * phen_i;
                }
            } 
        }
    }
    else if (normal == 2){
        //alignas(32) double *Ax_temp_data = Ax_temp.data();
        for (int i=0; i<M; i++){
            bed = &bed_data[i * mbytes];
            double xRphen_i = xR[i] * phen[i];
            #ifdef _OPENMP
                #pragma omp parallel for shared(Ax_temp)
            #endif
            for (int j=0; j<mbytes; j++) {
                //#ifdef _OPENMP
                //#pragma omp simd aligned(dotp_lut_a,Ax_temp:32)
                //#endif
                for (int k=0; k<4; k++) {
                    Ax_temp[j * 4 + k] += dotp_lut_a[bed[j] * 4 + k] * xRphen_i;
                }
            } 
        }

        for (int j=0; j<mbytes; j++){
            for (int k=0; k<4; k++) {
                Ax_temp[j * 4 + k] *= xL[j * 4 + k];
            }
        }
    }

    // collecting results from different nodes
    std::vector<double> Ax_total(4 * mbytes, 0.0);
    MPI_Allreduce(Ax_temp.data(), Ax_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (normal == 1)
        for(int i = 0; i < 4 * mbytes; i++)
            Ax_total[i] /= sqrt(N);
    return Ax_total;
#endif
}
else if (type_data == "meth"){

    double* mave = get_mave();
    double* msig = get_msig();
    double* meth;
    std::vector<double> Ax_temp(4 * mbytes, 0.0);    

    if (normal == 1){
        for (int i=0; i<M; i++){
            meth = &meth_data[i * N];
            double ave = mave[i];
            double sig_phen_i = msig[i] * phen[i];
            /*
            std::cout << "ave = " << ave << std::endl;
            std::cout << "msig[i] = " << msig[i] << std::endl;
            std::cout << "phen[i] = " << phen[i] << std::endl;
            std::cout << "sig_phen_i = " << sig_phen_i << std::endl;
            */
            
            #ifdef _OPENMP
                #pragma omp parallel for simd shared(Ax_temp)
            #endif
            for (int j=0; j<N; j++)
                Ax_temp[j] += (meth[j] - ave) * sig_phen_i;

            /*
            #ifdef _OPENMP
                #pragma omp parallel for shared(Ax_temp)
            #endif
            for (int j=0; j<mbytes; j++) {
                for (int k=0; k<4; k++) {
                    // if (!std::isnan(meth[j * 4 + k])) // currently only non-missing methylation data is allowed
                        Ax_temp[j * 4 + k] += (meth[j * 4 + k] - ave) * sig_phen_i;
                }
            } 
            */
        }
    }

    // collecting results from different nodes
    std::vector<double> Ax_total(4 * mbytes, 0.0);
    MPI_Allreduce(Ax_temp.data(), Ax_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (normal == 1)
        for(int i = 0; i < N; i++)
            Ax_total[i] /= sqrt(N);
    return Ax_total;
}
return std::vector<double>(4 * mbytes, 0.0);
}

void data::read_genotype_data(){

    double ts = MPI_Wtime();
    MPI_File bedfh;
    check_mpi(MPI_File_open(MPI_COMM_WORLD, bedfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &bedfh),  __LINE__, __FILE__);

    const size_t size_bytes = size_t(M) * size_t(mbytes) * sizeof(unsigned char);

    bed_data = (unsigned char*)_mm_malloc(size_bytes, 64);
    printf("INFO   : rank %d has allocated %zu bytes (%.3f GB) for raw data.\n", rank, size_bytes, double(size_bytes) / 1.0E9);

    // Offset to section of bed file to be processed by task
    MPI_Offset offset = size_t(3) + size_t(S) * size_t(mbytes) * sizeof(unsigned char);

    // Gather the sizes to determine common number of reads
    size_t max_size_bytes = 0;
    check_mpi(MPI_Allreduce(&size_bytes, &max_size_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);

    const int NREADS = size_t( ceil(double(max_size_bytes)/double(INT_MAX/2)) );
    size_t bytes = 0;
    mpi_file_read_at_all <unsigned char*> (size_bytes, offset, bedfh, MPI_UNSIGNED_CHAR, NREADS, bed_data, bytes);
    MPI_File_close(&bedfh);

    double te = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        std::cout <<"reading genotype data took " << te - ts << " seconds."<< std::endl;
}

void data::read_methylation_data(){

    double ts = MPI_Wtime();
    MPI_File methfh;
    if (rank == 0)
        std::cout << "meth file name = " << methfp.c_str() << std::endl;
    check_mpi(MPI_File_open(MPI_COMM_WORLD, methfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &methfh),  __LINE__, __FILE__);

    const size_t size_bytes = size_t(M) * size_t(N) * sizeof(double);

    meth_data = (double*)_mm_malloc(size_bytes, 64);
    printf("INFO  : rank %d has allocated %zu bytes (%.3f GB) for raw data.\n", rank, size_bytes, double(size_bytes) / 1.0E9);

    // Offset to section of bed file to be processed by task
    MPI_Offset offset = size_t(0) + size_t(S) * size_t(N) * sizeof(double);

    // Gather the sizes to determine common number of reads
    size_t max_size_bytes = 0;
    check_mpi(MPI_Allreduce(&size_bytes, &max_size_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);

    const int NREADS = size_t( ceil(double(max_size_bytes)/double(INT_MAX/2)) );
    size_t bytes = 0;
    mpi_file_read_at_all <double*> (size_t(M) * size_t(N), offset, methfh, MPI_DOUBLE, NREADS, meth_data, bytes);
    MPI_File_close(&methfh);
    
    /*if (rank == 0){
        std::cout << "meth_data[0] = " << meth_data[0] << std::endl;
        std::cout << "meth_data[1] = " << meth_data[1] << std::endl;
        std::cout << "meth_data[2] = " << meth_data[2] << std::endl;
        std::cout << "meth_data[3] = " << meth_data[3] << std::endl;
        std::cout << "meth_data[4] = " << meth_data[4] << std::endl;
        std::cout << "meth_data[5] = " << meth_data[5] << std::endl;
    }*/
    double te = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        std::cout <<"reading methylation data took " << te - ts << " seconds."<< std::endl;
}

/*
std::vector < double* > data::perm_luts(){

    std::vector < double* > luts;

    luts.push_back(perm_lut_01);
    luts.push_back(perm_lut_02);
    luts.push_back(perm_lut_03);

    for (int i=0; i<1024; i++){
        if (perm_lut_03[i] == 0.0)
        {
            if (perm_lut_01[i] == 1.0)
            {
                perm_lut_04[i] = 1.0;
                perm_lut_05[i] = 2.0;
                perm_lut_06[i] = 0.0;
            }
            else
            {
                perm_lut_04[i] = 0.0;
                perm_lut_05[i] = 0.0;
                perm_lut_06[i] = 0.0;
            }
        }
        else if (perm_lut_03[i] == 1.0)
        {
            perm_lut_04[i] = 2.0;
            perm_lut_05[i] = 1.0;
            perm_lut_06[i] = 2.0;
        }
        else if (perm_lut_03[i] == 2.0)
        {
            perm_lut_04[i] = 0.0;
            perm_lut_05[i] = 0.0;
            perm_lut_06[i] = 1.0;
        }
    }

    luts.push_back(perm_lut_04);
    luts.push_back(perm_lut_05);
    luts.push_back(perm_lut_06);

    return luts;

}
*/

/*
std::vector < double* > data::perm_dotp_luts(){

    std::vector < double* > luts;

    luts.push_back(perm_dotp_lut_ab_01);

    for (int i=0; i<2048;){
        if (perm_dotp_lut_ab_03[i] == 0.0)
        {
            if (perm_dotp_lut_ab_01[i] == 1.0)
            {
                perm_dotp_lut_ab_02[i] = 2.0;
                perm_dotp_lut_ab_04[i] = 1.0;
                perm_dotp_lut_ab_05[i] = 2.0;
                perm_dotp_lut_ab_06[i] = 0.0;
            }
            else
            {
                perm_dotp_lut_ab_02[i] = 0.0;
                perm_dotp_lut_ab_04[i] = 0.0;
                perm_dotp_lut_ab_05[i] = 0.0;
                perm_dotp_lut_ab_06[i] = 0.0;
            }
        }
        else if (perm_dotp_lut_ab_03[i] == 1.0)
        {
            perm_dotp_lut_ab_02[i] = 0.0;
            perm_dotp_lut_ab_04[i] = 2.0;
            perm_dotp_lut_ab_05[i] = 1.0;
            perm_dotp_lut_ab_06[i] = 2.0;
        }
        else if (perm_dotp_lut_ab_03[i] == 2.0)
        {
            perm_dotp_lut_ab_02[i] = 1.0;
            perm_dotp_lut_ab_04[i] = 0.0;
            perm_dotp_lut_ab_05[i] = 0.0;
            perm_dotp_lut_ab_06[i] = 1.0;
        }
        if (i % 4 == 3)
            i += 5;
        else
            i++;
    }

    luts.push_back(perm_dotp_lut_ab_02);
    luts.push_back(perm_dotp_lut_ab_03);
    luts.push_back(perm_dotp_lut_ab_04);
    luts.push_back(perm_dotp_lut_ab_05);
    luts.push_back(perm_dotp_lut_ab_06);

    return luts;
}

*/

void data::SinkhornKnopp(std::vector<double> &xL, std::vector<double> &xR, double err_thr, int max_iter){

    std::vector<double> err_vec1 = Ax(xR.data(), 0); // we don't use normalization here
    std::vector<double> err_vec1abs = err_vec1;
    for (int i=0; i<N; i++){
        //err_vec1[i] *= (*xL)[i];
        err_vec1abs[i] *= xL[i];
        err_vec1abs[i] -= Mtotal; 
        err_vec1abs[i] = std::abs(err_vec1abs[i]);
    }
    double max_value1 = *(std::max_element(err_vec1abs.begin(), err_vec1abs.end()));
    if (rank == 0)
        std::cout << "begin max_value1 = " << max_value1 << std::endl;

    std::vector<double> err_vec2 = ATx(xL.data(), 0); // we don't use normalization here
    std::vector<double> err_vec2abs = err_vec2;
    for (int j=0; j<M; j++){
        err_vec2abs[j] *= xR[j];
        err_vec2abs[j] -= N; 
        err_vec2abs[j] = std::abs(err_vec2abs[j]);
    }
    double max_value2 = *(std::max_element(err_vec2abs.begin(), err_vec2abs.end()));
    double max_value2_total;
    MPI_Allreduce(&max_value2, &max_value2_total, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0)
        std::cout << "begin max_value2_total = " << max_value2_total << std::endl;

    int iter_count = 0;
    while (max_value1 > err_thr * Mtotal || max_value2_total > err_thr * N)
    {
        for (int j=0; j<M; j++){
            xR[j] = N / err_vec2[j];
        }
        
        // updating err_vec taking into account newly updated xR
        err_vec1 = Ax(xR.data(), 0);
        err_vec1abs = err_vec1;
        for (int i=0; i<N; i++){
            err_vec1abs[i] *= xL[i];
            err_vec1abs[i] -= Mtotal; 
            err_vec1abs[i] = std::abs(err_vec1abs[i]);
        }

        max_value1 = *(std::max_element(err_vec1abs.begin(), std::next(err_vec1abs.begin(), N-1)));
        if (rank == 0)
            std::cout << "iter = " << iter_count << ", max_value1 = " << max_value1 << std::endl;

        for (int i=0; i<N; i++){
            xL[i] = Mtotal / err_vec1[i];
        }

        // updating max_value2
        err_vec2 = ATx(xL.data(), 0);
        err_vec2abs = err_vec2;
        for (int j=0; j<M; j++){
            err_vec2abs[j] *= xR[j];
            err_vec2abs[j] -= N; 
            err_vec2abs[j] = std::abs(err_vec2abs[j]);
        }
        max_value2 = *(std::max_element(err_vec2abs.begin(), err_vec2abs.end()));
        MPI_Allreduce(&max_value2, &max_value2_total, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        if (rank == 0)
            std::cout << "iter = " << iter_count << ", max_value2_total = " << max_value2_total << std::endl;
        
        iter_count++;

        if (iter_count == max_iter)
            break;
    }
    if (rank == 0)
        std::cout << "Sinkhorn - Knopp preprocessing finished after " << iter_count << " / " << max_iter << " iterations." << std::endl;

    // the function returns the left and right vector that is used in normalization
    for (int i = 0; i < N; i++)
        xL[i] = sqrt( xL[i] ); 
    for (int j = 0; j < M; j++)
        xR[j] = sqrt( xR[j] ); 
}

 std::vector<double> data::filter_pheno(){
    std::vector<double> y = get_phen();
    std::vector<unsigned char> mask4 = get_mask4();
    int im4 = get_im4();
    for (int j=0; j<im4; j++) 
        for (int k=0; k<4; k++) 
            if (4*j + k < N)
                if (na_lut[mask4[j] * 4 + k] == 0)
                    y[4*j + k] = 0;
    return y;
 }

 double data::largest_sing_val2(){
    
    double start_power_meth = MPI_Wtime();

    double power_meth_err_thr = 1e-5;
    int power_meth_maxiter = 80; 
    int it = 0;
    std::vector<double> res = simulate(M, std::vector<double> {1.0/M}, std::vector<double> {1});
    std::vector<double> res_temp(4*get_mbytes(), 0.0);
    std::vector<double> res_prev;
    double lambda=0, lambda_prev;
    for (; it<power_meth_maxiter; it++){
        res_temp = Ax(res.data(), normal_data);
        res_prev = res;
        res = ATx(res_temp.data(), normal_data);
        lambda_prev = lambda;
        lambda = inner_prod(res, res_prev, 1);
        // std::cout << "[largest_sing_val2] it = " << it << " & lambda = " << lambda << std::endl;
        if (std::abs(lambda-lambda_prev) / std::abs(lambda_prev) < power_meth_err_thr)
            break;
        double norm = sqrt(l2_norm2(res, 1));
        for (int i=0; i<M; i++)
            res[i] /= norm;
    }

    double end_power_meth = MPI_Wtime();
    if (rank == 0){
        std::cout << "Power method did " << it << " / " << power_meth_maxiter << " iterations which took " << end_power_meth - start_power_meth << " seconds." << std::endl;
        std::cout << "Largest eigenvalue of X^TX " << lambda << std::endl;
    }

    return lambda;
 }