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



//constructor for class data
data::data(std::string fp, std::string bedfp, const int N, const int M, const int Mt, const int S, const int normal, const int rank) :
    phenfp(fp),
    bedfp(bedfp),
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

    // in the constructor we load phen, read genotype data and compute marker statistics
    read_phen();
    read_genotype_data();
    compute_markers_statistics();

    if (normal == 2){
        double SK_err_thr = 1e-6;
        int SK_max_iter = 50;
        xR = std::vector<double> (M, 1.0);
        xL = std::vector<double> (4*mbytes, 1.0);
        //std::cout << "before SK!" << std::endl;
        SinkhornKnopp(xL, xR, SK_err_thr, SK_max_iter);
    }

    // std::cout << "after SK!" << std::endl;

    // perm_idxs = std::vector<int>(M, 1);
    // p_luts = perm_luts();
    // p_dotp_luts = perm_dotp_luts();
}

//constructor for class data
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
            if (m4 == 0)  mask4.push_back(0b00001111);

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
            if (line_n % 4 == 0 && line_n > 3 && line_n < 20) {
                    //std::cout << tokens[1] << std::endl;
                    //std::cout << tokens[2] << std::endl;
                    //std::cout << "mask4[" << int(line_n / 4) - 1 << "] = " << unsigned(mask4.at(int(line_n / 4) - 1)) << std::endl;
            }
        }
        infile.close();

        //std::cout << "nas + nonas = " << nas + nonas << ", N = " << N << std::endl;
        assert(nas + nonas == N);

        // Set last bits to 0 if ninds % 4 != 0
        const int m4 = line_n % 4;
        if (m4 != 0) {
            for (int i=m4; i<4; i++) {
                mask4.at(int(line_n / 4)) &= ~(0b1 << i);
            }
            //printf("line_n = %d\n", line_n);
            //printf("last byte starts for indiv %d\n", int(N/4)*4);
            //printf("set up to indiv %d\n", int(N/4 + 1) * 4);
            std::cout << "Setting last " << 4 - m4 << " bits to NAs" << std::endl;
            //std::cout << "fatal: missing implementation" << std::endl;
            //exit(1);
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
        scale = sqn;

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
                /*
                    if (luta[0] != 0 && luta[0] != 1 && luta[0] != 2 && j == 111 && rank == 0){
                        std::cout << "!!luta[0] neq = " << luta[0] << ", j = " << j << ", rank = " << rank << std::endl;
                        std::cout << "!!perm_idxs[i]-1 = " << perm_idxs[i]-1 << ", bedm[j] * 4 = " << bedm[j] * 4 << std::endl;
                        std::cout << "(p_luts[5])[1020] = " << (p_luts[5])[1020] << std::endl;
                        std::cout << "*(p_luts[5]) = " << *(p_luts[5]) << std::endl;
                        std::cout << "*(p_luts[5]+1) = " << *(p_luts[5]+1) << std::endl;
                        std::cout << "*(p_luts[5] + 1020) = " << *(p_luts[5] + 1020)<< std::endl;
                    }
                */
                /*
                if (rank == 0 && i == 110){
                    std::cout << "rank = " << rank << ", perm_idxs[i]-1 = " << perm_idxs[i]-1<<  ", i = " << i << std::endl; 
                    std::cout << "suma[0] = " << suma[0] << ", suma[1] = " << suma[1] << ", suma[2] = " << suma[2] << ", suma[3] = " << suma[3] << std::endl;
                    std::cout << "luta[0] = " << luta[0] << ", luta[1] = " << luta[1] << ", luta[2] = " << luta[2] << ", luta[3] = " << luta[3] << std::endl;
                }
                */
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
                //if (i<10)
                //    printf("marker %d: %20.15f +/- %20.15f, %20.15f / %20.15f\n", i, mave[i], msig[i], asum, bsum);
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
                        //suma += dotp_lut_a[bedm[j] * 4 + k] * 1; // we pretend that all phenotypes are present in the data
                        //sumb += dotp_lut_b[bedm[j] * 4 + k] * 1;
                    }
                }
                mave[i] = suma / sumb;
                double sumsqr = 0.0;
                for (int j=0; j<im4; j++) {
                    for (int k=0; k<4; k++) {
                        double val = (dotp_lut_a[bedm[j] * 4 + k] - mave[i]) * dotp_lut_b[bedm[j] * 4 + k] * na_lut[mask4[j] * 4 + k];
                        //double val = (p_luts[perm_idxs[i]-1][bedm[j] * 4 + k] - mave[i]) * dotp_lut_b[bedm[j] * 4 + k] * na_lut[mask4[j] * 4 + k];
                        //double val = (dotp_lut_a[bedm[j] * 4 + k] - mave[i]) * dotp_lut_b[bedm[j] * 4 + k] * 1; // we pretend that all phenotypes are present in the data
                        // calculate the value and filter for nans in genotype and phenotype
                        sumsqr += val * val;
                    }
                }
                msig[i] = 1.0 / sqrt(sumsqr / (double( get_nonas() ) - 1.0));
                //printf("marker %8d: %20.15f +/- %20.15f\n", i, mave[i], msig[i]);
            }

        #endif
        double end = MPI_Wtime();
        if (rank == 0)
            std::cout << "statistics took " << end - start << " seconds to run." << std::endl;
}

double data::dot_product(const int mloc, double* __restrict__ phen, const double mu, const double sigma_inv, int normal) {
// __restrict__ means that phen is the only pointer pointing to that data

    unsigned char* bed = &bed_data[mloc * mbytes];

    #ifdef MANVECT
        if (normal == 1){
            __m256d luta; // lutb, lutna;
            __m512d lutab, p42;
            // __m256d p4   = _mm256_set1_pd(0.0);
            __m256d suma = _mm256_set1_pd(0.0);
            // __m256d sumb = _mm256_set1_pd(0.0);
            __m512d sum42 = _mm512_set1_pd(0.0);
            #ifdef _OPENMP
            //#pragma omp parallel for schedule(static) reduction(addpd4:suma,sumb)
            #pragma omp parallel for schedule(static) private(luta,lutab,p42) reduction(addpd8:sum42)
            #endif
            for (int j=0; j<mbytes; j++){
                //luta  = _mm256_load_pd(&dotp_lut_a[bed[j] * 4]);
                //lutb  = _mm256_load_pd(&dotp_lut_b[bed[j] * 4]);
                //luta  = _mm256_load_pd(&dotp_lut_ab[bed[j] * 8]);
                //lutb  = _mm256_load_pd(&dotp_lut_ab[bed[j] * 8 + 4]);
                lutab = _mm512_load_pd(&dotp_lut_ab[bed[j] * 8]);
                //lutab = _mm512_load_pd(&p_dotp_luts[perm_idxs[mloc]-1][bed[j] * 8]);
                //p4    = _mm256_load_pd(&phen[j * 4]);
                p42 = _mm512_broadcast_f64x4(_mm256_load_pd(&phen[j * 4])); // broadcasts the 4 packed double-precision (64-bit) floating-point elements
                ////lutna = _mm256_load_pd(&na_lut[mask4[j] * 4]); // phen = 0.0 on NAs!
                //luta  = _mm256_mul_pd(luta, p4);
                //lutb  = _mm256_mul_pd(lutb, p4);
                p42 = _mm512_mul_pd(p42, lutab);
                //suma  = _mm256_add_pd(suma, luta);
                //sumb  = _mm256_add_pd(sumb, lutb);
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
                /*if (j < 1){
                    std::cout << "after loading luta" << std::endl;
                    std::cout << "phen[0] = " << phen[0] << std::endl;
                    std::cout << "phen[1] = " << phen[1] << std::endl;
                    std::cout << "phen[2] = " << phen[2] << std::endl;
                    std::cout << "phen[3] = " << phen[3] << std::endl;
                    std::cout << "phen[4] = " << phen[4] << std::endl;
                    std::cout << "phen[5] = " << phen[5] << std::endl;
                    std::cout << "phen[11999] = " << phen[11999] << std::endl;
                    std::cout << "j = " << j << std::endl;
                } */
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
                if (j<=3 && mloc<=0)
                    std::cout << "mloc=" << mloc << ", j = " << j << ", phen[j]=" << phen[j] << std::endl;
                p4 = _mm256_load_pd(&phen[j * 4]); 
                if (j<=3 && mloc<=0)
                    std::cout << "after loading phen in VAMP ATx"<< std::endl;
                mmxL =  _mm256_load_pd(&xL[j * 4]);
                p4 = _mm256_mul_pd(p4, mmxL);
                luta  = _mm256_mul_pd(luta, p4);
                suma  = _mm256_add_pd(suma, luta);
            }
            return (xR[mloc] * (suma[0] + suma[1] + suma[2] + suma[3]));
        }
        //return sigma_inv *
        //    (suma[0] + suma[1] + suma[2] + suma[3] - mu * (sumb[0] + sumb[1] + sumb[2] + sumb[3]));

        return std::numeric_limits<double>::max();

    #else

        if (normal == 1){
            double dpa = 0.0;
            double dpb = 0.0;

            #ifdef _OPENMP
                //#pragma omp parallel for schedule(static) reduction(addpd4:suma,sumb)
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

        return std::numeric_limits<double>::max();

    #endif
}


std::vector<double> data::Ax(double* __restrict__ phen, int normal) {

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
                //if ( (i < 5 || i == M-1) && j == 0 && rank == 0)
                //    std::cout << "suma[0] = " << suma[0] << ", rank = " << rank << std::endl;
                _mm256_storeu_pd(&Ax_temp[j * 4], suma);
                //if ( (i < 5 || i == M-1) && j == 0 && rank == 0)
                //    std::cout << "Ax_temp[0] = " << Ax_temp[0] << ", rank = " << rank << std::endl;
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
            __m256d xR_mmphen = _mm256_set1_pd(xR[i] * phen[i]); // xR should already containt sqrt of a diag vector
            // std::cout << "i = " << i << ", before starting inner loop!" << std::endl;
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

    if (normal == 2){
        for(int i = 0; i < 4 * mbytes; i++)
            Ax_total[i] *= xL[i];  

        /*
        __m256d GxRvec, mmxL, xLGxRvec;
        for (int j=0; j<mbytes; j++) {
            GxRvec = _mm256_load_pd(&Ax_total[j * 4]);
            std::cout << "j = " << j << ", after GxRvec" << std::endl;
            mmxL = _mm256_load_pd(&xL[j * 4]);
            std::cout << "after mmxL" << std::endl;
            xLGxRvec = _mm256_mul_pd(mmxL, GxRvec);
            std::cout << "after xLGxRvec" << std::endl;
            _mm256_storeu_pd(&Ax_total[j * 4], xLGxRvec);
        }
        */
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
            
            #ifdef _OPENMP
                #pragma omp parallel for shared(Ax_temp)
            #endif
            
            for (int j=0; j<mbytes; j++) {
                for (int k=0; k<4; k++) {
                    Ax_temp[j * 4 + k] += (dotp_lut_a[bed[j] * 4 + k] - ave) * sig * phen_i * dotp_lut_b[bed[j] * 4 + k]; // because msig[i] is the precision of the i-th marker
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
                for (int k=0; k<4; k++) {
                    Ax_temp[j * 4 + k] += dotp_lut_a[bed[j] * 4 + k] * phen_i;
                }
            } 
        }
    }
    else if (normal == 2){
        for (int i=0; i<M; i++){
            bed = &bed_data[i * mbytes];
            double xRphen_i = xR[i] * phen[i];
            
            #ifdef _OPENMP
                #pragma omp parallel for shared(Ax_temp)
            #endif
            for (int j=0; j<mbytes; j++) {
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

    std::vector<double> Ax_total(4 * mbytes, 0.0);
    MPI_Allreduce(Ax_temp.data(), Ax_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    if (normal == 1)
        for(int i = 0; i < 4 * mbytes; i++)
            Ax_total[i] /= sqrt(N);
    return Ax_total;

#endif
}


std::vector<double> data::ATx(double* __restrict__ phen, int normal) {

    std::vector<double> ATx(M, 0.0);
    #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
    #endif
    for (int mloc=0; mloc < M; mloc++){
        // ATx[mloc] = dot_product( mloc, phen, mave[mloc], msig[mloc], normal ) / sqrt(N); // we scale elements of A with 1/sqrt(N)
        ATx[mloc] = dot_product(mloc, phen, mave[mloc], msig[mloc], normal);
    }
    // std::cout << "final ATx[0] = " << ATx[0] << std::endl;
    if (normal == 1)
        for (int mloc=0; mloc < M; mloc++)
            ATx[mloc] /= sqrt(N);
    return ATx;
}

void data::read_genotype_data(){

    double ts = MPI_Wtime();
    MPI_File bedfh;
    //const std::string bedfp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/1kg_chr20_genotypes.bed";
    //const std::string phenfp = "/nfs/scistore13/robingrp/human_data/adepope_preprocessing/VAMPJune2022/1kg_chr20_genotypes.txt";
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
    //std::cout << "begin err_vec1[10160] = " << err_vec1[10160] << std::endl;
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

        // for (int i = N; i <4*mbytes; i++)
        //    std::cout << "i = " << i << "err_vec1[i] = " << err_vec1[i] << std::endl;

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

    for (int i = 0; i < N; i++)
        xL[i] = sqrt( xL[i] ); 
    for (int j = 0; j < M; j++)
        xR[j] = sqrt( xR[j] ); // the function returns the left and right vector that is used in normalization

    /*
    std::cout << "xL[0] = " << xL[0] << std::endl;
    std::cout << "xL[1] = " << xL[1] << std::endl;
    std::cout << "xL[2] = " << xL[2] << std::endl;

    std::cout << "xR[0] = " << xR[0] << std::endl;
    std::cout << "xR[1] = " << xR[1] << std::endl;
    std::cout << "xR[2] = " << xR[2] << std::endl;
    */
}