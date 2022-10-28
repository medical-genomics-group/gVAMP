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
#include "perm_lut.hpp"
#include "utilities.hpp"
#include <immintrin.h>
#include <bits/stdc++.h>

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



//constructor for class data
data::data(std::string fp, std::string bedfp, const int N, const int M, const int Mt, const int S, const int rank) :
    phenfp(fp),
    bedfp(bedfp),
    N(N),
    M(M),
    S(S),
    rank(rank),
    mbytes(( N % 4 ) ? (size_t) N / 4 + 1 : (size_t) N / 4),
    im4(N%4 == 0 ? N/4 : N/4+1) {
    mave = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(mave, __LINE__, __FILE__);
    msig = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(msig, __LINE__, __FILE__);
    perm_idxs = std::vector<int>(M, 1);
}

//constructor for class data
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
}

double data::dot_product(const int mloc, double* __restrict__ phen, const double mu, const double sigma_inv) {

    unsigned char* bed = &bed_data[mloc * mbytes];

    #ifdef MANVEC
        __m256d luta, lutb; //, lutna;
        __m512d lutab, p42;
        __m256d p4   = _mm256_set1_pd(0.0);
        __m256d suma = _mm256_set1_pd(0.0);
        __m256d sumb = _mm256_set1_pd(0.0);
        __m512d sum42 = _mm512_set1_pd(0.0);

        #ifdef _OPENMP
            //#pragma omp parallel for schedule(static) reduction(addpd4:suma,sumb)
        #pragma omp parallel for schedule(static) reduction(addpd8:sum42)
        #endif
        for (int j=0; j<mbytes; j++) {
            //luta  = _mm256_load_pd(&dotp_lut_a[bed[j] * 4]);
            //lutb  = _mm256_load_pd(&dotp_lut_b[bed[j] * 4]);
            //luta  = _mm256_load_pd(&dotp_lut_ab[bed[j] * 8]);
            //lutb  = _mm256_load_pd(&dotp_lut_ab[bed[j] * 8 + 4]);
            lutab = _mm512_load_pd(&dotp_lut_ab[bed[j] * 8]);
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

        //return sigma_inv *
        //    (suma[0] + suma[1] + suma[2] + suma[3] - mu * (sumb[0] + sumb[1] + sumb[2] + sumb[3]));
        return sigma_inv *
            (sum42[0] + sum42[1] + sum42[2] + sum42[3] - mu * (sum42[4] + sum42[5] + sum42[6] + sum42[7]));

    #else

        double dpa = 0.0;
        double dpb = 0.0;

        #ifdef _OPENMP
            //#pragma omp parallel for schedule(static) reduction(addpd4:suma,sumb)
        #pragma omp parallel for schedule(static) reduction(+:dpa,dpb)
        #endif
        for (int i=0; i<mbytes; i++) {
        #ifdef _OPENMP
        #pragma omp simd aligned(dotp_lut_a,dotp_lut_b,phen:32)
        #endif
            for (int j=0; j<4; j++) {
                dpa += dotp_lut_a[bed[i] * 4 + j] * phen[i * 4 + j];
                dpb += dotp_lut_b[bed[i] * 4 + j] * phen[i * 4 + j];
            }
        }

        return sigma_inv * (dpa - mu * dpb);

    #endif

}


std::vector<double> data::Ax(double* __restrict__ phen) {

 #ifdef MANVECT

    //if (rank == 0)
        //std::cout << "MANCVECT is active!" << std::endl;

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
    
    #ifdef _OPENMP2

    if (rank == 0)
        std::cout << "_OPENMP2 is defined!" << std::endl;

    #pragma omp declare reduction(vec_add_d : std::vector<double> : std::transform(omp_in.begin(),omp_in.end(),omp_out.begin(),omp_out.begin(),std::plus<double>())) initializer (omp_priv=omp_orig)

    #pragma omp parallel
    {
        #pragma omp for nowait schedule(static) reduction(vec_add_d:Ax_temp)

    #endif

    for (int i=0; i<M; i++){
        unsigned char* bedm = &bed_data[i * mbytes];

        __m256d luta, lutb;
        __m256d minave = _mm256_set1_pd(-mave[i]);
        __m256d sig = _mm256_set1_pd(msig[i]);
        __m256d suma = _mm256_set1_pd(0.0); // setting all 4 values to 0.0
        __m256d mmphen = _mm256_set1_pd(phen[i]);
        #ifdef _OPENMP2

        //#pragma omp parallel private(suma, luta, lutb) shared(Ax_temp)
        //{
            //#pragma omp for nowait schedule(static) // reduction(vec_add_d:Ax_temp)
        
        #endif
            for (int j=0; j<mbytes; j++) {
                luta = _mm256_load_pd(&dotp_lut_a[bedm[j] * 4]);
                lutb = _mm256_load_pd(&dotp_lut_b[bedm[j] * 4]);
                luta = _mm256_add_pd(luta, minave);
                luta = _mm256_mul_pd(luta, sig);
                luta = _mm256_mul_pd(luta, lutb);
                suma = _mm256_load_pd(&Ax_temp[j * 4]);
                suma = _mm256_fmadd_pd(luta, mmphen, suma);
                //if ( (i < 5 || i == M-1) && j == 0 && rank == 0)
                //    std::cout << "suma[0] = " << suma[0] << ", rank = " << rank << std::endl;
                _mm256_storeu_pd(&Ax_temp[j * 4], suma);
                //if ( (i < 5 || i == M-1) && j == 0 && rank == 0)
                //    std::cout << "Ax_temp[0] = " << Ax_temp[0] << ", rank = " << rank << std::endl;
            }
        #ifdef _OPENMP2
        //}
        #endif
    }

    #ifdef _OPENMP2
    }
    #endif

    std::vector<double> Ax_total(4 * mbytes, 0.0);
    MPI_Allreduce(Ax_temp.data(), Ax_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for(int i = 0; i < 4 * mbytes; i++)
        Ax_total[i] /= sqrt(N);

    //std::cout << "dividing by sqrt(N) where N = " << N << " is finished." << std::endl;
    
    //double sqrtNm1 = 1/sqrt(N);
    //for_each(Ax_total.begin(), Ax_total.end(), [sqrtNm1](double &c){ c *= sqrtNm1; });
    
    //Ax_total = std::accumulate(begin(vars), end(vars), sqrtNm1 , std::multiplies<double>());
    //foreach( double val, Ax_total ) val / sqrtN;

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
    //shared(

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
                //if ( k == 0 && (i < 5 || i == M-1) && j == 0 && rank == 0 )
                //    std::cout << "Ax_temp[0] = " << Ax_temp[0] << ", rank = "<< rank << std::endl;
            }
        } 
    }
    
    //std::cout << "Ax[1] = " << Ax[1] << ", rank = " << rank << std::endl; 
    std::vector<double> Ax_total(4 * mbytes, 0.0 );
    //std::cout << "Ax_total[1] = " << Ax_total[1]<< ", rank = "<< rank << std::endl;
    //double suma = 0;
    //for(int i = 0; i < 4 * mbytes; i++){   
        //if (i % 100 == 0)
            //std::cout << "suma = " << suma << std::endl;
        //suma = 0;
        //if (i == N-1 ||i == 0)
        //    std::cout << "before MPI all reduce, rank =" << rank << std::endl;
        //MPI_Allreduce(&Ax_temp[i], &suma, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        //if (i == N-1 || i == 0)
        //    std::cout << "after MPI all reduce, rank =" << rank << std::endl;
        //Ax_total[i] = suma / sqrt(N); // we scale elements of A with 1 / sqrt(N)
        //if (i == N-1 || i == 0)
        //    std::cout << "Ax_total[i], rank = " << rank << std::endl;
    //}
    MPI_Allreduce(Ax_temp.data(), Ax_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    for(int i = 0; i < 4 * mbytes; i++)
        Ax_total[i] /= sqrt(N);

    return Ax_total;

    #endif
}



std::vector<double> data::ATx(double* __restrict__ phen) {

    //if (rank == 0)
        //std::cout << "inside ATx" << std::endl;
    std::vector<double> ATx(M, 0.0);

    //unsigned char* bed;
    #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
    #endif
    for (int mloc=0; mloc < M; mloc++){
        ATx[mloc] = dot_product( mloc, phen, mave[mloc], msig[mloc] ) / sqrt(N); // we scale elements of A with 1/sqrt(N)
        //if (std::isnan(std::abs(ATx[mloc])))
            //std::cout << "ATx nan index is " << mloc << ", rank = " << rank << std::endl;
    }
    //std::cout << "final ATx[0] = " << ATx[0] << std::endl;
    //std::cout << "std::isnan(ATx[0]) = " << std::isnan(ATx[0]) << std::endl;
    return ATx;
    /*
    std::vector<double> ATx_total(Mtotal, 0.0);
    for(int i = 0; i < Mtotal; i++)
    {   
        double val = 0;
        MPI_Allreduce(&ATx[i], &val, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        Ax_total[i] = val;
    }

    return ATx_total;
    */
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

std::vector < double* > data::perm_luts(){

    std::vector < double* > luts;

    luts.push_back(perm_lut_01);
    luts.push_back(perm_lut_02);
    luts.push_back(perm_lut_03);

    alignas(32) double perm_lut_04[1024], perm_lut_05[1024], perm_lut_06[1024];

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
