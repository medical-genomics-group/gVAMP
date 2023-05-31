#include "data.hpp"
#include <cmath> 
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <regex>
#include <filesystem>
#include <cassert> // contains assert
#include <mpi.h>
#include <omp.h>
#include "dotp_lut.hpp"
#include "na_lut.hpp"
#include "utilities.hpp"
#include <immintrin.h>
#include <bits/stdc++.h>
#include <boost/math/distributions/students_t.hpp> // contains Student's t distribution needed for pvals calculation




//******************
//  CONSTRUCTORS 
//******************

// -> DESCRIPTION:
//
//      constructor that is given a file pointer to the phenotype and genotype file
//
data::data(std::string fp, std::string genofp, const int N, const int M, const int Mt, const int S, const int rank, std::string type_data, double alpha_scale, std::string bimfp) :
    phenfp(fp),
    bimfp(bimfp),
    type_data(type_data),
    N(N),
    M(M),
    Mt(Mt),
    S(S),
    rank(rank),
    alpha_scale(alpha_scale),
    mbytes(( N % 4 ) ? (size_t) N / 4 + 1 : (size_t) N / 4),
    im4(N%4 == 0 ? N/4 : N/4+1) {
    mave = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(mave, __LINE__, __FILE__);
    msig = (double*) _mm_malloc(size_t(M) * sizeof(double), 32);
    check_malloc(msig, __LINE__, __FILE__);

    if (type_data == "bed"){
        // in the constructor we load phen, read genotype data and compute marker statistics
        bedfp = genofp;
        read_phen();
        read_genotype_data();
        compute_markers_statistics();
    } 
    else if(type_data == "meth"){
        methfp = genofp;
        read_phen();
        read_methylation_data();
        compute_markers_statistics();
    }
    
}


// -> DESCRIPTION:
//
//      constructor that is given a file pointer to the genotype file, but phenotype values are assigned for a vector provided. 
//      All values of phenotype are assume to be non-NA, usually used in simulation settings.
//
data::data(std::vector<double> y, std::string genofp, const int N, const int M, const int Mt, const int S, const int rank, std::string type_data, double alpha_scale, std::string bimfp) :
    type_data(type_data),
    bimfp(bimfp),
    N(N),
    M(M),
    Mt(Mt),
    S(S),
    rank(rank),
    phen_data(y),
    alpha_scale(alpha_scale),
    mbytes(( N % 4 ) ? (size_t) N / 4 + 1 : (size_t) N / 4),
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
        read_genotype_data();
        compute_markers_statistics();
    }
    else if (type_data == "meth"){
        methfp = genofp;
        read_methylation_data();
        compute_markers_statistics();
    }
    
}





//**************************
// DATA LOADING PROCEDURES
//**************************

// -> DESCRIPTION:
//
//      Read phenotype file assuming PLINK format:
//      Family ID, Individual ID, Phenotype; One row per individual
// 
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

        // fail if the input size doesn not match the size of the read data
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




// -> DESCRIPTION:
//
//      Read genotype file assuming PLINK .bed format
// 
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


// -> DESCRIPTION:
//
//      Reads methylation design matrix assuming matrix of doubles stores in binary format
// 
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
    
    double te = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0)
        std::cout <<"reading methylation data took " << te - ts << " seconds."<< std::endl;

}



// -> DESCRIPTION:
//
//      Reads covariates files, used with a probit model
// 
void data::read_covariates(std::string covfp, int C){ // values should be separate with space delimiter

    if (C==0)
        return;

    double start = MPI_Wtime();

    std::ifstream covf(covfp);
    std::string line; 
    //std::regex re("\\S+");
    std::regex re("\\s+");

    while (std::getline(covf, line)) // read the current line
    {
        int Cobs = 0;
        std::vector<double> entries;
        std::sregex_token_iterator iter(line.begin(), line.end(), re, -1);
        std::sregex_token_iterator re_end;

        for ( ; iter != re_end; ++iter){
            entries.push_back(std::stod(*iter));
            Cobs++;
        }
            
        /*
        for (auto it = std::sregex_iterator(line.begin(), line.end(), re); it != std::sregex_iterator(); it++) {
            std::string token = (*it).str();
            entries.push_back(std::stod(token));
            Cobs++;      
        }   
        */

        if (Cobs != C){
            std::cout << "FATAL: number of covariates = " << Cobs << " does not match to the specified number of covariates = " << C << std::endl;
            exit(EXIT_FAILURE);
        }

        covs.push_back(entries);        
    }

    double end = MPI_Wtime();

    if (rank == 0)
        std::cout << "rank = " << rank << ": reading covariates took " << end - start << " seconds to run." << std::endl;

}

// -> DESCRIPTION:
//
//      a function that reads .bim file and returns an integer vector of length M (number of markers accessible to a single worker)
//      whose elements specify the chromosome on which certain marker is present
//
// -> INPUT:
//
//       [string] bim_file - name of bim file to be loaded
//
// -> OUTPUT:  
//
//      [vector<int>] chroms - chromosome indices corresponding to markers assigned to a worker 
//
std::vector<int> data::read_chromosome_info(std::string bim_file){
    
    //initializing the output vector
    std::vector<int> chroms(M);

    // specifying bim file and reggex expression
    std::ifstream infile(bim_file);
    std::string line;
    std::regex re("\\s+");

    if (infile.is_open()) {
        int line_n = 0;
        while (getline(infile, line)) {
            if (line_n >= S && line_n < S + M){
                std::sregex_token_iterator first{line.begin(), line.end(), re, -1}, last;
                std::vector<std::string> tokens{first, last};
                // first line of .bim file contains information on the chromosome index
                chroms.push_back( atof( tokens[0].c_str() ) );
            }
            line_n += 1;
        }
        infile.close();
    } else {
        std::cout << "FATAL: could not open bim file: " << bim_file << std::endl;
        exit(EXIT_FAILURE);
    }

    return chroms;
}



// -> DESCRIPTION:
//
//      Compute mean and associated standard deviation for markers
//      for each of the phenotypes (stats are NA dependent)
//      ! one byte of bed  contains information for 4 individuals
//      ! one byte of phen contains information for 8 individuals  
//      whose elements specify the chromosome on which certain marker is present
//
void data::compute_markers_statistics() {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const std::vector<unsigned char> mask4 = get_mask4();
    const int im4 = get_im4();
    double* mave = get_mave();
    double* msig = get_msig();

    double start = MPI_Wtime();

    // distinction between genomic and methylation data
    if (type_data == "bed"){

        // MANual VECTorization is used if computer architecture allows it
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
                        }
                    }
                    if (sumb != 0)
                        mave[i] = suma / sumb;
                    else
                        mave[i] = 0.0;

                    double sumsqr = 0.0;
                    for (int j=0; j<im4; j++) {
                        for (int k=0; k<4; k++) {
                            double val = (dotp_lut_a[bedm[j] * 4 + k] - mave[i]) * dotp_lut_b[bedm[j] * 4 + k] * na_lut[mask4[j] * 4 + k];
                            // calculate the value and filter for nans in genotype and phenotype
                            sumsqr += val * val;
                        }
                    }
                    if (sumsqr != 0)
                        // we scale inverse standard deviation to the exponent alpha (usually 0, 0.3, 1)
                        if (alpha_scale == 1.0)
                            msig[i] = 1.0 / sqrt(sumsqr / (double( get_nonas() ) - 1.0));
                        else
                            msig[i] = 1.0 / pow( sqrt(sumsqr / (double( get_nonas() ) - 1.0)), alpha_scale );
                    else 
                        // in case entire column contains only zero-elements
                        msig[i] = 1.0;
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

                    if (sumsqr != 0.0)
                        if (alpha_scale == 1.0)
                            msig[i] = 1.0 / sqrt(sumsqr / (double( get_nonas() ) - 1.0));
                        else
                            msig[i] = 1.0 / pow( sqrt(sumsqr / (double( get_nonas() ) - 1.0)), alpha_scale );
                    else 
                        msig[i] = 1.0;
                }
    }
        double end = MPI_Wtime();
        if (rank == 0)
            std::cout << "rank = " << rank << ": statistics took " << end - start << " seconds to run." << std::endl;
}


// -> DESCRIPTION:
//
//      Compute mean and associated standard deviation for people
//      !one byte of bed  contains information for 4 individuals
//      ! one byte of phen contains information for 8 individuals
//      we modify vectors of length 4*mbytes to contain to contain 
//      average value and inverse standard deviation per individual
//      (0 if phenotype is missing)
//
void data::compute_people_statistics() {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    const std::vector<unsigned char> mask4 = get_mask4();
    const int im4 = get_im4();
    double* mave = get_mave();
    double* msig = get_msig();

    mave_people = std::vector<double> (4 * mbytes, 0.0);
    msig_people = std::vector<double> (4 * mbytes, 0.0);
    numb_people = std::vector<double> (4 * mbytes, 0.0);

    double start = MPI_Wtime();

    if (type_data == "bed"){

        for (int i=0; i<M; i++) {

            size_t bedix = size_t(i) * size_t(mbytes);
            const unsigned char* bedm = &bed_data[bedix];

            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (int j=0; j<im4; j++) {

                for (int k=0; k<4; k++) {

                    double value = (dotp_lut_a[bedm[j] * 4 + k] - mave[i]) * msig[i] * dotp_lut_b[bedm[j] * 4 + k] * na_lut[mask4[j] * 4 + k];

                    mave_people[4*j + k] += value; 
                    numb_people[4*j + k] += dotp_lut_b[bedm[j] * 4 + k] * na_lut[mask4[j] * 4 + k];
                    msig_people[4*j + k] += value * value;

                }
            }
        }


        // collecting results from different nodes
        std::vector<double> mave_people_total(4 * mbytes, 0.0);
        std::vector<double> msig_people_total(4 * mbytes, 0.0);
        std::vector<double> numb_people_total(4 * mbytes, 0.0);

        MPI_Allreduce(mave_people.data(), mave_people_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(numb_people.data(), numb_people_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(msig_people.data(), msig_people_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (int j=0; j<im4; j++)

            for (int k=0; k<4; k++)

                if (4*j + k < N)

                    if (na_lut[mask4[j] * 4 + k] == 1){

                        mave_people_total[4*j+k] /= numb_people_total[4*j+k];
                        msig_people_total[4*j+k] = (numb_people_total[4*j+k] - 1) / (msig_people_total[4*j+k] - numb_people_total[4*j+k] * mave_people_total[4*j+k] * mave_people_total[4*j+k]);
                    }
                    else {

                        mave_people_total[4*j+k] = 0;
                        msig_people_total[4*j+k] = 0;
                    }
        
        mave_people = mave_people_total;
        msig_people = msig_people_total;
        numb_people = numb_people_total;

        for (int i=0; i<N; i++) 
            msig_people[i] = sqrt(msig_people[i]);  

    }
    else if (type_data == "meth"){
        
        int im4m1 = im4-1;
        double* mave = get_mave();
        double* msig = get_msig();

        for (int i=0; i<M; i++) {

            size_t methix = size_t(i) * size_t(N);
            const double* methm = &meth_data[methix];

            // calculating marker mean in 2 step process since size(phen) = 4*mbytes and size(methm) = N
            // currently only non-missing methylation data is allowed 
            #ifdef _OPENMP
            #pragma omp parallel for
            #endif
            for (int j=0; j<(im4-1); j++) {

                for (int k=0; k<4; k++){

                    double value = (methm[4*j + k] - mave[i]) * msig[i] * na_lut[mask4[j] * 4 + k]; // only non-missing methylation data is allowed 

                    mave_people[4*j + k] += value; 
                    numb_people[4*j + k] += na_lut[mask4[j] * 4 + k]; // only non-missing methylation data is allowed 
                    msig_people[4*j + k] += value * value;

                }
            }  

            for (int k=0; k<4; k++) {

                if (4*im4m1 + k < N){

                    double value = (methm[4*im4m1 + k] - mave[i]) * msig[i];

                    mave_people[4*im4m1 + k] += value * na_lut[mask4[im4m1] * 4 + k]; 
                    numb_people[4*im4m1 + k] += na_lut[mask4[im4m1] * 4 + k];
                    msig_people[4*im4m1 + k] += value * value * na_lut[mask4[im4m1] * 4 + k];

                } 
            }
        }

        // collecting results from different nodes
        std::vector<double> mave_people_total(4 * mbytes, 0.0);
        std::vector<double> msig_people_total(4 * mbytes, 0.0);
        std::vector<double> numb_people_total(4 * mbytes, 0.0);

        MPI_Allreduce(mave_people.data(), mave_people_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(numb_people.data(), numb_people_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(msig_people.data(), msig_people_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for (int j=0; j<im4; j++)

            for (int k=0; k<4; k++)

                if (4*j + k < N)

                    if (na_lut[mask4[j] * 4 + k] == 1){

                        mave_people_total[4*j+k] /= numb_people_total[4*j+k];
                        msig_people_total[4*j+k] = (numb_people_total[4*j+k] - 1) / (msig_people_total[4*j+k] - numb_people_total[4*j+k] * mave_people_total[4*j+k] * mave_people_total[4*j+k]);
                    }
                    else {

                        mave_people_total[4*j+k] = 0;
                        msig_people_total[4*j+k] = 0;
                    }

        mave_people = mave_people_total;
        msig_people = msig_people_total;
        numb_people = numb_people_total;

        for (int i=0; i<N; i++) 
            msig_people[i] = sqrt(msig_people[i]);   

    }

    double end = MPI_Wtime();

    if (rank == 0)
        std::cout << "rank = " << rank << ": statistics took " << end - start << " seconds to run." << std::endl;

}


//**************************************************************
// PROCEDURES IMPLEMENTING OPERATIONS INVOLVING DESIGN MATRIX A
//**************************************************************


// -> DESCRIPTION:
//
//      Computes < phen, (A_mloc - mu) * sigma_inv > from 4*SB to 4*(SB+LB)
//
    double data::dot_product(const int mloc, double* __restrict__ phen, const double mu, const double sigma_inv, const int SB, const int LB) {
    // __restrict__ means that phen is the only pointer pointing to that data

    if (type_data == "bed"){

        unsigned char* bed = &bed_data[mloc * mbytes];

        #ifdef MANVECT

            __m256d luta; // lutb, lutna;
            __m512d lutab, p42;
            __m256d suma = _mm256_set1_pd(0.0);
            __m512d sum42 = _mm512_set1_pd(0.0);
            #ifdef _OPENMP
            //#pragma omp parallel for schedule(static) reduction(addpd4:suma,sumb)
            #pragma omp parallel for schedule(static) private(luta,lutab,p42) reduction(addpd8:sum42)
            #endif
            //for (int j=0; j<mbytes; j++){
            for (int j=SB; j<SB+LB; j++){
                lutab = _mm512_load_pd(&dotp_lut_ab[bed[j] * 8]);
                // broadcasts the 4 packed double-precision (64-bit) floating-point elements
                //p42 = _mm512_broadcast_f64x4(_mm256_load_pd(&phen[j * 4])); 
                p42 = _mm512_broadcast_f64x4(_mm256_load_pd(&phen[(j-SB) * 4])); 
                p42 = _mm512_mul_pd(p42, lutab);
                sum42 = _mm512_add_pd(sum42, p42);
            }
            return sigma_inv * (sum42[0] + sum42[1] + sum42[2] + sum42[3] - mu * (sum42[4] + sum42[5] + sum42[6] + sum42[7]));

        #else

            double dpa = 0.0;
            double dpb = 0.0;

            #ifdef _OPENMP
            #pragma omp parallel for schedule(static) reduction(+:dpa,dpb)
            #endif
            //for (int i=0; i<mbytes; i++) {
            for (int i=SB; i<SB+LB; i++) {

                #ifdef _OPENMP
                #pragma omp simd aligned(dotp_lut_a,dotp_lut_b,phen:32)
                #endif
                    for (int j=0; j<4; j++) {

                        dpa += dotp_lut_a[bed[i] * 4 + j] * phen[(i-SB) * 4 + j];
                        dpb += dotp_lut_b[bed[i] * 4 + j] * phen[(i-SB) * 4 + j];

                    }
                    
            }
            return sigma_inv * (dpa - mu * dpb);

        #endif
    }
    else if (type_data == "meth"){ // always takes all individuals as their number is typically small in methylation studies

        double* meth = &meth_data[mloc * N];

            double dpa = 0.0;

            #ifdef _OPENMP
            #pragma omp parallel for simd schedule(static) reduction(+:dpa)
            #endif
            for (int i=0; i<N; i++)
                dpa += (meth[i] - mu) * phen[i];

            return sigma_inv * dpa;

        }

    return std::numeric_limits<double>::max();

}


// -> DESCRIPTION:
//
//      Computes A^T phen using dot_product function above.
//      There are 2 versions: one that takes full data and one that
//      takes a subset of individuals.
//
std::vector<double> data::ATx(double* __restrict__ phen) {
    return ATx(phen, 0, mbytes);
}

std::vector<double> data::ATx(double* __restrict__ phen, int SB, int LB) {

    std::vector<double> ATx(M, 0.0);

    #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
    #endif
    for (int mloc=0; mloc < M; mloc++)
         ATx[mloc] = dot_product(mloc, phen, mave[mloc], msig[mloc], SB, LB);


    // for stability reasons we scale design matrix A with 1/sqrt(number of people)
    double scale;
    if (SB == 0 && LB == mbytes)
        scale = 1.0 / sqrt(N);
    else
        scale = 1.0 / sqrt(4 * LB);

    for (int mloc=0; mloc<M; mloc++)
        ATx[mloc] *= scale;

    return ATx;
}





// -> DESCRIPTION:
//
//      Computes A * phen using dot_product function above.
//      There are 2 versions: one that takes full data and one that
//      takes a subset of individuals.
//

std::vector<double> data::Ax(double* __restrict__ phen) {
    return Ax(phen, 0, mbytes);
}

std::vector<double> data::Ax(double* __restrict__ phen, int SB, int LB) {

    if (type_data == "bed"){
    #ifdef MANVECT

        //if (rank == 0)
        //    std::cout << "MANCVECT is active!" << std::endl;
        double* mave = get_mave();
        double* msig = get_msig();
        // std::vector<double> Ax_temp(4 * mbytes, 0.0);
        std::vector<double> Ax_temp(4 * LB, 0.0);

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
            //for (int j=0; j<mbytes; j++) {
            for (int j=SB; j<SB+LB; j++) {
                luta = _mm256_load_pd(&dotp_lut_a[bedm[j] * 4]);
                lutb = _mm256_load_pd(&dotp_lut_b[bedm[j] * 4]);
                luta = _mm256_add_pd(luta, minave);
                luta = _mm256_mul_pd(luta, sig);
                luta = _mm256_mul_pd(luta, lutb); // because we need to multiply - mu / sigma with 0
                suma = _mm256_load_pd(&Ax_temp[j * 4]);
                suma = _mm256_fmadd_pd(luta, mmphen, suma);
                //_mm256_storeu_pd(&Ax_temp[j * 4], suma);
                _mm256_storeu_pd(&Ax_temp[(j-SB) * 4], suma);
            }
            // #ifdef _OPENMP2
            //}
            // #endif  
        }


        // std::vector<double> Ax_total(4 * mbytes, 0.0);
        std::vector<double> Ax_total(4 * LB, 0.0);

        // MPI_Allreduce(Ax_temp.data(), Ax_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(Ax_temp.data(), Ax_total.data(), 4*LB, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        
        double scale;
        if (SB == 0 && LB == mbytes)
            scale = 1.0 / sqrt(N);
        else
            scale = 1.0 / sqrt(4 * LB);

        for (int i=0; i<4*LB; i++)
            Ax_total[i] *= scale;

        //for(int i = 0; i < 4 * mbytes; i++)
        //    Ax_total[i] /= sqrt(N);

        return Ax_total;

    #else

        // std::vector<double> Ax_temp(4 * mbytes, 0.0);
        std::vector<double> Ax_temp(4 * LB, 0.0);
        unsigned char* bed;


        for (int i=0; i<M; i++){

            bed = &bed_data[i * mbytes];

            double ave = mave[i];
            double phen_i = phen[i];
            double sig_phen_i = msig[i] * phen_i;

            #ifdef _OPENMP
                #pragma omp parallel for shared(Ax_temp)
            #endif
            for (int j=SB; j<SB+LB; j++) {
            // for (int j=0; j<mbytes; j++) {

                #ifdef _OPENMP
                #pragma omp simd aligned(dotp_lut_a,dotp_lut_b:32)
                #endif
                for (int k=0; k<4; k++) {

                    // because msig[i] is the precision of the i-th marker
                    //Ax_temp[j * 4 + k] += (dotp_lut_a[bed[j] * 4 + k] - ave) * sig_phen_i * dotp_lut_b[bed[j] * 4 + k] * na_lut[mask4[j] * 4 + k]; 
                    Ax_temp[(j-SB) * 4 + k] += (dotp_lut_a[bed[j] * 4 + k] - ave) * sig_phen_i * dotp_lut_b[bed[j] * 4 + k] * na_lut[mask4[j] * 4 + k];

                    /*
                    if ((j==0 && ave==0) || (j==0 && i==0)){
                        if (rank == 0){
                            std::cout << "i = " << i << std::endl;
                            std::cout << "dotp_lut_a[bed[j] * 4 + k] = " << dotp_lut_a[bed[j] * 4 + k] << std::endl;
                            std::cout << "ave = " << ave << std::endl;
                            std::cout << "sig_phen_i = " << sig_phen_i << std::endl;
                            std::cout << "dotp_lut_b[bed[j] * 4 + k]  = " << dotp_lut_b[bed[j] * 4 + k]  << std::endl;
                            std::cout << "na_lut[mask4[j] * 4 + k] = " << na_lut[mask4[j] * 4 + k] << std::endl;
                            std::cout << "Ax_temp[(j-SB) * 4 + k] = " << Ax_temp[(j-SB) * 4 + k] << std::endl;
                        }
                    } */
                }
            } 
        }

        // collecting results from different nodes
        // std::vector<double> Ax_total(4 * mbytes, 0.0);
        std::vector<double> Ax_total(4 * LB, 0.0);

        // MPI_Allreduce(Ax_temp.data(), Ax_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(Ax_temp.data(), Ax_total.data(), 4*LB, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);


        double scale;
        if (SB == 0 && LB == mbytes)
            scale = 1.0 / sqrt(N);
        else
            scale = 1.0 / sqrt(4 * LB);

        for (int i=0; i<4*LB; i++)
            Ax_total[i] *= scale;

        return Ax_total;

    #endif

    }

    else if (type_data == "meth"){ // always takes all individuals as their number is typically small in methylation studies

        double* mave = get_mave();
        double* msig = get_msig();
        double* meth;
        std::vector<double> Ax_temp(4 * mbytes, 0.0);    

        for (int i=0; i<M; i++){

            meth = &meth_data[i * N];

            double ave = mave[i];
            double sig_phen_i = msig[i] * phen[i];
            
            #ifdef _OPENMP
                #pragma omp parallel for simd shared(Ax_temp)
            #endif
            for (int j=0; j<N; j++)
                Ax_temp[j] += (meth[j] - ave) * sig_phen_i;

        }

        // collecting results from different nodes
        std::vector<double> Ax_total(4 * mbytes, 0.0);

        MPI_Allreduce(Ax_temp.data(), Ax_total.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        for(int i = 0; i < N; i++)
            Ax_total[i] /= sqrt(N);

        return Ax_total;

    }

    return std::vector<double>(4 * mbytes, 0.0);
}

std::vector<double> data::Zx(std::vector<double> phen){
    
    std::vector<double> Zx_temp(4 * mbytes, 0.0);

    for (int i=0; i<N; i++)
        Zx_temp[i] = inner_prod(covs[i], phen, 0);

    return Zx_temp;
}


//****************************
// FILTERING PHENOTYPE VALUES
//*****************************

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

std::vector<double> data::filter_pheno(int *nonnan){

    *nonnan = 0;
    std::vector<double> y = get_phen();
    std::vector<unsigned char> mask4 = get_mask4();
    int im4 = get_im4();

    for (int j=0; j<im4; j++) 
        for (int k=0; k<4; k++) 
            if (4*j + k < N)
                if (na_lut[mask4[j] * 4 + k] == 0)
                    y[4*j + k] = 0;
                else
                    (*nonnan)++;
    return y;

}


 
//*****************************************************
// CALCULATION OF p-values using LOO and LOCO APPROACH
//*****************************************************

// finding p-values from t-test on regression coefficient = 0 
// in leave-one-out setting, i.e. y - A_{-k}x_{-k} = alpha_k * x_k, H0: alpha_k = 0
// Supports only full - individual version of the algorithm.
std::vector<double> data::pvals_calc(std::vector<double> z1, std::vector<double> y, std::vector<double> x1_hat, std::string filepath){
    std::vector<double> pvals(M, 0.0);
    // phenotypic values corrected for genetic predictors
    std::vector<double> y_mod = std::vector<double> (4*mbytes, 0.0);
    for (int i=0; i<N; i++)
        y_mod[i] = y[i] - z1[i];
    double* mave = get_mave();
    double* msig = get_msig();

    if (type_data == "bed"){

        for (int k=0; k<M; k++){

            if (rank == 0 && k % 1000 == 0)
                std::cout << "so far worker 0 has calculated " << k << " pvals." << std::endl;

            // phenotype values after marker specific correction
            std::vector<double> y_mark = std::vector<double> (4*mbytes, 0.0);

            for (int i=0; i<N; i++)
                y_mark[i] = y_mod[i];

            unsigned char* bed = &bed_data[k * mbytes];
            double sumx = 0, sumsqx = 0, sumxy = 0, sumy = 0, sumsqy = 0;
            int count = 0;
            
            #ifdef _OPENMP
            #pragma omp parallel for reduction(+ : sumx, sumsqx, sumxy, sumy, sumsqy, count)
            #endif
            for (int i=0; i<mbytes; i++) {

                #ifdef _OPENMP
                #pragma omp simd aligned(dotp_lut_a,dotp_lut_b:32)
                #endif
                for (int j=0; j<4; j++) 
                    y_mark[4*i+j] += (dotp_lut_a[bed[i] * 4 + j] - mave[k]) * msig[k] * dotp_lut_b[bed[i] * 4 + j] * na_lut[mask4[i] * 4 + j] * x1_hat[k] / sqrt(N);

                //#ifdef _OPENMP
                //#pragma omp simd aligned(dotp_lut_a,dotp_lut_b:32) reduction(+:sumx, sumsqx, sumxy)
                //#endif
                for (int j=0; j<4; j++) {

                    double value = (dotp_lut_a[bed[i] * 4 + j] - mave[k]) * msig[k] * dotp_lut_b[bed[i] * 4 + j] * na_lut[mask4[i] * 4 + j];

                    sumx += value;
                    sumsqx += value * value;
                    sumxy += value * y_mark[4*i+j];   
                    sumy += y_mark[4*i+j] * dotp_lut_b[bed[i] * 4 + j] * na_lut[mask4[i] * 4 + j];
                    sumsqy += y_mark[4*i+j] * y_mark[4*i+j] * dotp_lut_b[bed[i] * 4 + j] * na_lut[mask4[i] * 4 + j];
                    count += dotp_lut_b[bed[i] * 4 + j] * na_lut[mask4[i] * 4 + j];

                }
            }

            pvals[k] = linear_reg1d_pvals(sumx, sumsqx, sumxy, sumy, sumsqy, count);
        }  
        mpi_store_vec_to_file(filepath, pvals, S, M);
    }
    else if (type_data == "meth"){ // add normalization to expressions & take into account scaling & account for missing data points & reduction 
        // not yet properly implemented
        for (int k=0; k<M; k++){

            std::vector<double> y_mark = y_mod;
            double* meth = &meth_data[k * N];

            double sumx = 0, sumsqx = 0, sumxy = 0;
            #ifdef _OPENMP
                #pragma omp simd
            #endif
            for (int i=0; i<N; i++) 
                y_mark[i] += meth[i] * x1_hat[k];
            
            #ifdef _OPENMP
                #pragma omp simd reduction(+:sumx, sumsqx, sumxy)
            #endif
            for (int i=0; i<N; i++) {
                sumx += meth[i];
                sumsqx += meth[i]*meth[i];
                sumxy += meth[i]*y_mark[i];
            }

            sumx = 0.0;
            for (int i=0; i<N; i++) {
                sumx += meth[i];
            }
            //pvals[k] = linear_reg1d_pvals(sumx, sumsqx, sumxy, y_mark, N);
        }
    }

    return pvals;
}



// finding p-values from t-test on regression coefficient = 0 
// in leave-one-out setting, i.e. y - A_{-S}x_{-S} = alpha_k * x_k, H0: alpha_k = 0,
// S contains markers from all the other chromosome except the one that contains marker
// with index k.
// !! Currently implemented only for .bed type of data !!
std::vector<double> data::pvals_calc_LOCO(std::vector<double> z1, std::vector<double> y, std::vector<double> x1_hat, std::string filepath){

    std::vector<double> pvals(M, 0.0);
    // phenotypic values corrected for genetic predictors
    std::vector<double> y_mod = std::vector<double> (4*mbytes, 0.0);
    for (int i=0; i<N; i++)
        y_mod[i] = y[i] - z1[i];

    // fetch mean and inverse standard deviation values per markers
    double* mave = get_mave();
    double* msig = get_msig();

    // phenotype values after chromosome specific correction
    std::vector<double> y_chrom = std::vector<double> (4*mbytes, 0.0);

    if (rank == 0)
        std::cout << "before reading chr data" << std::endl;
    std::vector<int> ch_info = read_chromosome_info(bimfp);
    if (rank == 0)
        std::cout << "after reading chr data" << std::endl;

    // we iterate over 22 non-sex chromosomes in humans
    for (int ch=1; ch<=22; ch++){

        std::vector<double> y_chrom_tmp = std::vector<double> (4*mbytes, 0.0);

        for (int m=0; m<M; m++){    

            if (ch_info[m] == ch){

                unsigned char* bed = &bed_data[m * mbytes];

                for (int i=0; i<mbytes; i++) {

                    #ifdef _OPENMP
                    #pragma omp simd aligned(dotp_lut_a,dotp_lut_b:32)
                    #endif
                    for (int j=0; j<4; j++) 
                        y_chrom_tmp[4*i+j] += (dotp_lut_a[bed[i] * 4 + j] - mave[m]) * msig[m] * dotp_lut_b[bed[i] * 4 + j] * na_lut[mask4[i] * 4 + j] * x1_hat[m] / sqrt(N);
                }
            }
        }
    
        MPI_Allreduce(y_chrom_tmp.data(), y_chrom.data(), N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        std::transform (y_chrom.begin(), y_chrom.end(), y_mod.begin(), y_chrom.begin(), std::plus<double>());
    
        for (int m=0; m<M; m++){    
            if (ch_info[m] == ch){

                unsigned char* bed = &bed_data[m * mbytes];
                double sumx = 0, sumsqx = 0, sumxy = 0, sumy = 0, sumsqy = 0;
                int count = 0;
                
                #ifdef _OPENMP
                #pragma omp parallel for reduction(+ : sumx, sumsqx, sumxy, sumy, sumsqy, count)
                #endif
                for (int i=0; i<mbytes; i++) {

                    for (int j=0; j<4; j++) {

                        double value = (dotp_lut_a[bed[i] * 4 + j] - mave[m]) * msig[m] * dotp_lut_b[bed[i] * 4 + j] * na_lut[mask4[i] * 4 + j];

                        sumx += value;
                        sumsqx += value * value;
                        sumxy += value * y_chrom[4*i+j];   
                        sumy += y_chrom[4*i+j] * dotp_lut_b[bed[i] * 4 + j] * na_lut[mask4[i] * 4 + j];
                        sumsqy += y_chrom[4*i+j] * y_chrom[4*i+j] * dotp_lut_b[bed[i] * 4 + j] * na_lut[mask4[i] * 4 + j];
                        count += dotp_lut_b[bed[i] * 4 + j] * na_lut[mask4[i] * 4 + j];

                    }
                }

                pvals[m] = linear_reg1d_pvals(sumx, sumsqx, sumxy, sumy, sumsqy, count);   
            }
        }
    }

    mpi_store_vec_to_file(filepath, pvals, S, M);

    return pvals;
}