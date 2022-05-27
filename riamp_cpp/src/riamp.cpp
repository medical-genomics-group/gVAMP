#include <iostream>
#include <fstream>
#include <omp.h>
#include "utilities.hpp" // contains check_malloc()
#include "riamp.hpp"
#pragma once


void riamp::infere(const int Nit, priorMarkerDistr *pdistr, double *y, double delta, int M) {

    // const int Nit -> maximum number of iterations of RI-AMP to perform
    // priorMarkerDistr *pdistr -> assumed probability distribution on signal effect sizes
    // double *y -> vector of phenotype / trait values
    // double delta -> undersampling ratio, delta = N / M
    // int M -> number of markers in the study

    double* kappas = (double*) _mm_malloc( size_t( 2*(Nit + 1) ) * sizeof(double), 32);
    check_malloc(kappas, __LINE__, __FILE__);
    //calculation of the first 2(Nit + 1) even free cumulants
    kappas = ...

    //initialization of Sigma, Omega and mu 
    double var_beta = 0;
    for (int i0 = 0; i0 < pdistr.L; i0++) {
        var_beta+= pdistr.probs[i0] * pdistr.vars[i0];
    }
    double Sigma = var_beta * kappas(1);
    
    double var_y = 0, mean_y = 0;
    for (int i0 = 0; i0 < M; i0++) {
        var_y += y[i0]*y[i0];
        mean_y += y[i0];
    }
    mean_y /= M;
    var_y /= M;
    double Omega = delta * ( kappas(1) * var_y + kappas(2) *  var_beta * mean_y * mean_y );
    double mu1 = delta * kappas(1);

    for (int k = 0; k < Nit; k++){




        
    }




    //freeing the memory
    _mm_free(kappas);


}

// EO, review!!
double riamp::dot_product(const int mloc, double* __restrict__ phen, const double mu, const double sigma_inv) {

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
        p42 = _mm512_broadcast_f64x4(_mm256_load_pd(&phen[j * 4]));
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



// Setup processing: load input files and define MPI task workload
void riamp::setup_processing() {

    //finding out the number of bytes needed for storing genotype of a person
    // if 4 | N, then take N / 4, otherwise take N /4 + 
    mbytes = ( N %  4) ? (size_t) N /  4 + 1 : (size_t) N /  4;

    //loading genotype
    load_genotype();

    //reading phenotypes
    pmgr.read_phen_files(opt, get_N(), get_M());

    //checking if everything was ok
    check_processing_setup();

    if (rank == 0)
        printf("INFO   : output directory: %s\n", opt.get_out_dir().c_str());

    MPI_Barrier(MPI_COMM_WORLD);
    double ts = MPI_Wtime();
    pmgr.compute_markers_statistics(bed_data, get_N(), get_M(), mbytes);
    MPI_Barrier(MPI_COMM_WORLD);
    double te = MPI_Wtime();
    if (rank == 0)
        printf("INFO   : Time to compute the markers' statistics: %.2f seconds.\n", te - ts);

    //if (opt.predict()) return;
    
    read_group_index_file(opt.get_group_index_file());

    for (int i=0; i<Mt; i++) {
        mtotgrp.at(get_marker_group(i)) += 1;
    }
    //for (int i=0; i<G; i++)
    //    printf("mtotgrp at %d = %d\n", i, mtotgrp.at(i));
}


void riamp::check_openmp() {
#ifdef _OPENMP
#pragma omp parallel
    {
        if (rank == 0) {
            int nt = omp_get_num_threads();
            if (omp_get_thread_num() == 0)
                printf("INFO   : OMP parallel regions will use %d thread(s)\n", nt);
        }
    }
#else
    printf("WARNING: no OpenMP support!\n");
#endif
}


void riamp::read_group_index_file(const std::string& file) {

    std::ifstream infile(file.c_str());
    if (! infile)
        throw ("Error: can not open the group file [" + file + "] to read. Use the --group-index-file option!");

    if (rank == 0)
        std::cout << "INFO   : Reading groups from " + file + "." << std::endl;

    std::string label;
    int group;

    group_index.clear();

    while (infile >> label >> group) {
        //std::cout << label << " - " << group << std::endl;
        if (group > G) {
            printf("FATAL  : group index file contains a value that exceeds the number of groups given in group mixture file.\n");
            printf("       : check the consistency between your group index and mixture input files.\n");
            exit(1);
        }
        group_index.push_back(group);
    }
}


void riamp::check_processing_setup() {
    for (const auto& phen : pmgr.get_phens()) {
        const int Np = phen.get_nas() + phen.get_nonas();
        if (Np != N) {
            std::cout << "Fatal: N = " << N << " while phen file " << phen.get_filepath() << " has " << Np << " individuals!" << std::endl;
            exit(1);
        }
    }
}

//loading .bed matrix 
//dependencies: struct Options, ompi.h
void riamp::load_genotype() {

    double ts = MPI_Wtime();

    MPI_File bedfh;
    const std::string bedfp = opt.get_bed_file();
    check_mpi( MPI_File_open(MPI_COMM_WORLD, bedfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &bedfh),  __LINE__, __FILE__ );

    //Total size of the genome data we are reading ( M * N * 4 * sizeof(unsigned char) )
    const size_t size_bytes = size_t(M) * size_t(mbytes) * sizeof(unsigned char);

    bed_data = (unsigned char*)_mm_malloc(size_bytes, 64);
    check_malloc(bed_data, __LINE__, __FILE__);
    printf("INFO   : rank %4d has allocated %zu bytes (%.3f GB) for raw data.\n", rank, size_bytes, double(size_bytes) / 1.0E9);

    // Offset to section of bed file to be processed by task
    //MPI_Offset is an integer type of size sufficient to represent the size (in bytes) of the largest file supprted by MPI
    MPI_Offset offset = size_t(3) + size_t(S) * size_t(mbytes) * sizeof(unsigned char);

    // Gather the sizes to determine common number of reads
    size_t max_size_bytes = 0;
    check_mpi( MPI_Allreduce(&size_bytes, &max_size_bytes, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__ );

    const int NREADS = check_int_overflow(size_t( ceil(double(max_size_bytes)/double(INT_MAX/2)) ), __LINE__, __FILE__);
    size_t bytes = 0;
    mpi_file_read_at_all <unsigned char*> (size_bytes, offset, bedfh, MPI_UNSIGNED_CHAR, NREADS, bed_data, bytes);
    //@@@MPI_Barrier(MPI_COMM_WORLD);

    check_mpi(MPI_File_close(&bedfh), __LINE__, __FILE__);

    double te = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank >= 0)
        printf("INFO   : time to load genotype data = %.2f seconds.\n", te - ts);

}


//just distributing block of markers to different workers
void riamp::set_block_of_markers() {

    const int modu = Mt % nranks;
    const int size = Mt / nranks;

    Mm = Mt % nranks != 0 ? size + 1 : size;

    int len[nranks], start[nranks];
    int cum = 0;
    for (int i=0; i<nranks; i++) {
        len[i]  = i < modu ? size + 1 : size;
        start[i] = cum;
        cum += len[i];
    }
    //printf("cum %d vs %d  Mm = %d\n", cum, Mt, Mm);
    assert(cum == Mt);

    M = len[rank];
    S = start[rank];

    printf("INFO   : rank %4d has %d markers over tot Mt = %d, max Mm = %d, starting at S = %d\n", rank, M, Mt, Mm, S);
    //@todo: mpi check sum over tasks == Mt
}


void riamp::print_cva() {
    printf("INFO   : mixtures for all groups:\n");
    for (int i=0; i<G; i++) {
        printf("         grp %2d: ", i);
        for (int j=0; j<K; j++) {
            printf("%7.5f ", cva[i][j]);
        }
        printf("\n");
    }
}

void riamp::print_cvai() {
    printf("INFO   : inverse mixtures for all groups:\n");
    for (int i=0; i<G; i++) {
        printf("         grp %2d: ", i);
        for (int j=0; j<K; j++) {
            printf("%10.3f ", cvai[i][j]);
        }
        printf("\n");
    }
}