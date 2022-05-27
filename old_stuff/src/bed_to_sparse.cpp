#include "bed_to_sparse.hpp"
#include "hydra.h"
#include <mpi.h>
#include <chrono>
#include "data.hpp"
#include "options.hpp"


//EO: This method writes sparse data files out of a BED file
//    Note: will always convert the whole file
//    A two-step process due to RAM limitation for very large BED files:
//      1) Compute the number of ones and twos to be written by each task to compute
//         rank-wise offsets for writing the si1 and si2 files
//      2) Write files with global indexing
// ---------------------------------------------------------------------------------
void write_sparse_data_files(const uint bpr, const Data data, const Options opt) {

    int rank, nranks, result;

    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_File   bedfh, dimfh, si1fh, sl1fh, ss1fh, si2fh, sl2fh, ss2fh, simfh, slmfh, ssmfh;
    MPI_Offset offset;
    MPI_Status status;

    MPI_Barrier(MPI_COMM_WORLD);
    const auto st2 = std::chrono::high_resolution_clock::now();

    if (rank == 0) printf("INFO   : will generate sparse data files out of %d ranks and %d blocks per rank.\n", nranks, bpr);

    // Get dimensions of the dataset and define blocks
    // -----------------------------------------------
    const unsigned int N    = data.numInds;
    unsigned int       Mtot = data.numSnps;
    if (opt.numberMarkers) 
        Mtot = opt.numberMarkers;

    if (rank == 0) printf("INFO   : full dataset includes %d markers and %d individuals.\n", Mtot, N);

    // Fail if more blocks requested than available markers
    if (nranks * bpr > Mtot) {
        if (rank == 0)
            printf("Fatal: empty tasks defined. Useless and not allowed.\n      Requested %d tasks for %d markers to process.\n", nranks * bpr, Mtot);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Define global marker indexing
    // -----------------------------
    const uint nblocks = nranks * bpr;
    int MrankS[nblocks], MrankL[nblocks];
    define_blocks_of_markers(Mtot, MrankS, MrankL, nblocks);

    // Length of a column in bytes
    const size_t snpLenByt = (data.numInds % 4) ? data.numInds / 4 + 1 : data.numInds / 4;
    if (rank == 0) printf("INFO   : snpLenByt = %zu bytes.\n", snpLenByt);

    // Get bed file directory and basename
    std::string sparseOut = opt.get_sparse_output_filebase(rank);
    if (rank == 0)
        printf("INFO   : will write sparse output files as: %s.{ss1, sl1, si1, ss2, sl2, si2}\n", sparseOut.c_str());

    // Open bed file for reading
    std::string bedfp = opt.bedFile;
    bedfp += ".bed";
    check_mpi(MPI_File_open(MPI_COMM_WORLD, bedfp.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &bedfh), __LINE__, __FILE__);

    // Create sparse output files
    // --------------------------
    const std::string dim = sparseOut + ".dim";
    const std::string si1 = sparseOut + ".si1";
    const std::string sl1 = sparseOut + ".sl1";
    const std::string ss1 = sparseOut + ".ss1";
    const std::string si2 = sparseOut + ".si2";
    const std::string sl2 = sparseOut + ".sl2";
    const std::string ss2 = sparseOut + ".ss2";
    const std::string sim = sparseOut + ".sim";
    const std::string slm = sparseOut + ".slm";
    const std::string ssm = sparseOut + ".ssm";

    if (rank == 0) {
        MPI_File_delete(dim.c_str(), MPI_INFO_NULL);
        MPI_File_delete(si1.c_str(), MPI_INFO_NULL);
        MPI_File_delete(sl1.c_str(), MPI_INFO_NULL);
        MPI_File_delete(ss1.c_str(), MPI_INFO_NULL);
        MPI_File_delete(si2.c_str(), MPI_INFO_NULL);
        MPI_File_delete(sl2.c_str(), MPI_INFO_NULL);
        MPI_File_delete(ss2.c_str(), MPI_INFO_NULL);
        MPI_File_delete(sim.c_str(), MPI_INFO_NULL);
        MPI_File_delete(slm.c_str(), MPI_INFO_NULL);
        MPI_File_delete(ssm.c_str(), MPI_INFO_NULL);
    }

    MPI_Barrier(MPI_COMM_WORLD);

    check_mpi(MPI_File_open(MPI_COMM_WORLD, dim.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &dimfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, si1.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &si1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl1.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &sl1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss1.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &ss1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, si2.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &si2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sl2.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &sl2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ss2.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &ss2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, sim.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &simfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, slm.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &slmfh), __LINE__, __FILE__);
    check_mpi(MPI_File_open(MPI_COMM_WORLD, ssm.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY | MPI_MODE_EXCL, MPI_INFO_NULL, &ssmfh), __LINE__, __FILE__);

    // Write to dim file (as text)
    char buff[LENBUF];

    if (rank == 0) {
        int  left = snprintf(buff, LENBUF, "%d %d\n", N, Mtot);
        check_mpi(MPI_File_write_at(dimfh, 0, &buff, strlen(buff), MPI_CHAR, &status), __LINE__, __FILE__);
    }
    MPI_Barrier(MPI_COMM_WORLD);


    // STEP 1: compute rank-wise N1 and N2 (rN1 and rN2)
    // -------------------------------------------------
    size_t rN1 = 0, rN2 = 0, rNM = 0;
    size_t N1  = 0, N2  = 0, NM  = 0;

    for (int i=0; i<bpr; ++i) {

        if (rank == 0) printf("INFO   : reading (1/2) starting block %3d out of %3d\n", i+1, bpr);

        uint globi = rank*bpr + i;
        int  MLi   = MrankL[globi];
        int  MSi   = MrankS[globi];
        //printf("DEBUG  : 1| bpr %i  MLi = %d, MSi = %d\n", i, MLi, MSi);

        // Alloc memory for raw BED data
        const size_t rawdata_n = size_t(MLi) * size_t(snpLenByt) * sizeof(char);
        char* rawdata = (char*) _mm_malloc(rawdata_n, 64);  check_malloc(rawdata, __LINE__, __FILE__);

        // Gather sizes to determine common number of reads
        size_t rawdata_n_max = 0;
        check_mpi(MPI_Allreduce(&rawdata_n, &rawdata_n_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);

        int NREADS = check_int_overflow(size_t(ceil(double(rawdata_n_max)/double(INT_MAX/2))), __LINE__, __FILE__);
        if (NREADS <= 0) {
            if (rank == 0) printf("FATAL  : NREADS must be >= 1.");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Compute the offset of the section to read from the BED file
        offset = size_t(3) + size_t(MSi) * size_t(snpLenByt) * sizeof(char);

        // Read the bed data
        size_t bytes = 0;
        data.mpi_file_read_at_all <char*> (rawdata_n, offset, bedfh, MPI_CHAR, NREADS, rawdata, bytes);

        // Get number of ones, twos, and missing
        data.sparse_data_get_sizes_from_raw(rawdata, MLi, snpLenByt, 0, N1, N2, NM);
        //printf("DEBUG  : off rank %d: N1 = %15lu, N2 = %15lu, NM = %15lu\n", rank, N1, N2, NM);

        rN1 += N1;
        rN2 += N2;
        rNM += NM;

        _mm_free(rawdata);

        MPI_Barrier(MPI_COMM_WORLD);
    }


    // Gather offsets
    // --------------
    size_t *AllN1 = (size_t*)_mm_malloc(nranks * sizeof(size_t), 64);  check_malloc(AllN1, __LINE__, __FILE__);
    size_t *AllN2 = (size_t*)_mm_malloc(nranks * sizeof(size_t), 64);  check_malloc(AllN2, __LINE__, __FILE__);
    size_t *AllNM = (size_t*)_mm_malloc(nranks * sizeof(size_t), 64);  check_malloc(AllNM, __LINE__, __FILE__);

    check_mpi(MPI_Allgather(&rN1, 1, MPI_UNSIGNED_LONG_LONG, AllN1, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allgather(&rN2, 1, MPI_UNSIGNED_LONG_LONG, AllN2, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD), __LINE__, __FILE__);
    check_mpi(MPI_Allgather(&rNM, 1, MPI_UNSIGNED_LONG_LONG, AllNM, 1, MPI_UNSIGNED_LONG_LONG, MPI_COMM_WORLD), __LINE__, __FILE__);
    //printf("DEBUG  : rN1 = %lu and AllN1[0] = %lu\n", rN1, AllN1[0]);

    size_t N1tot = 0, N2tot = 0, NMtot = 0;
    for (int i=0; i<nranks; i++) {
        N1tot += AllN1[i];
        N2tot += AllN2[i];
        NMtot += AllNM[i];
    }
    if (rank ==0 ) printf("INFO   : N1tot = %lu, N2tot = %lu, NMtot = %lu.\n", N1tot, N2tot, NMtot);

    // STEP 2: write sparse structure files
    // ------------------------------------
    if (rank ==0 ) printf("\nINFO   : begining of step 2, writing of the sparse files.\n");
    size_t tN1 = 0, tN2 = 0, tNM = 0;

    for (int i=0; i<bpr; ++i) {

        if (rank == 0) printf("INFO   : reading (2/2) starting block %3d out of %3d\n", i+1, bpr);

        uint globi = rank*bpr + i;
        int  MLi   = MrankL[globi];
        int  MSi   = MrankS[globi];
        //printf("DEBUG  : 2| bpr %i  MLi = %d, MSi = %d\n", i, MLi, MSi);

        // Alloc memory for raw BED data
        const size_t rawdata_n = size_t(MLi) * size_t(snpLenByt) * sizeof(char);
        char* rawdata = (char*) _mm_malloc(rawdata_n, 64);  check_malloc(rawdata, __LINE__, __FILE__);

        // Gather sizes to determine common number of reads
        size_t rawdata_n_max = 0;
        check_mpi(MPI_Allreduce(&rawdata_n, &rawdata_n_max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);

        int NREADS = check_int_overflow(size_t(ceil(double(rawdata_n_max)/double(INT_MAX/2))), __LINE__, __FILE__);
        if (NREADS <= 0) {
            if (rank == 0) printf("FATAL  : NREADS must be >= 1.");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Compute the offset of the section to read from the BED file
        offset = size_t(3) + size_t(MSi) * size_t(snpLenByt) * sizeof(char);

        size_t bytes = 0;
        data.mpi_file_read_at_all<char*>(rawdata_n, offset, bedfh, MPI_CHAR, NREADS, rawdata, bytes);

        // Alloc memory for sparse representation
        size_t *N1S, *N1L, *N2S, *N2L, *NMS, *NML;
        N1S = (size_t*) _mm_malloc(size_t(MLi) * sizeof(size_t), 64);  check_malloc(N1S, __LINE__, __FILE__);
        N1L = (size_t*) _mm_malloc(size_t(MLi) * sizeof(size_t), 64);  check_malloc(N1L, __LINE__, __FILE__);
        N2S = (size_t*) _mm_malloc(size_t(MLi) * sizeof(size_t), 64);  check_malloc(N2S, __LINE__, __FILE__);
        N2L = (size_t*) _mm_malloc(size_t(MLi) * sizeof(size_t), 64);  check_malloc(N2L, __LINE__, __FILE__);
        NMS = (size_t*) _mm_malloc(size_t(MLi) * sizeof(size_t), 64);  check_malloc(NMS, __LINE__, __FILE__);
        NML = (size_t*) _mm_malloc(size_t(MLi) * sizeof(size_t), 64);  check_malloc(NML, __LINE__, __FILE__);

        size_t N1 = 0, N2 = 0, NM = 0;
        data.sparse_data_get_sizes_from_raw(rawdata, uint(MLi), snpLenByt, 0, N1, N2, NM);
        //printf("DEBUG  : N1 = %15lu, N2 = %15lu, NM = %15lu\n", N1, N2, NM);

        // Alloc and build sparse structure
        uint *I1, *I2, *IM;
        I1 = (uint*) _mm_malloc(N1 * sizeof(uint), 64);  check_malloc(I1, __LINE__, __FILE__);
        I2 = (uint*) _mm_malloc(N2 * sizeof(uint), 64);  check_malloc(I2, __LINE__, __FILE__);
        IM = (uint*) _mm_malloc(NM * sizeof(uint), 64);  check_malloc(IM, __LINE__, __FILE__);

        // To check that each element is properly set
        //for (int i=0; i<N1; i++) I1[i] = UINT_MAX;
        //for (int i=0; i<N2; i++) I2[i] = UINT_MAX;
        //for (int i=0; i<NM; i++) IM[i] = UINT_MAX;

        data.sparse_data_fill_indices(rawdata, MLi, snpLenByt, 0, N1S, N1L, I1,  N2S, N2L, I2,  NMS, NML, IM);

        //check_whole_array_was_set(I1, N1, __LINE__, __FILE__);
        //check_whole_array_was_set(I2, N2, __LINE__, __FILE__);
        //check_whole_array_was_set(IM, NM, __LINE__, __FILE__);

        // Compute the rank offset
        size_t N1Off = 0, N2Off = 0, NMOff = 0;
        for (int ii=0; ii<rank; ++ii) {
            N1Off += AllN1[ii];
            N2Off += AllN2[ii];
            NMOff += AllNM[ii];
        }

        N1Off += tN1;
        N2Off += tN2;
        NMOff += tNM;

        // ss1,2,m files must contain absolute start indices!
        for (int ii=0; ii<MLi; ++ii) {
            N1S[ii] += N1Off;
            N2S[ii] += N2Off;
            NMS[ii] += NMOff;
        }

        size_t N1max = 0, N2max = 0, NMmax = 0;
        check_mpi(MPI_Allreduce(&N1, &N1max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
        check_mpi(MPI_Allreduce(&N2, &N2max, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
        check_mpi(MPI_Allreduce(&NM, &NMmax, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
        int    MLimax = 0;
        check_mpi(MPI_Allreduce(&MLi, &MLimax, 1, MPI_INTEGER, MPI_MAX, MPI_COMM_WORLD), __LINE__, __FILE__);
        if (rank == 0) printf("INFO   : N1max = %lu, N2max = %lu, NMmax = %lu, MLimax = %d\n", N1max, N2max, NMmax, MLimax);

        int NREADS1   = check_int_overflow(size_t(ceil(double(N1max)/double(INT_MAX/2))),  __LINE__, __FILE__);
        int NREADS2   = check_int_overflow(size_t(ceil(double(N2max)/double(INT_MAX/2))),  __LINE__, __FILE__);
        int NREADSM   = check_int_overflow(size_t(ceil(double(NMmax)/double(INT_MAX/2))),  __LINE__, __FILE__);
        int NREADSMLi = check_int_overflow(size_t(ceil(double(MLimax)/double(INT_MAX/2))), __LINE__, __FILE__);
        if (rank == 0) printf("INFO   : NREADS1 = %d, NREADS2 = %d, NREADSM = %d, NREADSMLi = %d\n", NREADS1, NREADS2, NREADSM, NREADSMLi);

        // Sparse Ones files
        offset = N1Off * sizeof(uint);
        data.mpi_file_write_at_all <uint*>   (N1,  offset, si1fh, MPI_UNSIGNED,           NREADS1,   I1);
        offset = size_t(MSi) * sizeof(size_t);
        data.mpi_file_write_at_all <size_t*> (size_t(MLi), offset, sl1fh, MPI_UNSIGNED_LONG_LONG, NREADSMLi, N1L);
        data.mpi_file_write_at_all <size_t*> (size_t(MLi), offset, ss1fh, MPI_UNSIGNED_LONG_LONG, NREADSMLi, N1S);

        // Sparse Twos files
        offset = N2Off * sizeof(uint) ;
        data.mpi_file_write_at_all <uint*>   (N2,  offset, si2fh, MPI_UNSIGNED,           NREADS2,   I2);
        offset = size_t(MSi) * sizeof(size_t);
        data.mpi_file_write_at_all <size_t*> (size_t(MLi), offset, sl2fh, MPI_UNSIGNED_LONG_LONG, NREADSMLi, N2L);
        data.mpi_file_write_at_all <size_t*> (size_t(MLi), offset, ss2fh, MPI_UNSIGNED_LONG_LONG, NREADSMLi, N2S);

        // Sparse Missing files
        offset = NMOff * sizeof(uint) ;
        data.mpi_file_write_at_all <uint*>   (NM,  offset, simfh, MPI_UNSIGNED,           NREADSM,   IM);
        offset = size_t(MSi) * sizeof(size_t);
        data.mpi_file_write_at_all <size_t*> (size_t(MLi), offset, slmfh, MPI_UNSIGNED_LONG_LONG, NREADSMLi, NML);
        data.mpi_file_write_at_all <size_t*> (size_t(MLi), offset, ssmfh, MPI_UNSIGNED_LONG_LONG, NREADSMLi, NMS);

        // Free allocated memory
        _mm_free(rawdata);
        _mm_free(N1S); _mm_free(N1L); _mm_free(I1);
        _mm_free(N2S); _mm_free(N2L); _mm_free(I2);
        _mm_free(NMS); _mm_free(NML); _mm_free(IM);

        tN1 += N1;
        tN2 += N2;
        tNM += NM;
    }

    _mm_free(AllN1);
    _mm_free(AllN2);
    _mm_free(AllNM);

    // Sync
    MPI_Barrier(MPI_COMM_WORLD);

    // check size of the written files!
    check_file_size(si1fh, N1tot, sizeof(uint),   __LINE__, __FILE__);
    check_file_size(si2fh, N2tot, sizeof(uint),   __LINE__, __FILE__);
    check_file_size(simfh, NMtot, sizeof(uint),   __LINE__, __FILE__);
    check_file_size(ss1fh, Mtot,  sizeof(size_t), __LINE__, __FILE__);
    check_file_size(ss2fh, Mtot,  sizeof(size_t), __LINE__, __FILE__);
    check_file_size(ssmfh, Mtot,  sizeof(size_t), __LINE__, __FILE__);
    check_file_size(sl1fh, Mtot,  sizeof(size_t), __LINE__, __FILE__);
    check_file_size(sl2fh, Mtot,  sizeof(size_t), __LINE__, __FILE__);
    check_file_size(slmfh, Mtot,  sizeof(size_t), __LINE__, __FILE__);


    // Close files
    check_mpi(MPI_File_close(&bedfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&dimfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&si1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&sl1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ss1fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&si2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&sl2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ss2fh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&simfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&slmfh), __LINE__, __FILE__);
    check_mpi(MPI_File_close(&ssmfh), __LINE__, __FILE__);

    // Print approximate time for the conversion
    MPI_Barrier(MPI_COMM_WORLD);
    const auto et2 = std::chrono::high_resolution_clock::now();
    const auto dt2 = et2 - st2;
    const auto du2 = std::chrono::duration_cast<std::chrono::milliseconds>(dt2).count();
    if (rank == 0)   std::cout << "INFO   : time to convert the data: " << du2 / double(1000.0) << " seconds." << std::endl;
}
