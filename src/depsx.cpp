#include <mpi.h>
#include <vector>
#include "utils.hpp"
#include "data.hpp"
#include "dense.hpp"
#include "sparse.hpp"
#include "dotp_lut.h"

// Update local copy of epsilon
//MPI_Barrier(MPI_COMM_WORLD);

void delta_epsilon_exchange(const bool opt_bedSync,
                            const bool opt_sparseSync,
                            std::vector<int> mark2sync,
                            std::vector<double> dbet2sync,
                            const double* mave,
                            const double* mstd,
                            const size_t snpLenByt,
                            const size_t snpLenUint,
                            const bool* USEBED,
                            const sparse_info_t* sparse_info,
                            const uint Ntot,
                            const Data data,
                            const double* dEpsSum,
                            const double* tmpEps,
                            double* epsilon) {
    
    int rank, nranks;
    
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    int *glob_info, *tasks_len, *tasks_dis, *stats_len, *stats_dis;

    glob_info = (int*) _mm_malloc(size_t(nranks * 2) * sizeof(int), 64);  check_malloc(glob_info,  __LINE__, __FILE__);
    tasks_len = (int*) _mm_malloc(size_t(nranks)     * sizeof(int), 64);  check_malloc(tasks_len,  __LINE__, __FILE__);
    tasks_dis = (int*) _mm_malloc(size_t(nranks)     * sizeof(int), 64);  check_malloc(tasks_dis,  __LINE__, __FILE__);
    stats_len = (int*) _mm_malloc(size_t(nranks)     * sizeof(int), 64);  check_malloc(stats_len,  __LINE__, __FILE__);
    stats_dis = (int*) _mm_malloc(size_t(nranks)     * sizeof(int), 64);  check_malloc(stats_dis,  __LINE__, __FILE__);
    
    double* deltaSum   = (double*)_mm_malloc(size_t(Ntot) * sizeof(double), 64);  check_malloc(deltaSum,   __LINE__, __FILE__);

    if (nranks > 1) {

        // Bed synchronization
        //
        if (opt_bedSync) {
            
            // 1. Get overall number of markers contributing to synchronization
            // EO: check types below but should be fine as numbers should be relatively small
            //                        
            int task_m2s = (int) mark2sync.size();

            // Build task markers to sync statistics: mu | dbs | mu | dbs | ...
            //printf("task %d has %d m2s\n", rank, task_m2s);
            double* task_stat = (double*) _mm_malloc(size_t(task_m2s) * 2 * sizeof(double), 64);
            check_malloc(task_stat, __LINE__, __FILE__);
            for (int i=0; i<task_m2s; i++) {
                task_stat[2 * i + 0] = mave[ mark2sync[i] ];
                task_stat[2 * i + 1] = mstd[ mark2sync[i] ] * dbet2sync[i];
            }

            check_mpi(MPI_Allgather(&task_m2s, 1, MPI_INT, tasks_len, 1, MPI_INT, MPI_COMM_WORLD), __LINE__, __FILE__);

            int tdisp_ = 0, sdisp_ = 0, glob_m2s = 0, glob_size = 0;

            for (int i=0; i<nranks; i++) {
                glob_m2s     += tasks_len[i];     // in number of markers
                stats_len[i]  = tasks_len[i] * 2;
                stats_dis[i]  = sdisp_;
                sdisp_       += stats_len[i];
                tasks_len[i] *= snpLenByt;        // each marker is same length in BED
                tasks_dis[i]  = tdisp_;           // in bytes
                tdisp_       += tasks_len[i];     // now in bytes
            }
            glob_size = tdisp_; // in bytes
                        
            
            //if (rank == 0) {
            //for (int i=0; i<nranks; i++) 
            //printf("| %d:%d", i, tasks_len[i]/(int)snpLenByt);
            //printf(" |  => Tot = %d\n", glob_m2s);
            //}
            //fflush(stdout);
            
            // Alloc to store all task's markers in BED format
            //
                        
            char* task_bed = (char*)_mm_malloc(snpLenByt * task_m2s, 64);  check_malloc(task_bed, __LINE__, __FILE__);
                        
            for (int i=0; i<mark2sync.size(); i++) {
                            
                int m2si = mark2sync[i];
                            
                if (USEBED[m2si]) {

                    memcpy(&task_bed[i * snpLenByt], reinterpret_cast<char*>(&sparse_info->I1[sparse_info->N1S[m2si]]), snpLenByt);
                            
                } else {

                    data.get_bed_marker_from_sparse(&task_bed[(size_t)i * snpLenByt],
                                                    Ntot,
                                                    sparse_info->N1S[m2si], sparse_info->N1L[m2si], &sparse_info->I1[sparse_info->N1S[m2si]],
                                                    sparse_info->N2S[m2si], sparse_info->N2L[m2si], &sparse_info->I2[sparse_info->N2S[m2si]],
                                                    sparse_info->NMS[m2si], sparse_info->NML[m2si], &sparse_info->IM[sparse_info->NMS[m2si]]);
                    // local check ;-)
                    //size_t X1 = 0, X2 = 0, XM = 0;
                    //data.sparse_data_get_sizes_from_raw(&task_bed[(size_t) i * snpLenByt], 1, snpLenByt, data.numNAs, X1, X2, XM);
                    //printf("data.sparse_data_get_sizes_from_raw => (%2d, %3d): X1 = %9lu, X2 = %9lu, XM = %9lu #?# vs %9lu %9lu %9lu\n", rank, i, X1, X2, XM, N1L[m2si], N2L[m2si], NML[m2si]);
                    //fflush(stdout);

                }
            }


            // Collect BED data from all markers to sync from all tasks
            //
            char* glob_bed = (char*)_mm_malloc(snpLenByt * glob_m2s, 64);  check_malloc(task_bed, __LINE__, __FILE__);
                        
            check_mpi(MPI_Allgatherv(task_bed, tasks_len[rank], MPI_CHAR,
                                     glob_bed, tasks_len, tasks_dis, MPI_CHAR, MPI_COMM_WORLD), __LINE__, __FILE__);

            double* glob_stats = (double*) _mm_malloc(size_t(glob_m2s * 2) * sizeof(double), 64);
            check_malloc(glob_stats, __LINE__, __FILE__);
                        
            check_mpi(MPI_Allgatherv(task_stat, task_m2s * 2, MPI_DOUBLE,
                                     glob_stats, stats_len, stats_dis, MPI_DOUBLE, MPI_COMM_WORLD), __LINE__, __FILE__); 

            // Now apply corrections from each marker to sync
            //
            for (int i=0; i<glob_m2s; i++) {

                double lambda0 = glob_stats[2 * i + 1] * (0.0 - glob_stats[2 * i]);
                //printf("rank %d, %2d lambda0 = %20.15f\n", rank, i, lambda0);
                //printf("rank %d lambda0 = %15.10f with mu = %15.10f, dbetsig = %15.10f %d/%d\n", rank, lambda0, glob_stats[2 * i], glob_stats[2 * i + 1], i, glob_m2s);
                                                       
                            
                // Get sizes from BED data
                // note: locally available from markers local to task but ignored for now
                //
                size_t X1 = 0, X2 = 0, XM = 0;
                data.sparse_data_get_sizes_from_raw(&glob_bed[(size_t) i * snpLenByt], 1, snpLenByt, data.numNAs, X1, X2, XM);
                //printf("data.sparse_data_get_sizes_from_raw => (%2d, %3d): X1 = %9lu, X2 = %9lu, XM = %9lu ###\n", rank, i, X1, X2, XM);
                //fflush(stdout);
                // Allocate sparse structure
                //
                uint* XI1 = (uint*)_mm_malloc(X1 * sizeof(uint), 64);  check_malloc(XI1, __LINE__, __FILE__);
                uint* XI2 = (uint*)_mm_malloc(X2 * sizeof(uint), 64);  check_malloc(XI2, __LINE__, __FILE__);
                uint* XIM = (uint*)_mm_malloc(XM * sizeof(uint), 64);  check_malloc(XIM, __LINE__, __FILE__);
                            
                            
                // Fill the structure
                size_t fake_n1s = 0, fake_n2s = 0, fake_nms = 0;
                size_t fake_n1l = 0, fake_n2l = 0, fake_nml = 0;
                            
                //EO: bed data already adjusted for NAs
                data.sparse_data_fill_indices(&glob_bed[(size_t) i * snpLenByt], 1, snpLenByt, data.numNAs,
                                              &fake_n1s, &fake_n1l, XI1,
                                              &fake_n2s, &fake_n2l, XI2,
                                              &fake_nms, &fake_nml, XIM);
                
                // Use it
                // Set all to 0 contribution
                if (i == 0) {
                    set_array(deltaSum, lambda0, Ntot);
                } else {
                    offset_array(deltaSum, lambda0, Ntot);
                }
                            
                // M -> revert lambda 0 (so that equiv to add 0.0)
                sparse_add(deltaSum, -lambda0, XIM, 0, XM);
                            
                // 1 -> add dbet * sig * ( 1.0 - mu)
                double lambda = glob_stats[2 * i + 1] * (1.0 - glob_stats[2 * i]);
                sparse_add(deltaSum, lambda - lambda0, XI1, 0, X1);
                            
                // 2 -> add dbet * sig * ( 2.0 - mu)
                lambda = glob_stats[2 * i + 1] * (2.0 - glob_stats[2 * i]);
                sparse_add(deltaSum, lambda - lambda0, XI2, 0, X2);

                // Free memory and reset pointers
                _mm_free(XI1);  XI1 = NULL;
                _mm_free(XI2);  XI2 = NULL;
                _mm_free(XIM);  XIM = NULL;
            }
            //fflush(stdout);

            //MPI_Barrier(MPI_COMM_WORLD);

            _mm_free(glob_stats);
            _mm_free(glob_bed);                       
            _mm_free(task_bed);
            _mm_free(task_stat);

            mark2sync.clear();
            dbet2sync.clear();                            
        }

        // Sparse synchronization
        // ----------------------
        else if (opt_sparseSync) {

            uint task_m2s = (uint) mark2sync.size();
            //printf("task %3d has %3d markers to share at %d\n", rank, task_m2s, sinceLastSync);
            //fflush(stdout);

            // Build task markers to sync statistics: mu | dbs | mu | dbs | ...
            double* task_stat = (double*) _mm_malloc(size_t(task_m2s) * 2 * sizeof(double), 64);
            check_malloc(task_stat, __LINE__, __FILE__);

            // Compute total number of elements to be sent by each task
            uint task_size = 0;
            for (int i=0; i<task_m2s; i++) {
                if (USEBED[mark2sync[i]]) {
                    task_size += snpLenUint + 3;
                } else {
                    task_size += (sparse_info->N1L[ mark2sync[i] ] + 
                                  sparse_info->N2L[ mark2sync[i] ] +
                                  sparse_info->NML[ mark2sync[i] ] + 3);
                }
                task_stat[2 * i + 0] = mave[ mark2sync[i] ];
                task_stat[2 * i + 1] = mstd[ mark2sync[i] ] * dbet2sync[i];
                //printf("Task %3d, m2s %d/%d: 1: %8lu, 2: %8lu, m: %8lu, info: 3); stats are (%15.10f, %15.10f)\n", rank, i, task_m2s, N1L[ mark2sync[i] ], N2L[ mark2sync[i] ], NML[ mark2sync[i] ], task_stat[2 * i + 0], task_stat[2 * i + 1]);
            }
            //printf("Task %3d final task_size = %8d elements to send from task_m2s = %d markers to sync.\n", rank, task_size, task_m2s);
            //fflush(stdout);

            // Get the total numbers of markers and corresponding indices to gather

            const int NEL = 2;
            uint task_info[NEL] = {};
            task_info[0] = task_m2s;
            task_info[1] = task_size;

            check_mpi(MPI_Allgather(task_info, NEL, MPI_UNSIGNED, glob_info, NEL, MPI_UNSIGNED, MPI_COMM_WORLD), __LINE__, __FILE__);

            int tdisp_ = 0, sdisp_ = 0, glob_m2s = 0, glob_size = 0;
            for (int i=0; i<nranks; i++) {
                tasks_len[i]  = glob_info[2 * i + 1];
                tasks_dis[i]  = tdisp_;
                tdisp_       += tasks_len[i];
                stats_len[i]  = glob_info[2 * i] * 2; // number of markers in task i times 2 for stat1 and stat2
                stats_dis[i]  = sdisp_;
                sdisp_       += glob_info[2 * i] * 2;
                glob_size    += tasks_len[i];
                glob_m2s     += glob_info[2 * i];
            }
            //printf("glob_info: markers to sync: %d, with glob_size = %7d elements (sum of all task_size)\n", glob_m2s, glob_size);
            //fflush(stdout);

            // Build task's array to spread: | marker 1                             | marker 2
            //                               | n1 | n2 | nm | data1 | data2 | datam | n1 | n2 | nm | data1 | ...
            // -------------------------------------------------------------------------------------------------
            uint* task_dat = (uint*) _mm_malloc(size_t(task_size) * sizeof(uint), 64);
            check_malloc(task_dat, __LINE__, __FILE__);

            int loc = 0;

            for (int i=0; i<task_m2s; i++) {

                int m2si = mark2sync[i];
                            
                if (USEBED[m2si]) {

                    task_dat[loc++] = snpLenUint;  // bed data "as is" stored in uint
                    task_dat[loc++] = UINT_MAX;    // switch to detect a bed stored marker
                    task_dat[loc++] = 0;

                    const uint* rawdata = reinterpret_cast<uint*>(&sparse_info->I1[sparse_info->N1S[m2si]]);

                    for (int ii=0; ii<snpLenUint; ii++) {
                        task_dat[loc++] = rawdata[ii];
                    }

                } else {

                    task_dat[loc++] = sparse_info->N1L[m2si];
                    task_dat[loc++] = sparse_info->N2L[m2si];
                    task_dat[loc++] = sparse_info->NML[m2si];

                    //cout << "1: " << loc << ", " << N1L[m2si] << endl;
                    for (uint ii = 0; ii < sparse_info->N1L[m2si]; ii++) {
                        task_dat[loc] = sparse_info->I1[ sparse_info->N1S[m2si] + ii ];  loc += 1;
                    }  
                    //cout << "2: " << loc << ", " << N2L[m2si] << endl;                                                                
                    for (uint ii = 0; ii < sparse_info->N2L[m2si]; ii++) {
                        task_dat[loc] = sparse_info->I2[sparse_info-> N2S[m2si] + ii ];  loc += 1;
                    }
                    //cout << "M: " << loc << ", " << NML[m2si] << endl;
                    for (uint ii = 0; ii < sparse_info->NML[m2si]; ii++) {
                        task_dat[loc] = sparse_info->IM[ sparse_info->NMS[m2si] + ii ];  loc += 1;
                    }
                }
            }
            //printf("loc vs task_size = %d vs %d\n", loc, task_size);
            assert(loc == task_size);

                        
            // Allocate receive buffer for all the data
            //if (rank == 0)
            //    printf("glob_size = %d\n", glob_size);
            uint* glob_dat = (uint*) _mm_malloc(size_t(glob_size) * sizeof(uint), 64);
            check_malloc(glob_dat, __LINE__, __FILE__);

            check_mpi(MPI_Allgatherv(task_dat, task_size, MPI_UNSIGNED,
                                     glob_dat, tasks_len, tasks_dis, MPI_UNSIGNED, MPI_COMM_WORLD), __LINE__, __FILE__);
            _mm_free(task_dat);
            //cout << "glob_size = " << glob_size << endl;

            double* glob_stats = (double*) _mm_malloc(size_t(glob_size * 2) * sizeof(double), 64);
            check_malloc(glob_stats, __LINE__, __FILE__);

            check_mpi(MPI_Allgatherv(task_stat, task_m2s * 2, MPI_DOUBLE,
                                     glob_stats, stats_len, stats_dis, MPI_DOUBLE, MPI_COMM_WORLD), __LINE__, __FILE__);
            _mm_free(task_stat);


            // Compute global delta epsilon deltaSum
            //
            size_t loci = 0;
            double c1 = 0.0, c2 = 0.0;

            for (int i=0; i<glob_m2s ; i++) {

                //printf("m2s %d/%d (loci = %lu): %d, %d, %d\n", i, glob_m2s, loci, glob_dat[loci], glob_dat[loci + 1], glob_dat[loci + 2]);
                double lambda0 = glob_stats[2 * i + 1] * (0.0 - glob_stats[2 * i]);
                //printf("rank %d lambda0 = %15.10f with mu = %15.10f, dbetsig = %15.10f\n", rank, lambda0, glob_stats[2 * i], glob_stats[2 * i + 1]);

                if (glob_dat[loci + 1] == UINT_MAX) {

                    // Reset vector
                    if (i == 0)  set_array(deltaSum, 0.0, Ntot);

                    const uint8_t* rawdata = reinterpret_cast<uint8_t*>(&glob_dat[loci + 3]);

                    // main + remainder to avoid a test on idx < Ntot 
                    const int fullb = Ntot / 4;
                    int idx = 0;

                    // main
                                
                    double mu_ = glob_stats[2 * i];
                    double si_ = glob_stats[2 * i + 1];

                    __m256d vmu_ = _mm256_set1_pd(mu_);
                    __m256d vsi_ = _mm256_set1_pd(si_);
                                
#ifdef _OPENMP
#pragma omp parallel for
#endif                                
                    for (int ii=0; ii<fullb; ++ii) {
                                
                        __m256d p4c1  = _mm256_loadu_pd(&(dotp_lut_a[rawdata[ii] * 4]));
                        __m256d p4c2  = _mm256_loadu_pd(&(dotp_lut_b[rawdata[ii] * 4]));
                                    
                        p4c1 = _mm256_sub_pd(p4c1, vmu_);
                        p4c2 = _mm256_mul_pd(p4c2, vsi_);
                        p4c2 = _mm256_mul_pd(p4c2, p4c1);

                        __m256d p4dls = _mm256_loadu_pd(&(deltaSum[ii * 4])); 
                                    
                        p4dls = _mm256_add_pd(p4dls, p4c2);

                        _mm256_store_pd(&(deltaSum[ii * 4]), p4dls);

                    }

                    // remainder
                    if (Ntot % 4 != 0) {
                        int ii = fullb;
                        for (int iii = 0; iii < Ntot - fullb * 4; iii++) {
                            idx = rawdata[ii] * 4 + iii;
                            c1  = dotp_lut_a[idx];
                            c2  = dotp_lut_b[idx];
                            deltaSum[ii * 4 + iii] += (c1 - glob_stats[2 * i]) * c2 * glob_stats[2 * i + 1];
                        }
                    }
                                
                    loci += 3 + glob_dat[loci];  // + 1 contains UINT_MAX to mark BED data

                } else {
                                
                    if (i == 0) {
                        set_array(deltaSum, lambda0, Ntot);
                    } else {
                        offset_array(deltaSum, lambda0, Ntot);
                    }

                    // M -> revert lambda 0 (so that equiv to add 0.0)
                    size_t S = loci + (size_t) (3 + glob_dat[loci] + glob_dat[loci + 1]);
                    size_t L = glob_dat[loci + 2];
                    //cout << "task " << rank << " M: start = " << S << ", len = " << L <<  endl;
                    sparse_add(deltaSum, -lambda0, glob_dat, S, L);
                                
                    // 1 -> add dbet * sig * ( 1.0 - mu)
                    double lambda = glob_stats[2 * i + 1] * (1.0 - glob_stats[2 * i]);
                    //printf("1: lambda = %15.10f, l-l0 = %15.10f\n", lambda, lambda - lambda0);
                    S = loci + 3;
                    L = glob_dat[loci];
                    //cout << "1: start = " << S << ", len = " << L <<  endl;
                    sparse_add(deltaSum, lambda - lambda0, glob_dat, S, L);
                                
                    // 2 -> add dbet * sig * ( 2.0 - mu)
                    lambda = glob_stats[2 * i + 1] * (2.0 - glob_stats[2 * i]);
                    S = loci + 3 + glob_dat[loci];
                    L = glob_dat[loci + 1];
                    //cout << "2: start = " << S << ", len = " << L <<  endl;
                    sparse_add(deltaSum, lambda - lambda0, glob_dat, S, L);

                    loci += 3 + glob_dat[loci] + glob_dat[loci + 1] + glob_dat[loci + 2];
                }

            }

            _mm_free(glob_stats);
            _mm_free(glob_dat);

            mark2sync.clear();
            dbet2sync.clear();

        } else {

            check_mpi(MPI_Allreduce(&dEpsSum[0], &deltaSum[0], Ntot, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD), __LINE__, __FILE__);
        }

        add_arrays(epsilon, tmpEps, deltaSum, Ntot);

    } else { // case nranks == 1

        add_arrays(epsilon, tmpEps, dEpsSum, Ntot);
    }

    _mm_free(glob_info);
    _mm_free(tasks_len);
    _mm_free(tasks_dis);
    _mm_free(stats_len);
    _mm_free(stats_dis);
   
    _mm_free(deltaSum);
}
