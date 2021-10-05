#pragma once

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
                            double* epsilon);
