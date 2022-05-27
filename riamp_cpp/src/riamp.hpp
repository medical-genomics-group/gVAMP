#pragma once
#include <iostream>
#include <typeinfo>
#include <map>
#include "options.hpp"
#include "phenotype.hpp"
#include "dimensions.hpp"
#include <vector>


class riamp {

public:
    void infere();
    double dot_product(const int mloc, double* __restrict__ phen, const double mu, const double sigma);
    int  get_N()  { return N;  } // Invariant over tasks
    int  get_M()  { return M;  } // Number of markers processed by task
    int  get_Mt() { return Mt; } // Total number of markers, sum over tasks
    int  get_Mm() { return Mm; } // Maximum number of markers per task (others may have M + 1)
    int  get_K()  { return K;  }

    void check_openmp();
    void print_cva();
    void print_cvai();

private:
    const Options opt;
    PhenMgr pmgr;
    const int N = 0;
    const int Mt = 0;
    const int rank = 0;
    const int nranks = 0;
    unsigned char* bed_data = nullptr;
    const int K = 0;
    const int G = 0;

    int S = 0;              // task marker start 
    int M = 0;              // task marker length
    int Mm = 0;
    size_t mbytes = 0;

    std::vector<int> mtotgrp;
    const std::vector<std::vector<double>> cva;
    const std::vector<std::vector<double>> cvai;

    void check_options();
    void setup_processing();
    void load_genotype();
    void check_processing_setup();
    void read_group_index_file(const std::string& file); // not needed atm

    struct priorMarkerDistr {
        int L;
        //std::vector<double> probs;
        //std::vector<double> vars;
        double *probs;
        double *vars;        
     };

};
