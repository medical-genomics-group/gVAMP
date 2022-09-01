#include <stdio.h>
#include <iostream>

int main(){

    //std::cout << " argc "  << argc << std::endl;
    /*
    if (argc == 2){
        probs = {0.70412, 0.26945, 0.02643};
        vars = {0, N*0.001251585388785e-5, N*0.606523422454662e-5};
    }
    else
    {
        double zero_mix = atof(argv[2]);
        if (rank == 0)
            std::cout << "zero_mix = " << zero_mix << std::endl;
        double variance = atof(argv[3]);
        if (rank == 0)
            std::cout << "variance = " << variance << std::endl;
        probs = {zero_mix, 1 - zero_mix};
        vars = {0, N*variance};
    }
    */

    // loading the location of .bed data
    //const std::string bedfp = "/nfs/scistore13/robingrp/human_data/geno/ldp08/ukb22828_UKB_EST_v3_ldp005_maf01.bed";
    //std::string phenfp = "/nfs/scistore13/robingrp/human_data/geno/ldp08/ukb_ht_noNA.phen";



    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //from vamp.hpp
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    //double l2_norm2(std::vector<double> const& u);
    //double inner_prod (std::vector<double> const& u, std::vector<double> const& v);
    //double inner_prod_N (std::vector<double> const& u, std::vector<double> const& v);
    //void store_vec_to_file(std::string filepath, std::vector<double> vec);


    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //from vamp.cpp
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    /*
    double vamp::l2_norm2(std::vector<double> const& u) { // this calculations are done in space of the markers
        return inner_prod(u,u);
    }


    double vamp::inner_prod(std::vector<double> const& u, std::vector<double> const& v){ // this calculations are done in space of the markers
        double accum = 0;
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+ : accum)
        #endif
        for (int i = 0; i < u.size(); i++) {
            accum += u[i] * v[i];
        }
        double accum_total = 0;
        //std::cout << "accum = " << accum << ", rank = " << rank << std::endl;
        MPI_Allreduce(&accum, &accum_total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return accum_total;
    }


    double vamp::inner_prod_N(std::vector<double> const& u, std::vector<double> const& v){ // this calculations are done in space of the phenotypes
        double accum = 0;
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+ : accum)
        #endif
        for (int i = 0; i < u.size(); i++) {
            accum += u[i] * v[i];
        }
        return accum;
    }


    void vamp::store_vec_to_file(std::string filepath, std::vector<double> vec){

        std::ofstream file;
        file.open(filepath);
    
        for(int i=0; i<vec.size(); i++)
            file << vec[i] << std::endl;

        file.close();

        std::cout << "done saving vectors" << std::endl;
    }
    */

    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //from main_real.cpp
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

        /*
        std::cout << " argc "  << argc << std::endl;
        if (argc == 2){
            probs = {0.70412, 0.26945, 0.02643};
            vars = {0, N*0.001251585388785e-5, N*0.606523422454662e-5};
        }
        else
        {
            double zero_mix = atof(argv[2]);
            if (rank == 0)
                std::cout << "zero_mix = " << zero_mix << std::endl;
            double variance = atof(argv[3]);
            if (rank == 0)
                std::cout << "variance = " << variance << std::endl;
            probs = {zero_mix, 1 - zero_mix};
            vars = {0, N*variance};
        }
        */



    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //from main.cpp
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    /*
  
    //std::vector<double> data = phen.get_phen();
    //std::cout << "data[1] = " << data[1] << std::endl; 
    //std::vector<double> beta_true = read_vec_from_file(true_betafp, M, S);

    //phen.store_vec_to_file(phen_out, y);
    //phen.read_phen();

    
    //std::cout << "length( beta_true ) = " << beta_true.size() << std::endl;
    //std::cout << "beta_true[1] = " << beta_true[1] << std::endl;
    //std::cout << "beta_true[2] = " << beta_true[2] << std::endl;
    //std::cout << "beta_true[3] = " << beta_true[3] << std::endl;

    double *mave = phen.get_mave();

    std::cout << "mave[0]: " << mave[0] << std::endl;

    std::vector<double> test_y(M, 0.0);
    std::cout << "dot product: " << phen.dot_product(1, test_y.data(), mave[1], 1) << std::endl;

    std::vector<double> test_y2(N, 0.0);
    test_y2[1] = 1;

    std::vector<double> outAx = phen.Ax( test_y2.data() );
    std::cout << "Ax product[1]: " << outAx[1] << std::endl;
    */

}