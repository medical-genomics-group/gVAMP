#include <iostream>
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cassert>
#include <regex>
#include <mpi.h>
#include <boost/algorithm/string/trim.hpp> // -> module load boost 
#include "options.hpp"
// #include <filesystem>
#include <experimental/filesystem>

//namespace fs = std::filesystem;
namespace fs = std::experimental::filesystem;

// Function to parse command line options
void Options::read_command_line_options(int argc, char** argv) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::stringstream ss;
    ss << "\nardyh command line options:\n";

    for (int i=1; i<argc; ++i) {

        //std::cout << "input string = '" << argv[i] << "'"<< std::endl;

        if (!strcmp(argv[i], "--bed-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            bed_file = argv[++i];
            ss << "--bed-file " << bed_file << "\n";
        }
        else if (!strcmp(argv[i], "--bed-file-test")) {
            if (i == argc - 1) fail_if_last(argv, i);
            bed_file_test = argv[++i];
            ss << "--bed-file-test " << bed_file_test << "\n";
        }
        else if (!strcmp(argv[i], "--estimate-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            estimate_file = argv[++i];
            ss << "--estimate-file " << estimate_file << "\n";
        }
        else if (!strcmp(argv[i], "--run-mode")) {
            if (i == argc - 1) fail_if_last(argv, i);
            run_mode = argv[++i];
            ss << "--run-mode " << run_mode << "\n";
        }
        /*
        else if (!strcmp(argv[i], "--dim-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            dim_file = argv[++i];
            ss << "--dim-file " << dim_file << "\n";
        }
        */
        // List of phenotype files to read; comma separated if more than one.
        else if (!strcmp(argv[i], "--phen-files")) {
            if (i == argc - 1) fail_if_last(argv, i);
            std::string cslist = argv[++i];
            ss << "--phen-files " << cslist << "\n";
            std::stringstream sslist(cslist);
            std::string filepath;
            while (getline(sslist, filepath, ',')) {
                std::ifstream phen_file(filepath);
                if (phen_file.is_open()) {
                    phen_file.close();
                    phen_files.push_back(filepath);
                } else {
                    std::cout << "FATAL: file " << filepath << " not found\n";
                    exit(EXIT_FAILURE);
                }
            }
        }
        else if (!strcmp(argv[i], "--phen-files-test")) {
            if (i == argc - 1) fail_if_last(argv, i);
            std::string cslist = argv[++i];
            ss << "--phen-files-test " << cslist << "\n";
            std::stringstream sslist(cslist);
            std::string filepath;
            while (getline(sslist, filepath, ',')) {
                std::ifstream phen_file_test(filepath);
                if (phen_file_test.is_open()) {
                    phen_file_test.close();
                    phen_files_test.push_back(filepath);
                } else {
                    std::cout << "FATAL: file " << filepath << " not found\n";
                    exit(EXIT_FAILURE);
                }
            }
        }
        else if (!strcmp(argv[i], "--vars")) {
            if (i == argc - 1) fail_if_last(argv, i);
            std::string cslist = argv[++i];
            ss << "--vars " << cslist << "\n";
            std::stringstream sslist(cslist);
            std::string value;
            while (getline(sslist, value, ',')) {
                vars.push_back(atof(value.c_str()));
            }
        }
        else if (!strcmp(argv[i], "--probs")) {
            if (i == argc - 1) fail_if_last(argv, i);
            std::string cslist = argv[++i];
            ss << "--probs " << cslist << "\n";
            std::stringstream sslist(cslist);
            std::string value;
            while (getline(sslist, value, ',')) {
                probs.push_back(atof(value.c_str()));
            }
        }
        else if (!strcmp(argv[i], "--test-iter-range")) {
            if (i == argc - 1) fail_if_last(argv, i);
            std::string cslist = argv[++i];
            ss << "--test-iter-range " << cslist << "\n";
            std::stringstream sslist(cslist);
            std::string value;
            int nit = 0;
            while (getline(sslist, value, ',')) {
                test_iter_range[nit] = atoi(value.c_str());
                nit++;
            }
        }
        /* 
        else if (!strcmp(argv[i], "--group-index-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            group_index_file = argv[++i];
            ss << "--group-index-file " << group_index_file << "\n";
        } 
        else if (!strcmp(argv[i], "--group-mixture-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            group_mixture_file = argv[++i];
            ss << "--group-mixture-file " << group_mixture_file << "\n";
        } 
        else if (!strcmp(argv[i], "--verbosity")) {
            if (i == argc - 1) fail_if_last(argv, i);
            verbosity = atoi(argv[++i]);
            ss << "--verbosity " << verbosity << "\n";
        } */
        else if (!strcmp(argv[i], "--use-lmmse-damp")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 0) {
                std::cout << "FATAL  : option --iterations has to be a non-negative integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            use_lmmse_damp = (unsigned int) atoi(argv[++i]);
            ss << "--use-lmmse-damp " << use_lmmse_damp << "\n";
        }
        else if (!strcmp(argv[i], "--use-XXT-denoiser")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 0) {
                std::cout << "FATAL  : option --iterations has to be a non-negative integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            use_XXT_denoiser = (unsigned int) atoi(argv[++i]);
            ss << "--use-XXT-denoiser " << use_XXT_denoiser << "\n";
        }
        else if (!strcmp(argv[i], "--iterations")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --iterations has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            iterations = (unsigned int) atoi(argv[++i]);
            ss << "--iterations " << iterations << "\n";
        }
        else if (!strcmp(argv[i], "--num-mix-comp")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --num-mix-comp has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            num_mix_comp = (unsigned int) atoi(argv[++i]);
            ss << "--num-mix-comp " << num_mix_comp << "\n";
        } 
        else if (!strcmp(argv[i], "--store-pvals")){
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 0) {
                std::cout << "FATAL  : option --store_pvals has to be an integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            store_pvals = (unsigned int) atoi(argv[++i]);
            ss << "--store-pvals " << store_pvals << "\n";
        }
        /*else if (!strcmp(argv[i], "--meth-imp")){
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 0) {
                std::cout << "FATAL  : option --meth-imp has to be an integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            meth_imp = (unsigned int) atoi(argv[++i]);
            ss << "--meth-imp " << meth_imp << "\n";
        }*/
        else if (!strcmp(argv[i], "--out-dir")) {
            if (i == argc - 1) fail_if_last(argv, i);
            out_dir = argv[++i];
            fs::path pod = out_dir;
            if (!fs::exists(pod)) {
                fs::create_directory(pod);
            }
            ss << "--out-dir " << out_dir << "\n";
        }   
        else if (!strcmp(argv[i], "--out-name")) {
            if (i == argc - 1) fail_if_last(argv, i);
            out_name = argv[++i];
            ss << "--out-name " << out_name << "\n";
        }
        else if (!strcmp(argv[i], "--model")) {
            if (i == argc - 1) fail_if_last(argv, i);
            model = argv[++i];
            ss << "--model " << model << "\n";
        }
        /*else if (!strcmp(argv[i], "--predict")) {
            predict_ = true;
            ss << "--predict " << predict_ << "\n";
        } */
        else if (!strcmp(argv[i], "--bim-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            bim_file = argv[++i];
            ss << "--bim-file " << bim_file << "\n";

        } /* else if (!strcmp(argv[i], "--ref-bim-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            ref_bim_file = argv[++i];
            ss << "--ref-bim-file " << ref_bim_file << "\n";
        } */ 
        else if (!strcmp(argv[i], "--stop-criteria-thr")){ // strcmp return 0 if both strings are identical
            if (i == argc - 1) fail_if_last(argv, i);
            stop_criteria_thr = atof(argv[++i]);
            ss << "--stop-criteria-thr " << stop_criteria_thr << "\n";
        }
        else if (!strcmp(argv[i], "--EM-err-thr")){ // strcmp return 0 if both strings are identical
            if (i == argc - 1) fail_if_last(argv, i);
            EM_err_thr = atof(argv[++i]);
            ss << "--EM-err-thr " << EM_err_thr << "\n";
        }
        else if (!strcmp(argv[i], "--rho")){ // strcmp return 0 if both strings are identical
            if (i == argc - 1) fail_if_last(argv, i);
            rho = atof(argv[++i]);
            ss << "--rho " << rho << "\n";
        }
        else if (!strcmp(argv[i], "--EM-max-iter")){ // strcmp return 0 if both strings are identical
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --EM-max-iter has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            EM_max_iter = (unsigned int) atoi(argv[++i]);
            ss << "--EM-max-iter " << EM_err_thr << "\n";
        }
        else if (!strcmp(argv[i], "--Mt")){ // strcmp return 0 if both strings are identical
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --Mt has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            Mt = (unsigned int) atoi(argv[++i]);
            ss << "--Mt " << Mt << "\n";
        }
        else if (!strcmp(argv[i], "--N")){ // strcmp return 0 if both strings are identical
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --N has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            N = (unsigned int) atoi(argv[++i]);
            ss << "--N " << N << "\n";
        }
        else if (!strcmp(argv[i], "--N-test")){ // strcmp return 0 if both strings are identical
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --N_test has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            N_test = (unsigned int) atoi(argv[++i]);
            ss << "--N-test " << N_test << "\n";
        }
        else if (!strcmp(argv[i], "--Mt-test")){ // strcmp return 0 if both strings are identical
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --Mt_test has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            Mt_test = (unsigned int) atoi(argv[++i]);
            ss << "--Mt-test " << Mt_test << "\n";
        }
        else if (!strcmp(argv[i], "--CG-max-iter")){ // strcmp return 0 if both strings are identical
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --CG-max-iter has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            CG_max_iter = (unsigned int) atoi(argv[++i]);
            ss << "--CG-max-iter " << CG_max_iter << "\n";
        }
        else {
            std::cout << "FATAL: option \"" << argv[i] << "\" unknown\n";
            exit(EXIT_FAILURE);
        }
    }

    if (rank == 0)
        std::cout << ss.str() << std::endl;
}

void Options::list_phen_files() const {
    for (auto phen = phen_files.begin(); phen != phen_files.end(); ++phen) {
        std::cout << " phen file: " << *phen << std::endl;
    } 
}

// Catch missing argument on last passed option
void Options::fail_if_last(char** argv, const int i) {
    std::cout << "FATAL  : missing argument for last option \"" << argv[i] <<"\". Please check your input and relaunch." << std::endl;
    exit(EXIT_FAILURE);
}

// Check for minimal setup: a bed file + a dim file + phen file(s)
void Options::check_options() {
    if (get_bed_file() == "" && get_bed_file_test() == "") {
        std::cout << "FATAL  : no bed file provided! Please use the --bed-file option." << std::endl;
        exit(EXIT_FAILURE);
    }
    //std::cout << "  bed file: OK - " << get_bed_file() << "\n";

    /*
    if (get_dim_file() == "") {
        std::cout << "FATAL  : no dim file provided! Please use the --dim-file option." << std::endl;
        exit(EXIT_FAILURE);
    }
    */
    //std::cout << "  dim file: OK - " << get_dim_file() << "\n";
    /*
    if (count_phen_files() == 0 && count_phen_files_test() == 0) {
        std::cout << "FATAL  : no phen file(s) provided! Please use the --phen-files option." << std::endl;
        exit(EXIT_FAILURE);
    }
    */
    //std::cout << "  phen file(s): OK - " << count_phen_files() << " files passed.\n";
    //list_phen_files();


    // group index and mixture files: either both or none
    /*
    if (! predict_) {
        if ( (group_index_file == "" && group_mixture_file != "") ||
             (group_index_file != "" && group_mixture_file == ""))  {
            std::cout << "FATAL  : you need to activate BOTH --group-index-file and --group-mixture-file" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    if (predict_) {
        if (bim_file == "") {
            std::cout << "FATAL  : you need to pass a bim file with --bim-file when activating --predict" << std::endl;
            exit(EXIT_FAILURE);
        }
        if (ref_bim_file == "") {
            std::cout << "FATAL  : you need to pass a reference bim file with --ref-bim-file when activating --predict" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    */

     //assert(vars.size() == num_mix_comp || run_mode == "test");
     //assert(probs.size() == num_mix_comp || run_mode == "test");
}

/*
void Options::read_group_mixture_file() {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::fstream fs;
    fs.open(group_mixture_file, std::ios::in);

    int ngroups0 = -1;
    int groupi   =  0;
    std::vector<std::vector<std::string>> mixtures;

    if (fs.is_open()) {
        if (rank == 0)
            std::cout << "INFO   : Reading group mixtures from [" + group_mixture_file + "]." << std::endl;
        std::string mixture_line;
        std::regex ws_re("\\s+");
        while(getline(fs, mixture_line)) {
            //std::cout << "::: " << mixture_line << "\n";
            boost::algorithm::trim(mixture_line);
            if (mixture_line.length() == 0)  continue;
            std::vector<std::string> one_group_mixtures { 
                std::sregex_token_iterator(mixture_line.begin(), mixture_line.end(), ws_re, -1), {} 
            };
            int ngroups = one_group_mixtures.size();
            if (ngroups0 < 0) ngroups0 = ngroups;
            if (ngroups != ngroups0) {
                printf("FATAL  : check your mixture file. The same number of mixtures is expected for all groups.\n");
                printf("       : got %d mixtures for group %d, while first group had %d.\n", ngroups, groupi, ngroups0);
                exit(1);
            }
            //std::cout << "found " << ngroups << ", first el is >>" << one_group_mixtures.at(0)<< "<<" << std::endl;
            mixtures.push_back(one_group_mixtures);
            groupi++;
        }
        fs.close();
    } else {
        printf("FATAL  : can not open the mixture file %s. Use the --group-mixture-file option!\n", group_mixture_file.c_str());
        exit(1);
    }

    _set_ngroups(groupi);
    _set_nmixtures(ngroups0);

    cva.resize(ngroups);
    cvai.resize(ngroups);
    for (int i = 0; i < ngroups; i++) {
        cva[i].resize(nmixtures);
        cvai[i].resize(nmixtures);
        for (int j=0; j<nmixtures; j++) {
            cva[i][j]  = stod(mixtures[i][j]);
            if (j == 0 && cva[i][j] != 0.0) {
                printf("FATAL  : First element of group mixture must be 0.0! Check your input file %s.\n", group_mixture_file.c_str());
                exit(1);
            }            
            if (j > 0) {
                if (cva[i][j] <= cva[i][j-1]) {
                    printf("FATAL  : Mixtures must be given in ascending order! Check your input file %s.\n", group_mixture_file.c_str());
                    exit(1);
                }
                cvai[i][j] = 1.0 / cva[i][j];
            }
        }
    }
}
*/