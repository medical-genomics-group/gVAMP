#include <iostream>
#include <cstring>
#include <sstream>
#include <iostream>
#include <fstream>
#include <cassert>
#include <regex>
#include <mpi.h>
#include <boost/algorithm/string/trim.hpp>
#include "options.hpp"
#include <filesystem>

namespace fs = std::filesystem;

// Function to parse command line options
void Options::read_command_line_options(int argc, char** argv) {

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    std::stringstream ss;
    ss << "\nardyh command line options:\n";

    for (int i=1; i<argc; ++i) {

        if (!strcmp(argv[i], "--bed-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            bed_file = argv[++i];
            ss << "--bed-file " << bed_file << "\n";
        }
        else if (!strcmp(argv[i], "--dim-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            dim_file = argv[++i];
            ss << "--dim-file " << dim_file << "\n";
        }
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

        } else if (!strcmp(argv[i], "--shuffle-markers")) {
            if (i == argc - 1) fail_if_last(argv, i);
            shuffle = (bool) atoi(argv[++i]);
            ss << "--shuffle-markers " << shuffle << "\n";

        } else if (!strcmp(argv[i], "--mimic-hydra")) {
            mimic_hydra_ = true;
            ss << "--mimic-hydra " << mimic_hydra_ << "\n";

        } else if (!strcmp(argv[i], "--seed")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 0) {
                std::cout << "FATAL  : option --seed has to be a positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            seed = (unsigned int)atoi(argv[++i]);
            ss << "--seed " << seed << "\n";

        } else if (!strcmp(argv[i], "--iterations")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --iterations has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            iterations = (unsigned int) atoi(argv[++i]);
            ss << "--iterations " << iterations << "\n";

        }  else if (!strcmp(argv[i], "--trunc-markers")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --trunc-markers has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            truncm = (unsigned int) atoi(argv[++i]);
            ss << "--trunc-markers " << truncm << "\n";

        } else if (!strcmp(argv[i], "--S")) {
            if (i == argc - 1) fail_if_last(argv, i);
            std::stringstream s_stream(argv[++i]);
            while(s_stream.good()) {
                std::string substr;
                getline(s_stream, substr, ',');
                S.push_back(stod(substr));
                assert(S.back() > 0.0);
                if (S.size() > 1) {
                    //std::cout << S.back() << " > " << S.at(S.size() - 2) << std::endl;
                    assert(S.back() > S.at(S.size() - 2));
                }
            }
            ss << "--S " << argv[i] << "\n";

        }  else if (!strcmp(argv[i], "--out-dir")) {
            if (i == argc - 1) fail_if_last(argv, i);
            out_dir = argv[++i];
            fs::path pod = out_dir;
            if (!fs::exists(pod)) {
                fs::create_directory(pod);
            }
            ss << "--out-dir " << out_dir << "\n";

        } else if (!strcmp(argv[i], "--output-thin-rate")) {
            if (i == argc - 1) fail_if_last(argv, i);
            if (atoi(argv[i + 1]) < 1) {
                std::cout << "FATAL  : option --output-thin-rate has to be a strictly positive integer! (" << argv[i + 1] << " was passed)" << std::endl;
                exit(EXIT_FAILURE);
            }
            thin = (unsigned int)atoi(argv[++i]);
            ss << "--output-thin-rate " << thin << "\n";

        } else if (!strcmp(argv[i], "--predict")) {
            predict_ = true;
            ss << "--predict " << predict_ << "\n";

        } else if (!strcmp(argv[i], "--bim-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            bim_file = argv[++i];
            ss << "--bim-file " << bim_file << "\n";

        } else if (!strcmp(argv[i], "--ref-bim-file")) {
            if (i == argc - 1) fail_if_last(argv, i);
            ref_bim_file = argv[++i];
            ss << "--ref-bim-file " << ref_bim_file << "\n";

        } else {
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
    if (get_bed_file() == "") {
        std::cout << "FATAL  : no bed file provided! Please use the --bed-file option." << std::endl;
        exit(EXIT_FAILURE);
    }
    //std::cout << "  bed file: OK - " << get_bed_file() << "\n";

    if (get_dim_file() == "") {
        std::cout << "FATAL  : no dim file provided! Please use the --dim-file option." << std::endl;
        exit(EXIT_FAILURE);
    }
    //std::cout << "  dim file: OK - " << get_dim_file() << "\n";

    if (count_phen_files() == 0) {
        std::cout << "FATAL  : no phen file(s) provided! Please use the --phen-files option." << std::endl;
        exit(EXIT_FAILURE);
    }
    //std::cout << "  phen file(s): OK - " << count_phen_files() << " files passed.\n";
    //list_phen_files();


    // group index and mixture files: either both or none
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

    if (mimic_hydra_ && count_phen_files() > 1) {
        std::cout << "FATAL  : with --mimic-hydra, only a single phenotype can be processed." << std::endl;
        exit(EXIT_FAILURE);
    }
}

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
