#include <string>
#include <fstream>
#include <iostream>
#include <regex>
#include "dimensions.hpp"


void Dimensions::read_dim_file(const std::string filepath) {

    std::ifstream infile(filepath);
    std::string line;
    std::regex re("\\s+");

    if (infile.is_open()) {
        getline(infile, line);
        std::sregex_token_iterator first{line.begin(), line.end(), re, -1}, last;
        std::vector<std::string> tokens{first, last};
        infile.close();
        if (tokens.size() != 2) {
            std::cout << "FATAL: dim file should contain a single line with 2 integers" << std::endl;
            exit(EXIT_FAILURE);
        }
        Nt = atoi(tokens[0].c_str());
        Mt = atoi(tokens[1].c_str());
    } else {
        std::cout << "FATAL: could not open dim file: " << filepath << std::endl;
        exit(EXIT_FAILURE);
    }
}
