#!/bin/bash
set -e

module purge
module load gcc

g++ -std=c++17 mk_lut.cpp    -o mk_lut    || exit 1
g++ -std=c++17 mk_lut_na.cpp -o mk_lut_na || exit 1

HPP=na_lut.hpp_tmp
./mk_lut_na > $HPP
echo Wrote file $HPP

HPP=dotp_lut.hpp_tmp
./mk_lut > $HPP
echo Wrote file $HPP



