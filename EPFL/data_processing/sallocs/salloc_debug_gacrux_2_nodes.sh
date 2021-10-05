#!/bin/bash

salloc -N 2 -t 1:0:0 --mem=0 --exclusive -p debug --constraint=s6g1
