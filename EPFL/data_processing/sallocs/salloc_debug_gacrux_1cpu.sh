#!/bin/bash

salloc -N 1 -n 1 -t 1:0:0 --mem=30G -p debug --constraint=s6g1
