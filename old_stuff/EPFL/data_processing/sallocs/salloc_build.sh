#!/bin/bash

salloc -N 1 -t 2:0:0 --mem=0 --exclusive -p build
