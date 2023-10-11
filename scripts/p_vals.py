import numpy as np
import argparse
import struct
from scipy.stats import norm
import matplotlib.pyplot as plt
import csv
import os
from array import array

#This script calculates p values from r1

np.random.seed(1)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-csv", "--csv", help = "Path to csv file")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-N", "--N", help = "Number of samples")
args = parser.parse_args()
csvf = args.csv
print(csvf)

Mt = int(args.M)
N = int(args.N)
pvals_thr = 0.05 / Mt

gam1s = []
with open(csvf) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        gam1s.append(float(row[6]))
gam1s = np.array(gam1s)

it = np.argmax(gam1s)
gam1 = gam1s[it]
print("Max gam1=", gam1, " is in iteration ", it + 1)

basename = os.path.basename(csvf)
basename = basename.split('.')[0]
dirpath = os.path.dirname(csvf)
r1_fpath = os.path.join(dirpath, basename + "_r1_it_" + str(it+1) + ".bin")
print(r1_fpath)
r1_binfile = open(r1_fpath, "rb")
buffer = r1_binfile.read(Mt*8)
r1 = struct.unpack(str(Mt)+'d', buffer)

pvals = np.zeros(Mt)
for i in range(Mt):
    pvals[i] = norm.cdf(x=0, loc=r1[i], scale=np.sqrt(1 / (gam1 * N)))
    if r1[i] <= 0:
        pvals[i] = 1 - pvals[i]  

print("Number of causal markers: ", sum(pvals <= pvals_thr))

output_file = open(os.path.join(dirpath, basename+'.pval'), 'wb')
float_array = array('d', pvals)
float_array.tofile(output_file)
output_file.close()