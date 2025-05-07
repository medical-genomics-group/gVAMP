import numpy as np
import pandas as pd
import argparse
import struct
import json
from scipy.stats import norm

eps = 1e-16

parser = argparse.ArgumentParser()
parser.add_argument("-parsed-log", "--parsed-log", help="Parsed logfile in JSON format")
parser.add_argument("-r1", "--r1", help="Path to r1 vector")
parser.add_argument("-it", "--it", help="Final gVAMP iteration")
parser.add_argument("-N", "--N", help="Number of individuals")
parser.add_argument("-M", "--M", help="Number of markers")
parser.add_argument("-out", "--out", help="Name of output file where to store pvals")
args = parser.parse_args()
parsed_log_fpath = args.parsed_log
r1_fpath = args.r1
N = int(args.N)
M = int(args.M)
it = int(args.it)
out = args.out

def pvals_calc(r1, gam1):
    
    r1 = np.asmatrix(r1)
    pvals = norm.cdf(-abs(r1), loc=0, scale=1/np.sqrt(N * gam1)) # one-sided test
    return np.asarray(pvals).squeeze()

f = open(parsed_log_fpath, "r")
d = json.load(f)
gam1s = d['gam1']

print("Calculating pvals in iteration", it)
    
gam1 = gam1s[it-2]
f = open(r1_fpath, "rb")
buffer = f.read(M*8)
r1 = struct.unpack(str(M)+'d', buffer)
r1 = np.array(r1)
f.close()

pvals = pvals_calc(r1, gam1)
df = pd.DataFrame({'pvals': pvals})
df.to_csv(out, index=False, header=None)