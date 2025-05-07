import numpy as np
import pandas as pd
import argparse
import struct
import json

eps = 1e-16

parser = argparse.ArgumentParser()
parser.add_argument("-parsed-log", "--parsed-log", help="Parsed logfile in JSON format")
parser.add_argument("-r1", "--r1", help="Path to r1 vector")
parser.add_argument("-it", "--it", help="Final gVAMP iteration")
parser.add_argument("-N", "--N", help="Number of individuals")
parser.add_argument("-M", "--M", help="Number of markers")
parser.add_argument("-out", "--out", help="Name of output file where to store PIPs")
args = parser.parse_args()
parsed_log_fpath = args.parsed_log
r1_fpath = args.r1
N = int(args.N)
M = int(args.M)
it = int(args.it)
out = args.out

def pip_calc(r1, gam1, omegas, sigmas, la):
    r1 = np.asmatrix(r1)
    gam1inv = 1.0/gam1
    beta_tilde=np.multiply( np.exp( - np.power(np.transpose(r1),2) / 2 / (sigmas + gam1inv)), omegas / np.sqrt(gam1inv + sigmas) )
    sum_beta_tilde = beta_tilde.sum(axis=1)
    pi = la / ( la + (1-la) * np.exp(-np.power(np.transpose(r1),2) / 2 * gam1 ) / np.sqrt(gam1inv) / sum_beta_tilde )
    return np.asarray(pi).squeeze()

f = open(parsed_log_fpath, "r")
d = json.load(f)
gam1s = d['gam1']
gamws = d['gamw']
prior_vars = d['prior_vars']
prior_probs = d['prior_probs']

print("Calculating PIPs in iteration", it)
probs = prior_probs[it-2]
vars = np.array(prior_vars[it-2]) / N
    
gam1 = gam1s[it-2]
f = open(r1_fpath, "rb")
buffer = f.read(M*8)
r1 = struct.unpack(str(M)+'d', buffer)
r1 = np.array(r1)
f.close()
    
sigmas = np.array(vars[1:])
omegas = np.array([ p / sum(probs[1:]) for p in probs[1:]])
la = 1 - probs[0]

pips = pip_calc(r1, gam1 * N, omegas, sigmas, la)
df = pd.DataFrame({'pips': pips})
df.to_csv(out, index=False, header=None)