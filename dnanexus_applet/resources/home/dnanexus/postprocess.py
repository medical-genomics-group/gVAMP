import numpy as np
import argparse
import struct
import os
import pandas as pd
import csv

# This script for gVAMP postprocessing
print("---------- gVAMP postprocessing ----------")
print("\n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-pval", "--pval", help = "Path to pvals bin file")
parser.add_argument("-xhat", "--xhat", help = "Path to xhat bin file")
parser.add_argument("-bim", "--bim", help = "Path to bim file")
parser.add_argument("-out_name", "--out-name", help = "Output file name")
parser.add_argument("-M", "--M", help = "Number of markers")
args = parser.parse_args()

pval_fpath = args.pval
xhat_fpath = args.xhat
bim_fpath = args.bim
out_name = args.out_name
M = int(args.M)

print("Input arguments:")
print("--pval", pval_fpath)
print("--xhat", xhat_fpath)
print("--bim", bim_fpath)
print("--out-name", out_name)
print("--M", M)
print("\n", flush=True)

print("...Reading p-values from the file")
print(pval_fpath)
print("\n", flush=True)

f = open(pval_fpath, "rb")
buffer = f.read(M * 8)
pvals = struct.unpack(str(M)+'d', buffer)
pvals = np.array(pvals)

print("...Reading estimates from the file")
print(xhat_fpath)
print("\n", flush=True)

f = open(xhat_fpath, "rb")
buffer = f.read(M * 8)
xhat = struct.unpack(str(M)+'d', buffer)
xhat = np.array(xhat)

print("...Reading bim file")
print(bim_fpath)
print("\n", flush=True)

df_bim = pd.read_table(bim_fpath, header=None, names=['CHR', 'SNP', 'POS', 'BP', 'A1', 'A2'], sep="\s+")

df = pd.DataFrame({ 'CHR': df_bim['CHR'],
                    'SNP': df_bim['SNP'],
                    'BP': df_bim['BP'],
                    'A1': df_bim['A1'],
                    'A2': df_bim['A2'],
                    'BETA': xhat,
                    'P': pvals })

print("...Saving to file")
out_fpath = out_name + ".gvamp"
print(out_fpath)
print("\n", flush=True)
df.to_csv(out_fpath, index=None, sep="\t")