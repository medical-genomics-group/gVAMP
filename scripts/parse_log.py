import json
import numpy as np
import pandas as pd
import argparse
import re

parser = argparse.ArgumentParser()
parser.add_argument("-log", "--log", help="Path to gVAMP logfile")
args = parser.parse_args()
log = args.log
out = log.split('.')[0] + ".json"


def parse(txt, par, step):
    vals = []
    sub1s = txt.split("********************")
    for sub1 in sub1s:
        sub2s = sub1.split("______________________")
        for sub2 in sub2s:
            if step in sub2:
                lines = sub2.split("\n")
                for line in lines:
                    if line.startswith(par):
                        val = float(line.split("=")[1])
                        vals.append(val)
                        break
    return vals

def parse_2(txt, par, step):
    vals = []
    sub1s = txt.split("********************")
    for sub1 in sub1s:
        sub2s = sub1.split("______________________")
        for sub2 in sub2s:
            if step in sub2:
                lines = sub2.split("\n")
                for line in lines:
                    if line.startswith(par):
                        val = (line.split("=")[1])
                        # p = re.compile(r'\d+\.\d+')  # Compile a pattern to capture float values
                        p = re.compile('-?\ *[0-9]+\.?[0-9]*(?:[Ee]\ *-?\ *[0-9]+)?')
                        floats = [float(i) for i in p.findall(val)]  # Convert strings to float
                        vals.append(floats)
    return vals

f = open(log, "r")
text = f.read()
gam1s = parse(text, "gam1", "LMMSE")
gamws = parse(text, "gamw", "LMMSE")
R2_lmmse = parse(text, "R2", "LMMSE")
R2_den = parse(text, "R2", "DENOISING")
prior_vars = parse_2(text, "prior variances", "denoising")
prior_probs = parse_2(text, "prior probabilities", "denoising")
f.close()

d = {"gam1": gam1s,
    "gamw": gamws,
    "R2_den": R2_den,
    "R2_lmmse": R2_lmmse,
    "prior_vars": prior_vars,
    "prior_probs": prior_probs}
f = open(out, "w")
json.dump(d, f)
f.close()
