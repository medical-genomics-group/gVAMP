import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import csv
from textwrap import wrap

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-csv", "--csv", help = "Path to csv file")
parser.add_argument("-csv_test", "--csv-test", help = "Path to test csv file")
args = parser.parse_args()

csvf = args.csv
csv_test = args.csv_test

R2_test = []
R2_denois = []
R2_lmmse = []
gamw = []
gam1 = []

with open(csvf) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
        R2_denois.append(float(row[1]))
        R2_lmmse.append(float(row[3]))
        gam1.append(float(row[6]))
        gamw.append(float(row[5]))

with open(csv_test) as csv_test_file:
    csv_reader = csv.reader(csv_test_file, delimiter=',')
    for row in csv_reader:
        R2_test.append(float(row[1]))

R2_denois = np.array(R2_denois)
R2_lmmse = np.array(R2_lmmse)
R2_test = np.array(R2_test)
gamw = np.array(gamw)
gam1 = np.array(gam1)
it = np.argmax(gam1)

print("Max gam1 is in iteration:", it + 1)
print("heritability in iteration", it + 1, ": ", 1 - 1 / gamw[it])
print("Test R2 in iteration", it + 1, ": ", R2_test[it])

fig, ax = plt.subplots(2)
ax[0].plot(R2_denois, label="R2 denoising")
ax[0].plot(R2_lmmse, label="R2 LMMSE")
ax[0].plot(R2_test, label="R2 test")
ax[0].set_title("\n".join(wrap("", 100)))
ax[0].set_xlabel("iteration")
ax[0].set_ylabel("R2")
ax[0].set_ylim([0,1])
ax[0].legend()


ax[1].plot(gam1, label="signal error precision")
ax[1].plot(gamw, label="noise precision")
ax[1].set_xlabel("iteration")
ax[1].legend()

plt.show()
