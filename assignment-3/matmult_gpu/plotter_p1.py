"""
plotter.py

syntax: python plotter.py PATH_TO_TXT

"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if (len(sys.argv) != 2):
	print("Wrong syntax: python plotter.py PATH_TO_TXT")
	exit()

try:
	data = pd.read_csv(str(sys.argv[1]), sep= " ", header=None)
except:
	print("Couldn't find valid file at", str(sys.argv[1]))
	exit()

if not os.path.exists('plots_p1'):
    os.makedirs('plots_p1')

# 22 24 44 42 28 48 88 82 84 (ncol nrow)
data.columns = ["memory", "mflops", "version", "size_n"]
version_label = ["lib", "gpu1", "gpu2"]
# create column of sizes
distinct_versions = data.version.unique()

print(data)

# mflops vs size_n
fig = plt.figure()
ax = plt.subplot(111)
for version in distinct_versions:
	size_n = data[data.version == version].size_n
	mflops = data[data.version == version].mflops
	ax.plot(size_n, mflops, label=f'{version_label[version]}', marker="o")
ax.legend()
plt.xlabel("Size N")
plt.ylabel("MFLOPS")
fig.savefig('plots_p1' +'/mflopsVSsizeN.png')

# mflops vs memory
fig = plt.figure()
ax = plt.subplot(111)
for version in distinct_versions:
	memory = data[data.version == version].memory
	mflops = data[data.version == version].mflops
	ax.plot(memory, mflops, label=f'{version_label[version]}', marker="o")
ax.legend()
plt.xlabel("Memory footprint (kBytes)")
plt.ylabel("MFLOPS")
fig.savefig('plots_p1' +'/mflopsVSmemory.png')

# mflops vs size_n
fig = plt.figure()
ax = plt.subplot(111)
for version in distinct_versions:
	size_n = data[data.version == version].size_n
	memory = data[data.version == version].memory
	ax.plot(size_n, memory, label=f'{version_label[version]}', marker="o")
ax.legend()
plt.xlabel("Size N")
plt.ylabel("Memory footprint (kBytes)")
fig.savefig('plots_p1' +'/memoryVSsizeN.png')

print("Success!. Images saved at /plots_p1")
data.to_csv('plots_p1' +'/latex.txt', header=True, index=False, sep='&')
exit()