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

if not os.path.exists('plots'):
    os.makedirs('plots')


data.columns = ["memory", "mflops", "version"]
version_label = ["Below neighbour", "Right neighbour"]

# create column of sizes
distinct_versions = data.version.unique()
size_n = [64, 128, 256, 512, 1024, 2048, 4096, 8192]*2
print(data)
print(size_n)
data["size_n"] = size_n
distinct_sizes = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

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
fig.savefig('plots' +'/mflopsVSsizeN.png')

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
fig.savefig('plots' +'/mflopsVSmemory.png')

# mflops vs size_n
fig = plt.figure()
ax = plt.subplot(111)
for version in distinct_versions:
	size_n = data[data.version == version].size_n
	memory = data[data.version == version].memory
	ax.plot(size_n, memory, label=f'{version_label[version]}', marker="o")
ax.legend()
plt.xlabel("Size N")
plt.ylabel("Mmeory footprint (kBytes)")
fig.savefig('plots' +'/memoryVSsizeN.png')

print("Success!. Images saved at /plots")
data.to_csv('plots' +'/latex.txt', header=True, index=False, sep='&')
exit()