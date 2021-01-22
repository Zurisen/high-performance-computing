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
	data = pd.read_csv(str(sys.argv[1]), sep= " ")
except:
	print("Couldn't find valid file at", str(sys.argv[1]))
	exit()

if not os.path.exists('plots'):
    os.makedirs('plots')

data.columns = ["size_N", "iterations", "time", "iters_per_second", "version"]
distinct_sizes_N = data.size_N.unique()
distinct_versions = data.version.unique()

def compute_memory(data):
	memory = (data.size_N**3)*8*data.iterations/1000
	return memory
memories = data.apply(compute_memory, axis=1)
data["memory"] = memories.to_numpy()

# compute mlups
mlups = []
memories = data.memory.to_numpy()
times_vect = data.time.to_numpy()
iters = data.iterations.to_numpy()
for idx, memory in enumerate(memories):
	mlups.append(memory*iters[idx]/(8*1000*times_vect[idx]))
data["mlups"] = mlups

print(data)

# total time vs matrix size N
fig = plt.figure()
ax = plt.subplot(111)
for version in distinct_versions:
	size_n = data[data.version == version].size_N
	time = data[data.version == version].time
	ax.plot(size_n, time, label=f'{version} version', marker="o")
ax.legend()
plt.xlabel("size N")
plt.ylabel("time (s)")
fig.savefig('plots' +'/timeVSsizeN.png')

# iters per second vs matrix size N
fig = plt.figure()
ax = plt.subplot(111)
for version in distinct_versions:
	size_n = data[data.version == version].size_N
	iters_per_second = data[data.version == version].iters_per_second
	ax.plot(size_n, iters_per_second, label=f'{version} version', marker="o")
ax.legend()
plt.xlabel("size N")
plt.ylabel("iters/sec")
fig.savefig('plots'+'/itersVSsizeN.png')

# memory footprint vs matriz size N
fig = plt.figure()
ax = plt.subplot(111)
for version in distinct_versions:
	size_n = data[data.version == version].size_N
	memory = data[data.version == version].memory
	ax.plot(size_n, memory, label=f'{version} version', marker="o")
ax.legend()
plt.xlabel("size N")
plt.ylabel("Memory footprint (kBytes)")
fig.savefig('plots'+'/memoryVSsizeN.png')

# mlups vs matrix size N
fig = plt.figure()
ax = plt.subplot(111)
for version in distinct_versions:
	size_n = data[data.version == version].size_N
	mlups = data[data.version == version].mlups
	ax.plot(size_n, mlups, label=f'{version} version', marker="o")
ax.legend()
plt.xlabel("size N")
plt.ylabel("MLUPS")
fig.savefig('plots' +'/mlupsVSsizeN.png')

# total time vs Memory footprint
fig = plt.figure()
ax = plt.subplot(111)
for version in distinct_versions:
	time = data[data.version == version].time
	memory = data[data.version == version].memory
	ax.plot(memory, time, label=f'{version} version', marker="o")
ax.legend()
plt.xlabel("Memory footprint (kBytes)")
plt.ylabel("time (s)")
fig.savefig('plots'+'/timeVSmemory.png')

# total num iters vs memory footprint
fig = plt.figure()
ax = plt.subplot(111)
for version in distinct_versions:
	iters_per_second = data[data.version == version].iters_per_second
	memory = data[data.version == version].memory
	ax.plot(memory, iters_per_second, label=f'{version} version', marker="o")
ax.legend()
plt.xlabel("Memory footprint (kBytes)")
plt.ylabel("iters/sec")
fig.savefig('plots'+'/itersVSmemory.png')

# mlups vs memory footprint
fig = plt.figure()
ax = plt.subplot(111)
for version in distinct_versions:
	mlups = data[data.version == version].mlups
	memory = data[data.version == version].memory
	ax.plot(memory, mlups, label=f'{version} version', marker="o")
ax.legend()
plt.xlabel("Memory footprint (kBytes)")
plt.ylabel("MLUPS")
fig.savefig('plots' +'/mlupsVSmemory.png')

print("Success!. Images saved at /plots")
data.to_csv('plots' +'/latex.txt', header=True, index=False, sep='&')
exit()