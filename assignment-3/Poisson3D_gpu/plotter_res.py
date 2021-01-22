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

parent_folder = str(sys.argv[1]).split("/")[0]
if not os.path.exists(parent_folder +'/plots'):
    os.makedirs(parent_folder +'/plots')


data.columns = ["size_N", "iterations", "time", "iters_per_second"]
distinct_sizes_N = data.size_N.unique()
def compute_memory(data):
	memory = (data.size_N**3)*8/1000
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

# print(data)

# total time vs matrix size N
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(data.size_N, data.time, marker="o")
plt.xlabel("size N")
plt.ylabel("time (s)")
fig.savefig(parent_folder +'/plots' +'/timeVSsizeN.png')

# iters per second vs matrix size N
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(data.size_N, data.iters_per_second, marker="o")
plt.xlabel("size N")
plt.ylabel("iters/sec")
fig.savefig(parent_folder +'/plots'+'/itersVSsizeN.png')

# memory footprint vs matriz size N
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(data.size_N, data.memory, marker="o")
plt.xlabel("size N")
plt.ylabel("Memory footprint (kBytes)")
fig.savefig(parent_folder +'/plots'+'/memoryVSsizeN.png')

# mlups vs matrix size N
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(data.size_N, data.mlups, marker="o")
plt.xlabel("size N")
plt.ylabel("MLUPS")
fig.savefig(parent_folder +'/plots' +'/mlupsVSsizeN.png')

# total time vs Memory footprint
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(data.memory, data.time, marker="o")
plt.xlabel("Memory footprint (kBytes)")
plt.ylabel("time (s)")
fig.savefig(parent_folder +'/plots'+'/timeVSmemory.png')

# total num iters vs memory footprint
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(data.memory, data.iters_per_second, marker="o")
plt.xlabel("Memory footprint (kBytes)")
plt.ylabel("iters/sec")
fig.savefig(parent_folder +'/plots'+'/itersVSmemory.png')

# mlups vs memory footprint
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(data.memory, data.mlups, marker="o")
plt.xlabel("Memory footprint (kBytes)")
plt.ylabel("MLUPS")
fig.savefig(parent_folder +'/plots' +'/mlupsVSmemory.png')

print("Success!. Images saved at", parent_folder +'/plots')
data.to_csv(parent_folder +'/plots' +'latex.txt', header=True, index=False, sep='&')
exit()
