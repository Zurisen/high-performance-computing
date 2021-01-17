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

parent_folder = str(sys.argv[1]).split("/")[0]
if not os.path.exists(parent_folder +'/plots'):
    os.makedirs(parent_folder +'/plots')

data.columns = ["size_N", "iterations", "time", "iters_per_second", "memory"]

distinct_sizes_N = data.size_N.unique()

# create column of num_threads
num_threads = [1, 2, 4, 8, 16]*len(distinct_sizes_N)
data["num_threads"] = num_threads
distinct_num_threads = [1, 2, 4, 8, 16]

# compute speedup
baseline_times = data[data["num_threads"] == 1].time.to_numpy()
times_vect = data.time.to_numpy()

speedup = []
i = 0
for times in times_vect:
	if i<5:
		speedup.append(baseline_times[0]/times)
	elif i<10:
		speedup.append(baseline_times[1]/times)
	elif i<15:
		speedup.append(baseline_times[2]/times)
	elif i<20:
		speedup.append(baseline_times[3]/times)
	elif i<25:
		speedup.append(baseline_times[4]/times)
	elif i<30:
		speedup.append(baseline_times[5]/times)
	elif i<35:
		speedup.append(baseline_times[6]/times)
	else:
		print("error in speedup loop")
		break
	i = i + 1
data["speedup"] = speedup


# compute mlups
mlups = []
memories = data.memory.to_numpy()
iters = data.iterations.to_numpy()
for idx, memory in enumerate(memories):
	mlups.append(memory*iters[idx]/(8*1000*times_vect[idx]))
data["mlups"] = mlups

# print(data)

# total time vs matrix size N
fig = plt.figure()
ax = plt.subplot(111)
for num_threads in distinct_num_threads:
	size_N = data[data.num_threads == num_threads].size_N
	time = data[data.num_threads == num_threads].time
	ax.plot(size_N, time, label=f'{num_threads} threads', marker="o")

ax.legend()
plt.xlabel("size N")
plt.ylabel("time (s)")
fig.savefig(parent_folder +'/plots' +'/timeVSsizeN.png')

# iters per second vs matrix size N
fig = plt.figure()
ax = plt.subplot(111)
for num_threads in distinct_num_threads:
	size_N = data[data.num_threads == num_threads].size_N
	iters_per_second = data[data.num_threads == num_threads].iters_per_second
	ax.plot(size_N, iters_per_second, label=f'{num_threads} threads', marker="o")

ax.legend()
plt.xlabel("size N")
plt.ylabel("iters/sec")
fig.savefig(parent_folder +'/plots'+'/itersVSsizeN.png')

# memory footprint vs matriz size N
fig = plt.figure()
ax = plt.subplot(111)
for num_threads in distinct_num_threads:
	size_N = data[data.num_threads == num_threads].size_N
	memory = data[data.num_threads == num_threads].memory
	ax.plot(size_N, memory, label=f'{num_threads} threads', marker="o")

ax.legend()
plt.xlabel("size N")
plt.ylabel("Memory footprint (kBytes)")
fig.savefig(parent_folder +'/plots'+'/memoryVSsizeN.png')

# mlups vs matrix size N
fig = plt.figure()
ax = plt.subplot(111)
for num_threads in distinct_num_threads:
	size_N = data[data.num_threads == num_threads].size_N
	mlups = data[data.num_threads == num_threads].mlups
	ax.plot(size_N, mlups, label=f'{num_threads} threads', marker="o")

ax.legend()
plt.xlabel("size N")
plt.ylabel("MLUPS")
fig.savefig(parent_folder +'/plots' +'/mlupsVSsizeN.png')

# total time vs Memory footprint
fig = plt.figure()
ax = plt.subplot(111)
for num_threads in distinct_num_threads:
	time = data[data.num_threads == num_threads].time
	memory = data[data.num_threads == num_threads].memory
	ax.plot(memory, time, label=f'{num_threads} threads', marker="o")

ax.legend()
plt.xlabel("Memory footprint (kBytes)")
plt.ylabel("time (s)")
fig.savefig(parent_folder +'/plots'+'/timeVSmemory.png')

# total num iters vs memory footprint
fig = plt.figure()
ax = plt.subplot(111)
for num_threads in distinct_num_threads:
	iters = data[data.num_threads == num_threads].iters_per_second
	memory = data[data.num_threads == num_threads].memory
	ax.plot(memory, iters, label=f'{num_threads} threads', marker="o")

ax.legend()
plt.xlabel("Memory footprint (kBytes)")
plt.ylabel("iters/sec")
fig.savefig(parent_folder +'/plots'+'/itersVSmemory.png')

# mlups vs memory footprint
fig = plt.figure()
ax = plt.subplot(111)
for num_threads in distinct_num_threads:
	memory = data[data.num_threads == num_threads].memory
	mlups = data[data.num_threads == num_threads].mlups
	ax.plot(memory, mlups, label=f'{num_threads} threads', marker="o")

ax.legend()
plt.xlabel("Memory footprint (kBytes)")
plt.ylabel("MLUPS")
fig.savefig(parent_folder +'/plots' +'/mlupsVSmemory.png')

# speedup vs num_threads
fig = plt.figure()
ax = plt.subplot(111)
memories = data.memory.unique()
for mem in memories:
	speedup = data[data.memory == mem].speedup
	num_threads = data[data.memory == mem].num_threads
	ax.plot(num_threads, speedup, label=f'{mem} kBytes', marker="o")

ax.legend()
plt.xlabel("Number of threads")
plt.ylabel("Speedup")
fig.savefig(parent_folder +'/plots' +'/speedupVSnumthreads.png')


print("Success!. Images saved at", parent_folder +'/plots')
data.to_csv(parent_folder +'/plots' +'latex.txt', header=True, index=False, sep='&')
exit()

