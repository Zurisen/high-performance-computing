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

data.columns = ["size_N", "iterations", "time", "iters_per_second", "memory", "num_threads"]

distinct_num_threads = data.num_threads.unique()

fig = plt.figure()
ax = plt.subplot(111)
for num_threads in distinct_num_threads:
	size_N = data[data.num_threads == num_threads].size_N
	time = data[data.num_threads == num_threads].time

	ax.plot(size_N, time, label=f'{num_threads} threads')
	# plt.title(' ')

ax.legend()
plt.xlabel("size N")
plt.ylabel("time (s)")
fig.savefig(parent_folder +'/plots' +'/timeVSsizeN.png')


fig = plt.figure()
ax = plt.subplot(111)
for num_threads in distinct_num_threads:
	size_N = data[data.num_threads == num_threads].size_N
	iters_per_second = data[data.num_threads == num_threads].iters_per_second

	ax.plot(size_N, iters_per_second, label=f'{num_threads} threads')
	# plt.title(' ')

ax.legend()
plt.xlabel("size N")
plt.ylabel("iters/sec")
fig.savefig(parent_folder +'/plots'+'/itersVSsizeN.png')

fig = plt.figure()
ax = plt.subplot(111)
for num_threads in distinct_num_threads:
	size_N = data[data.num_threads == num_threads].size_N
	memory = data[data.num_threads == num_threads].memory

	ax.plot(size_N, memory, label=f'{num_threads} threads')
	# plt.title(' ')

ax.legend()
plt.xlabel("size N")
plt.ylabel("Memory footprint (kBytes)")
fig.savefig(parent_folder +'/plots'+'/memoryVSsizeN.png')


fig = plt.figure()
ax = plt.subplot(111)
for num_threads in distinct_num_threads:
	time = data[data.num_threads == num_threads].time
	memory = data[data.num_threads == num_threads].memory

	ax.plot(memory, time, label=f'{num_threads} threads')
	# plt.title(' ')

ax.legend()
plt.xlabel("Memory footprint (kBytes)")
plt.ylabel("time (s)")
fig.savefig(parent_folder +'/plots'+'/timeVSmemory.png')

fig = plt.figure()
ax = plt.subplot(111)
for num_threads in distinct_num_threads:
	iters = data[data.num_threads == num_threads].iters_per_second
	memory = data[data.num_threads == num_threads].memory

	ax.plot(memory, iters, label=f'{num_threads} threads')
	# plt.title(' ')

ax.legend()
plt.xlabel("Memory footprint (kBytes)")
plt.ylabel("iters/sec")
fig.savefig(parent_folder +'/plots'+'/itersVSmemory.png')

print("Success!. Images saved at", parent_folder +'/plots')
exit()