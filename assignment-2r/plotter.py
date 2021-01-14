"""
plotter.py


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
	data = pd.read_csv(str(sys.argv[1], sep= " "))
except:
	print("Couldn't find valid file at", str(sys.argv[1]))
	exit()

parent_folder = str(sys.argv[1]).split("/")[0]
if not os.path.exists(parent_folder +'/plots'):
    os.makedirs(parent_folder +'/plots')

data.columns = ["num_threads", "size_N", "iterations", "time", "iters_per_second", "memory"]

distinct_num_threads = data.num_threads.distinct()
print(distinct_num_threads)


print("Success!")
exit()
