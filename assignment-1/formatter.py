"""
Python scipt for plotting
"""

# import matplotlib.pyplot as plt
import glob,os

# create folder to dump the txt
if not os.path.exists('results_txt'):
    os.makedirs('results_txt')

# read all .out files
for outfile in glob.glob(os.path.join('*.out') ):
    f = open(outfile)
    new_file = open("results_txt/" + outfile + ".txt", "w")
    print(outfile + ".txt")


    # we only need first 5 lines of .out file
    # not the cleanest code
    Lines = f.readlines() 
    count = 1
    for line in Lines: 
    	print(line)
    	new_file.writelines(line)

    	if (count > 5):
    		break
    	count += 1
