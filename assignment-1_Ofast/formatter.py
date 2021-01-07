"""
Python scipt to extract data from the .out files
in /results and create the txt in results_text/ to
later plot them
"""

NUM_LINES = 5; # num of lines from .out to extract

import glob,os

# create folder to save the txt
if not os.path.exists('results_txt'):
    os.makedirs('results_txt')

# read all .out files
for outfile in glob.glob(os.path.join('*.out') ):
    f = open(outfile)
    new_file = open("results_txt/" + outfile + ".txt", "w")
    print(outfile + ".txt")

    # we only need first 5 lines of .out file
    Lines = f.readlines() 
    count = 1
    for line in Lines: 
    	print(line)
    	new_file.writelines(line)

    	if (count > NUM_LINES):
    		break
    	count += 1
