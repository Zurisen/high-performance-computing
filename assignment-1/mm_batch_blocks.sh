#!/bin/bash
# 02614 - High-Performance Computing, January 2018
# 
# batch script to run collect on a decidated server in the hpcintro
# queue
#
# Author: Bernd Dammann <bd@cc.dtu.dk>
#
#BSUB -J collector
#BSUB -o collector_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

module load studio

# define the driver name to use
# valid values: matmult_c.studio, matmult_f.studio, matmult_c.gcc or
# matmult_f.gcc
#
OUTFILE=matmult_out.blocks.gcc.txt
EXECUTABLE=matmult_c.gcc

# define the mkn values in the MKN variable
#
MKN="100 100 100"

# define the permutation type in PERM
#
PERM="blk"

# uncomment and set a reasonable BLKSIZE for the blk version
#
# BLKSIZE=1

# define the max no. of iterations the driver should use - adjust to
# get a reasonable run time.  You can get an estimate by trying this
# on the command line, i.e. "MFLOPS_MAX_IT=10 ./matmult_...." for the
# problem size you want to analyze.
#

for bls in 1 5 10 20 25 50 100
do
	BLKSIZE="${bls}"

	export MFLOPS_MAX_IT=1000

	# enable(1)/disable(0) result checking
	# export MATMULT_COMPARE=0

	printf "${MKN} ${PERM} ${BLKSIZE}" >> $OUTFILE
	# start the collect command with the above settings
    MATMULT_COMPARE=0 /usr/bin/time -f'%E' ./$EXECUTABLE $PERM $MKN $BLKSIZE >> $OUTFILE 2>&1
done