#!/bin/bash
# 02614 - High-Performance Computing, January 2018
# 
# batch script to run matmult on a decidated server in the hpcintro
# queue
#
# Author: Bernd Dammann <bd@cc.dtu.dk>
#
#BSUB -J mm_batch
#BSUB -o mm_batch_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

OUTFILE=matmult_out.perm.gcc.txt

# define the driver name to use
# valid values: matmult_c.studio, matmult_f.studio, matmult_c.gcc or
# matmult_f.gcc
#
EXECUTABLE=matmult_c.gcc

# define the mkn values in the MKN variable
#
declare -A size_its

size_its = ( [10]=5000000 [50]=30000 [100]=4000 [500]=20 [1000]=4 [2000]=1 [5000]=1)
for size in 10 50 100 500 1000 2000 5000
do 
	MKN = "${size} ${size} ${size}"

	for permutation in kmn mkn nkm knm mnk nkm
	do
		# permutation type
		PERM="${permutation}"

		# uncomment and set a reasonable BLKSIZE for the blk version
		#
		# BLKSIZE=1
		export MFLOPS_MAX_IT=${size_its[${size}]}

		# enable(1)/disable(0) result checking
		# export MATMULT_COMPARE=0

		printf "${size} ${PERM} " >> $OUTFILE
		# start the collect command with the above settings
    	MATMULT_COMPARE=0 /usr/bin/time -f'%E' ./$EXECUTABLE $PERM $MKN $BLKSIZE >> $OUTFILE 2>&1
    done
done