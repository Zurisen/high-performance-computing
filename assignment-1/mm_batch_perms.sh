#!/bin/bash
# 02614 - High-Performance Computing, January 2018

# bash file to the test performance for
# different permutations of the loops

OUTFILE=matmult_out.perm.gcc.txt
EXECUTABLE=matmult_c.gcc

declare -A MAX_ITS

MAX_ITS=( [10]=5000000 [50]=30000 [100]=4000 [500]=20 [1000]=4 [2000]=1 [5000]=1 )

for size in 10 50 100 500 1000 2000 5000
do 
	MKN="${size} ${size} ${size}"

	for permutation in kmn mkn nkm knm mnk nkm
	do
		# permutation type
		PERM="${permutation}"

		# uncomment and set a reasonable BLKSIZE for the blk version
		# BLKSIZE=1
		export MFLOPS_MAX_IT=${MAX_ITS[${size}]}

		printf "${size} ${PERM} " >> $OUTFILE
    	MATMULT_COMPARE=0 /usr/bin/time -f'%E' ./$EXECUTABLE $PERM $MKN $BLKSIZE >> $OUTFILE 2>&1
    done
done