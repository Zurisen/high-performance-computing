#!/bin/bash
# 02614 - High-Performance Computing

#BSUB -J collector
#BSUB -o collector_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

# bash file to the test performance for
# different block sizes

module load studio

OUTFILE=matmult_out.blocks.gcc.txt
EXECUTABLE=matmult_c.gcc

# define MKN sizes 
MKN="3000 1000 2000"

# set the type of permutation (blk for blocks)
PERM="blk"

# loop over the different sizes of blocks
for bls in 100 200 300 
do
	BLKSIZE="${bls}"

	# define the max no. of iterations the driver should use - adjust to
	# get a reasonable run time.  You can get an estimate by trying this
	# on the command line, i.e. "MFLOPS_MAX_IT=10 ./matmult_...." for the
	# problem size you want to analyze.
	export MFLOPS_MAX_IT=10

	printf "${MKN} ${PERM} ${BLKSIZE}" >> $OUTFILE
    MATMULT_COMPARE=0 /usr/bin/time -f'%E' ./$EXECUTABLE $PERM $MKN $BLKSIZE >> $OUTFILE 2>&1
done
