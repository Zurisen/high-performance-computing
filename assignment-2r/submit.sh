#!/bin/bash
#BSUB -J mm_batch
#BSUB -o mm_batch_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

EXECUTABLE=poisson_j
MAX_ITER=1000
TOLERANCE=0.1
TEMPERATURE=10.0
OUTPUT_TYPE=3

for i in 100 200 500 
do
	./$EXECUTABLE $i $MAX_ITER $TOLERANCE $TEMPERATURE $OUTPUT_TYPE 
done

