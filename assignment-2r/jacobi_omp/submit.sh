#!/bin/bash
#BSUB -J mm_batch
#BSUB -o mm_batch_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

EXECUTABLE_J=poisson_j
EXECUTABLE_GS=poisson_gs
MAX_ITER=100
TOLERANCE=0.1
TEMPERATURE=10.0
OUTPUT_TYPE=4
OUTFILE="results.txt"
rm $OUTFILE

echo "size(N) iter time(total) iter/s memory(kBytes)" >> $OUTFILE
for i in 50 75 100 125 150 175 200 
do
	./$EXECUTABLE_J $i $MAX_ITER $TOLERANCE $TEMPERATURE $OUTPUT_TYPE >> $OUTFILE
done

