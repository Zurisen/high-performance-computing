#!/bin/bash
#BSUB -J mm_batch
#BSUB -o mm_batch_%J.out
#BSUB -q hpcintro
#BSUB -n 16
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

#EXECUTABLE_J=poisson_j
EXECUTABLE_GS=poisson_gs
MAX_ITER=1000
TOLERANCE=0.1
TEMPERATURE=10.0
OUTPUT_TYPE=4
OUTFILE="results.txt"
rm $OUTFILE

echo "size(N) iter time(total) iter/s memory(kBytes)" >> $OUTFILE
for i in 50 75 100 125 150 175 200 
do
	for NUM_THREADS in 1 2 4 8 16
	do
		OMP_NUM_THREADS=$NUM_TRHEADS ./$EXECUTABLE_GS $i $MAX_ITER $TOLERANCE $TEMPERATURE $OUTPUT_TYPE >> $OUTFILE
	done
done

