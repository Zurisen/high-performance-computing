#!/bin/bash
#BSUB -J mm_batch
#BSUB -o mm_batch_%J.out
#BSUB -n 16
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

EXECUTABLE_J=poisson_j
EXECUTABLE_GS=poisson_gs
MAX_ITER=10000
TEMPERATURE=10.0
OUTPUT_TYPE=4
OUTFILE="results.txt"
NUM_THREADS=1
rm $OUTFILE

echo "size(N) iter time(total) iter/s memory(kBytes)" >> $OUTFILE
for i in 10 20 50 75 100 125 150
do
	OMP_NUM_THREADS=$NUM_THREADS ./$EXECUTABLE_J $i $MAX_ITER $TEMPERATURE $OUTPUT_TYPE >> $OUTFILE
done

