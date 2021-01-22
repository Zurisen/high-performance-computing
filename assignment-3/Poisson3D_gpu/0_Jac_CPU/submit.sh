#!/bin/bash
#BSUB -J mm_batch
#BSUB -o mm_batch_%J.out
#BSUB -q hpcintro
#BSUB -n 16
#BSUB -R "rusage[mem=2048]"
#BSUB -W 45

EXECUTABLE_J=poisson_j
EXECUTABLE_GS=poisson_gs
MAX_ITER=100
TOLERANCE=0.1
TEMPERATURE=10.0
OUTPUT_TYPE=4
OUTFILE="results.txt"
rm $OUTFILE

module load studio

echo "size(N) iter time(total) iter/s memory(kBytes)" >> $OUTFILE


for i in 64 128 256 512 
do
	for j in 1 2 4 8 16
	do
	OMP_NUM_THREADS=$j ./$EXECUTABLE_J $i $MAX_ITER $TOLERANCE $TEMPERATURE $OUTPUT_TYPE >> $OUTFILE
	done
done

