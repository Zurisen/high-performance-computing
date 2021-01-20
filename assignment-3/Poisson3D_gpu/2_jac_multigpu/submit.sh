#!/bin/bash
#BSUB -J jacobi
#BSUB -o jacobi_%J.out
#BSUB -e jacobi_%J.err
#BSUB -q hpcintrogpu
#BSUB -gpu "num=1:mode=exclusive_process:mps=yes"
#BSUB -n 16
#BSUB -W 590
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=1024MB]"

EXECUTABLE_J=poisson_j
EXECUTABLE_GS=poisson_gs
MAX_ITER=10
TEMPERATURE=10.0
OUTPUT_TYPE=4
OUTFILE="results.txt"
rm $OUTFILE

echo "Elapsed Time" >> $OUTFILE
for i in 10 20 50 75 100 125 150
do
	./$EXECUTABLE_J $i $MAX_ITER $TEMPERATURE $OUTPUT_TYPE >> $OUTFILE
done

