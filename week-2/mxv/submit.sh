#!/bin/bash

#BSUB -J mm_batch
#BSUB -o mm_batch_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

module load studio

EXECUTABLE=mxv_run
OUTFILE=results_out.txt
rm $(OUTFILE)

for num_threads in 1 2 3 4 5 6 7 8
do
	echo "CPU threads: $(num_threads)" >> $(OUTFILE)
	start=`date + %s%N`
	OMP_NUM_THREADS=$(num_threads) ./$(EXECUTABLE)
	end=`date + %s%N`
	echo "Time: $($(end)-$(start)) ns" >> $(OUTFILE)
	echo "" >> $(OUTFILE)
done