#!/bin/bash
#BSUB -J mm_batch
#BSUB -o mm_batch_%J.out
#BSUB -q hpcintro
#BSUB -n 1
#BSUB -R "rusage[mem=2048]"
#BSUB -W 45

EXECUTABLE=poisson_j
HWCOUNT="-h dch,on,dcm,on,l2h,on,l2m,on"
ITER=100
TOLERANCE=30
INIT_T=10.0
OUTPUT_TYPE=4

module load studio



for i in 20 50 80 100
do

	for iter in 10 50 100 500 1000 5000 10000
	do	
		FILENAME="${EXECUTABLE}_${i}_${iter}_${INIT_T}"
		collect -o ./results/$EXECUTABLE/${FILENAME}.er $HWCOUNT ./$EXECUTABLE $i $ITER $TOLERANCE $INIT_T $OUTPUT_TYPE
		mv poisson_res_${i}.vtk ./results/$EXECUTABLE/${FILENAME}.vtk
		erprint -func ./results/$EXECUTABLE/${FILENAME}.er >> ./results/$EXECUTABLE/${FILENAME}.txt
	done
done
