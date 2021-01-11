#!/bin/bash

#BSUB -J parallel
#BSUB -o parallel_%J.out
#BSUB -q hpcintro
#BSUB -n 8
#BSUB -R "rusage[mem=2048]"
#BSUB -W 15

module load studio

EXECUTABLE=paralelizer
OUTFILE=outfile.txt
rm $OUTFILE

echo "CPU threads = 1" >> $OUTFILE
start=`date +%s%N`
OMP_NUM_THREADS=1 ./$EXECUTABLE >> $OUTFILE
end1=`date +%s%N`
echo "Runtime $(($end1-$start)) nsec" >> $OUTFILE
echo "" >> $OUTFILE

echo "CPU threads = 2" >> $OUTFILE
OMP_NUM_THREADS=2 ./$EXECUTABLE >> $OUTFILE
end2=`date +%s%N`
echo "Runtime $(($end2-$end1)) nsec" >> $OUTFILE
echo "" >> $OUTFILE

echo "CPU threads = 4" >> $OUTFILE
OMP_NUM_THREADS=4 ./$EXECUTABLE >> $OUTFILE
end3=`date +%s%N`
echo "Runtime $(($end3-$end2)) nsec" >> $OUTFILE
echo "" >> $OUTFILE

echo "CPU threads = 8" >> $OUTFILE
OMP_NUM_THREADS=8 ./$EXECUTABLE >> $OUTFILE
end4=`date +%s%N`
echo "Runtime $(($end4-$end3)) nsec" >> $OUTFILE
