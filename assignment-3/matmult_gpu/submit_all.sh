#!/bin/bash
#BSUB -q hpcintrogpu
#BSUB -J matmul_gpu
#BSUB -o matmul_gpu%J.out
#BSUB -e matmul_gpu%J.err
#BSUB -n 1
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 60
#BSUB -gpu "num=1:mode=exclusive_process"

module load cuda/11.1
module load gcc/9.2.0
OUTFILE="results_final.txt"
rm OUTFILE
EXECUTABLE=matmult_f.nvcc
declare -A ITER
ITER=([64]=100 [128]=100 [256]=100 [512]=100 [1024]=10 [2048]=1 [4096]=1 [8192]=1 [10240]=1 )
for method in lib gpu2 gpu3 gpu4 gpu5 gpulib
do
	for size in 64 128 256 512 1024 2048 4096 8192
	do
		MFLOPS_MAX_IT=${ITER[${size}]} MATMULT_COMPARE=1 ./matmult_f.nvcc $method $size $size $size >> $OUTFILE
	done
done
