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

OUTFILE="results.txt"
rm $OUTFILE
EXECUTABLE=matmult_f.nvcc
method=gpu1
declare -A ITER
ITER=( [512]=100 [1024]=10 [2048]=1 [4096]=1 [8192]=1 [10240]=1 )


for size in 512 1024 2048 4096 8192 #10240
do
    printf "${method} "
    printf "${size} "
    MFLOPS_MAX_IT=${ITER[${size}]} MATMULT_COMPARE=0 ./matmult_f.nvcc $method $size $size $size >> $OUTFILE
done

