#!/bin/bash
#BSUB -q hpcintrogpu
#BSUB -J poisson_gpu
#BSUB -o poisson_j_gpu_%J.out
#BSUB -n 1
#BSUB -R "rusage[mem=5GB]"
#BSUB -W 15
#BSUB -gpu "num=1:mode=exclusive_process"

module load cuda/11.1
module load gcc/9.2.0

OUTFILE="specs.txt"

nvidia-smi -q>> $OUTFILE
