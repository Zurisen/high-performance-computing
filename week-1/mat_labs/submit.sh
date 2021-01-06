#!/bin/bash
#BSUB -J matmult
#BSUB -o matmult_%J.out
#BSUB -q hpcintro
#BSUB -W 2 -R "rusage[mem=2GB]"

./madd.gcc
