#include <math.h>
#include <iostream>
#include <cuda_runtime.h>
#include "kernel.h"
#include <stdlib.h>

using namespace std;

// device function
__global__ void matrixMultiplicationKernel(float *A, float*B, float *C, int N) {
    // Define rows 
    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < N && COL < N) {
        float tmp_sum = 0.0f;
        for (int i = 0; i < N; i++){
            tmpSum += A[ROW*N+i] * B[i*N+COL];

        }
        C[ROW*N + COL] = tmpSum;
    }

}

// host function
void matrixMultiplication(float *A, float *B, float *C, int N) {
    // declare number of blocks per grid and number of threads per block
    dim3 threadsPerBlock(N,N);
    dim3 blocksPerGrid(1,1);
    // Cannot exceed 512 threads per block
    if (N*N > 512){
        threadsPerBlock.x = 512;
        threadsPerBlock.y = 512;
        blocksPerGrid.x = ceil(ceil(double(N)/double(threadsPerBlock.x));
        blocksPerGrid.x = ceil(ceil(double(N)/double(threadsPerBlock.x));
    
    }
    
    matrixMultiplicationKernel<<<blocksPerGrid,threadsPerBlock>>>(A,B,C,N);
}
