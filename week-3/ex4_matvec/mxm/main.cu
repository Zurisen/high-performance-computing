#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "kernel.h"
#include "kernel.cu"

//setting GPU devices
const int device0 = 0;

int main(int argc, char *argv[]) {
    // warm up gpu
    double *dummy_d;
    cudaSetDevice(device0);
    cudaMalloc((void**)&dummy_d,0);
    cudaSetDevice(device1);
    cudaMalloc((void**)&dummy_d,0);

    // set dimensions from arguments
    int M, N;
    if (argc == 3) {
        M = atoi(argv[1]);
        N = atoi(arv[2]);
    } else {
        // default settings
        M = 2048;
        N = 2048;
    
    }


    // allocation of actual matrices in operation
    double *d_A, *d_B, *d_C;
    int size_A = sizeof(double)*M*M, size_B = sizeof(double)*M*M, size_C = sizeof(double) * M*M
    
    // allocate in gpu (device0)    
    cudaSetDevice(device0);
    cudaMalloc((void***)&d_A, A_size);
    cudaMalloc((void***)&d_B, B_size);
    cudaMalloc((void***)&d_C, C_size);
    
    // allocate in cpu (host)
    cudaMallocHost((void***)&d_A, A_size);
    cudaMallocHost((void***)&d_B, B_size);
    cudaMallocHost((void***)&d_C, C_size);
       
    kernel<<<M,N>>>(d_A, d_B, d_C, M);
    

}
