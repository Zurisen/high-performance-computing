#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d_gpu.h"
#include "func.h"
#include "jacobi.h"

int main(int argc, char *argv[]){
    
    double start_T = atof(argv[])
    int N;
    int N2 = N * N;
    // Wake up gpu
    cudaSetDevice(0);
    double *d_dummy;
    cudaMalloc((void**)&d_dummy,0;

    double *d_u, *d_uOld, *d_uSwap, *d_f;
    double *h_u, *h_uOld, *h_uSwap, *h_f;
    double size = N * N * N * sizeof(double);

    // Device memory allocation 
    cudaMalloc((void**)&d_u, size);
    cudaMalloc((void**)&d_uOld, size);
    cudaMalloc((void**)&d_uSwap, size);
    cudaMalloc((void**)&d_f, size);

    // Pinning memory in host
    cudaMallocHost((void**)&h_u, size);
    cudaMallocHost((void**)&h_uOld, size);
    cudaMallocHost((void**)&h_uSwap, size);
    cudaMallocHost((void**)&h_f, size);

    // Initialization of the arrays
    u_init(h_u, N, N2, start_T); 
    u_init(h_uOld, N, N2, start_T); 
    u_init(h_uSwap, N, N2, start_T); 
    f_init(f, N, N2);



}
