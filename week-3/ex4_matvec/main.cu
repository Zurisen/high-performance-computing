
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// set multiple GPU devices for computing operation
const int device0 = 0;

// gpu (device) matrix * vector function
void __global__ matvec(double *y, double *A, double *x, int M, int N) {
        
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < M && j < N) atomicAdd(&y[i], A[i * N + j] * x[j]);
}

int main(int argc, char *argv[]) {
    
    // warm up
    printf("Warming up GPU...\n");
    double *dummy_d;
    cudaSetDevice(device0);
    cudaMalloc((void**)&dummy_d, 0);

    // input arguments
    int M, N;
    if (argc == 3) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
    }
    else {
        M = 2048;
        N = 2048;
    }

    //// Memory allocation 
    double *d0_A, *d0_b, *d0_C;
    double *h_A, *h_b, *h_C;
    int size_A = sizeof(double)*N*M, size_b = sizeof(double)*N, size_C = sizeof(double)*M;
    
    printf("Allocating device memory...\n");
    // Allocate A, b and C in Device 0
    cudaSetDevice(0);
    cudaMalloc((void**)&d0_A, size_A);
    cudaMalloc((void**)&d0_C, size_C);
    cudaMalloc((void**)&d0_b, size_b); 

    printf("Pinning host memory...\n");
     // Allocate A, b and C pinned in host memory
    cudaMallocHost((void**)&h_A, size_A);
    cudaMallocHost((void**)&h_b, size_b);
    cudaMallocHost((void**)&h_C, size_C);    


    /* HERE GOES THE MATRIX INIT */

    printf("Copying data from host to device...\n");
    // Copy data from host to device 0
    cudaMemcpy(d0_b, h_b, size_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d0_C, h_C, size_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d0_A, h_A, size_A, cudaMemcpyHostToDevice);

    // Invoke Kernel 
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Invoking kernel...\n");
    double begin_compute = omp_get_wtime();
    
        // operations on device 0
        cudaSetDevice(device0);   
        matvec<<<blocksPerGrid,threadsPerBlock>>>(d0_C, d0_A, d0_b, M, N);

    cudaDeviceSynchronize();

    double end_compute = omp_get_wtime() - begin_compute;
    printf("Operation finished!\t RUNTIME: %3.2f\n", end_compute);
    
    // Copy results back to host memory
    printf("Copying results back to host memory...\n");
    cudaSetDevice(device0);
    cudaMemcpy(h_C, d0_C, size_C, cudaMemcpyDeviceToHost);
    
    // Free memory
    printf("Liberating memory allocation...\n");
    cudaFreeHost(h_A), cudaFreeHost(h_b), cudaFreeHost(h_C); 
    cudaFree(d0_A), cudaFree(d0_b), cudaFree(d0_C); 
    printf("--- End of script ---\n");
    
    return(0);
}
