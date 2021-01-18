
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

// set multiple GPU devices for computing operation
const int device0 = 0;
const int device1 = 1;

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
    cudaSetDevice(device1);
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
    double *d0_A, *d0_b, *d0_C, *d1_A, *d1_b, *d1_C;
    double *h_A, *h_b, *h_C;
    int size_A = sizeof(double)*N*M, size_b = sizeof(double)*N, size_C = sizeof(double)*M;
    
    printf("Allocating device memory...\n");
    // Allocate A, b and C in Device 0
    cudaSetDevice(0);
    cudaMalloc((void**)&d0_A, size_A/2);
    cudaMalloc((void**)&d0_C, size_C/2);
    cudaMalloc((void**)&d0_b, size_b);    
    // Allocate A, b and C in Device 1
    cudaSetDevice(1);
    cudaMalloc((void**)&d1_A, size_A/2);
    cudaMalloc((void**)&d1_C, size_C/2);
    cudaMalloc((void**)&d1_b, size_b);
   
    printf("Pinning host memory...\n");
    // Allocate A, b and C pinned in host memory
    cudaMallocHost((void**)&h_A, size_A);
    cudaMallocHost((void**)&h_b, size_b);
    cudaMallocHost((void**)&h_C, size_C);    


    /* HERE GOES THE MATRIX INIT */
    h_A[0]=1.0; h_A[1]=1.0; h_A[2]=1.0; h_A[3]=1.0;
    h_b[0]=1.0; h_b[1]=1.0;

    printf("Copying data from host to device...\n");
    // Copy data from host to device 0
    cudaMemcpy(d0_b, h_b, size_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d0_C, h_C, size_C/2, cudaMemcpyHostToDevice);
    cudaMemcpy(d0_A, h_A, size_A/2, cudaMemcpyHostToDevice);
    // Copy data from host to device 1
    cudaMemcpy(d1_b, h_b, size_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d1_C, h_C + size_C/2, size_C/2, cudaMemcpyHostToDevice);
    cudaMemcpy(d1_A, h_A + size_A/2, size_A/2, cudaMemcpyHostToDevice);
    
 

    // Invoke Kernel 
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Invoking kernel...\n");
    double begin_compute = omp_get_wtime();
    
        // operations on device 0
        cudaSetDevice(device0);   
        matvec<<<blocksPerGrid,threadsPerBlock>>>(d0_C, d0_A, d0_b, M/2, N);
        // operations on device 1
        cudaSetDevice(device1);   
        matvec<<<blocksPerGrid,threadsPerBlock>>>(d1_C, d1_A, d1_b, M/2, N);

    cudaDeviceSynchronize();
    cudaSetDevice(device0);
    cudaDeviceSynchronize();

    double end_compute = omp_get_wtime() - begin_compute;
    printf("Operation finished!\t RUNTIME: %3.2f\n", end_compute);
    
    // Copy results back to host memory
    printf("Copying results back to host memory...\n");
    cudaSetDevice(device0);
    cudaMemcpy(h_C, d0_C, size_C/2, cudaMemcpyDeviceToHost);
    cudaSetDevice(device1);
    cudaMemcpy(h_C + size_C/2, d1_C, size_C/2, cudaMemcpyDeviceToHost);
    
    printf("%3.2f\t %3.2f\t %3.2f\t %3.2f\n", h_C[0], h_C[1], h_C[2], h_C[3]);
    // Free memory
    printf("Liberating memory allocation...\n");
    cudaFreeHost(h_A), cudaFreeHost(h_b), cudaFreeHost(h_C); 
    cudaFree(d0_A), cudaFree(d0_b), cudaFree(d0_C); 
    cudaFree(d1_A), cudaFree(d1_b), cudaFree(d1_C); 
    printf("--- End of script ---\n");
    
    return(0);
}
