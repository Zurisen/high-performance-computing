
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

// set multiple GPU devices for computing operation
const int device0 = 0;

// gpu (device) matrix * vector function
__global__ void matvec(double *y, double *A, double *x, int M, int N) {
        
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    printf("hello world");    
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
    //cudaSetDevice(0);
    cudaMalloc((void**)&d0_A, size_A);
    cudaMalloc((void**)&d0_C, size_C);
    cudaMalloc((void**)&d0_b, size_b); 

    printf("Pinning host memory...\n");
     // Allocate A, b and C pinned in host memory
    cudaMallocHost((void**)&h_A, size_A);
    cudaMallocHost((void**)&h_b, size_b);
    cudaMallocHost((void**)&h_C, size_C);    


    /* HERE GOES THE MATRIX INIT */
    double init_A = 2.0, init_b = 2.0;
    for (int i = 0; i < M*N; i++) h_A[i] = init_A;
    for (int i = 0; i < N; i++) h_b[i] = init_b;

    printf("Copying data from host to device...\n");
    // Copy data from host to device 0
    cudaMemcpyAsync(d0_b, h_b, size_b, cudaMemcpyHostToDevice);
    cudaMemcpyAsync(d0_A, h_A, size_A, cudaMemcpyHostToDevice);

    // Invoke Kernel 
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 dimGrid((((M)+BLOCK_SIZE-1) / BLOCK_SIZE), (((N)+BLOCK_SIZE-1) / BLOCK_SIZE));
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    
    printf("Invoking kernel...\n");
    cudaEventRecord(start); 
        // operations on device 0
        //cudaSetDevice(device0);   
        matvec<<<dimGrid,dimBlock>>>(d0_C, d0_A, d0_b, M, N);
    cudaEventRecord(stop);
    cudaDeviceSynchronize();
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Operation finished!  GPU runtime (ms): %3.6f\n\n", milliseconds);
    
    // Copy results back to host memory
    printf("Copying results back to host memory...\n");
    //cudaSetDevice(device0);
    cudaMemcpy(h_C, d0_C, size_C, cudaMemcpyDeviceToHost);

    //print first terms of the result
    printf("\n%3.2f\t\n%3.2f\t\n%3.2f\t\n%3.2f\n%3.2f\t\n%3.2f\t\n%3.2f\t\n%3.2f\n", h_C[0], h_C[1], h_C[2], h_C[3], h_C[4], h_C[5], h_C[6], h_C[7]);    
    
    // Free memory
    printf("Liberating memory allocation...\n");
    cudaFreeHost(h_A), cudaFreeHost(h_b), cudaFreeHost(h_C); 
    cudaFree(d0_A), cudaFree(d0_b), cudaFree(d0_C); 
    
    printf("--- End of script ---\n");
   
     
    return(0);
}
