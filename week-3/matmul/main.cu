#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(double *A, double *B, double *C, int M, int L, int N)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < L; ++e)
        Cvalue += A[row * L + e]
                * B[e * N + col];
    C[row * N + col] = Cvalue;
}

int main(int argc, char *argv[]){
    int M, L, N;

    if (argc == 4) {
        M = atoi(argv[1]);
        L = atoi(argv[2]);
        N = atoi(argv[3]);
    }
    else {
        //default values
        M = 2;
        L = 2;
        N = 2;
    }

    printf("Warming up GPU...\n");
    cudaSetDevice(0);
    double *d_dummy;
    cudaMalloc((void**)&d_dummy, 0);

    printf("Allocating GPU memory...\n");
    double *d_A, *d_B, *d_C, *h_A, *h_B, *h_C;
    double size_A = M*L*sizeof(double), size_B = N*L*sizeof(double), size_C = N*M*sizeof(double);
    
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    printf("Pinning host memory...\n");
    cudaMallocHost((void**)&h_A, size_A);
    cudaMallocHost((void**)&h_B, size_B);
    cudaMallocHost((void**)&h_C, size_C);

    /* HERE GOES THE MATRIX INIT */
    double init_A = 2.0, init_B = 2.0;
    for (int i = 0; i < M*N; i++) h_A[i] = init_A;
    for (int i = 0; i < N*L; i++) h_B[i] = init_B;

    printf("Copying data from host to GPU memory...\n");
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);   
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);   

    printf("Invoking kernel...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    // dim3 dimGrid(N/dimBlock.x, M/dimBlock.y);
    dim3 dimGrid((((M)+BLOCK_SIZE-1) / BLOCK_SIZE), (((N)+BLOCK_SIZE-1) / BLOCK_SIZE));
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
        cudaEventRecord(start);
        MatMulKernel<<<dimGrid,dimBlock>>>(d_A, d_B, d_C, M, N, L);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);   

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Operation finished!  GPU runtime (ms): %3.6f\n\n", milliseconds);
    
    printf("Copying results back to host memory...\n");
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    printf("\n%3.2f\t\n%3.2f\t\n%3.2f\t\n%3.2f\n%3.2f\t\n%3.2f\t\n%3.2f\t\n%3.2f\n", h_C[0], h_C[1], h_C[2], h_C[3], h_C[4],
                                                                                     h_C[5], h_C[6], h_C[7], h_C[8]);
    printf("Liberating memory allocation...\n");
    cudaFreeHost(h_A), cudaFreeHost(h_B), cudaFreeHost(h_C);
    cudaFree(d_A), cudaFree(d_B), cudaFree(d_C);

    printf("--- End of script ---");
    
    return(0);
}
