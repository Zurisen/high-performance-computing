
#include <stdio.h>
#include <stdlib.h>
#include "func.h" // for init_2d()
#include <math.h> // for pow()
#include <omp.h>
#include <cblas.h> // for matmult_lib()

#define BLOCK_SIZE 16

// set multiple GPU devices for computing operation
const int device0 = 0;


/* Native CBLAS CPU implementation of matrix multiplication */
void matmult_lib(int M, int N, int K, double *A, double *B, double *C) {
        double alpha = 1.0, beta = 0.0;
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,A,K,B,N,beta,C, N);
}

/* part 1: sequential implementation in GPU (single thread) */
__global__ void matmult_gpu1_kernel(int M, int N, int K, double* A, double *B, double* C) {
    double temp = 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N){ // ensure that the extra threads do not do any work
        for (int step = 0; step < K; step++) {
            temp += A[i*K + step] * B[step*N + j];
        }
        C[i*N + j] = temp;
    }
}

void matmult_gpu1(int M, int N, int K, double* A, double *B, double* C) {
    // Define grid and threads per block
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block

    matmult_gpu1_kernel<<<1,1>>>(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize():
}

/* part 2: naive implementation in GPU (one thread per element in C) */
__global__ void matmult_gpu2_kernel(int M, int N, int K, double* A, double* B, double* C) {
    double temp = 0;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < M && j < N){ // ensure that the extra threads do not do any work
        for (int step = 0; step < K; step++) {
            temp += A[i*K + step] * B[step*N + j];
        }
        C[i*N + j] = temp;
    }
}

void matmult_gpu2(int M, int N, int K, double* A, double *B, double* C) {
    // Define grid and threads per block
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    int BLOCK_SIZE = 16;
    dim3 blocksPerGrid(((M+BLOCK_SIZE-1) / BLOCK_SIZE), ((N+BLOCK_SIZE-1) / BLOCK_SIZE));
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    matmult_gpu2_kernel<<<blocksPerGrid,threadsPerBlock>>>(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize():
}

/* part 3: GPU (thread computes 2 elements of C) */
__global__ void matmult_gpu3_kernel(int M, int N, int K, double* A, double* B, double* C, int stride) {
    double temp = 0;
    int i = (blockIdx.x * blockDim.x + threadIdx.x)*stride;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    for (int s = 0; s < stride; s++) { // does right neighbour
        if ((s + i) < M && j < N){ // ensure that the extra threads do not do any work
            for (int step = 0; step < K; step++) {
                temp += A[(s + i)*K + step] * B[step*N + j];
            }
            C[(s + i)*N + j] = temp;
        }
    }
}

void matmult_gpu3(int M, int N, int K, double* A, double *B, double* C) {
    // Define grid and threads per block
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    int BLOCK_SIZE = 16;
    int stride = 2;
    dim3 blocksPerGrid(((M+BLOCK_SIZE-1) / (stride * BLOCK_SIZE)), ((N+BLOCK_SIZE-1) / BLOCK_SIZE));
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    matmult_gpu3_kernel<<<blocksPerGrid,threadsPerBlock>>>(M, N, K, d_A, d_B, d_C, stride);
    cudaDeviceSynchronize():
}
/* part 4: GPU (thread computes >2 elements of C) */
__global__ void matmult_gpu4_kernel(int M, int N, int K, double* A, double* B, double* C, int stride_row, int stride_col) {
    double temp = 0;
    int i = (blockIdx.x * blockDim.x + threadIdx.x)*stride_row;
    int j = (blockIdx.y * blockDim.y + threadIdx.y)*stride_col;

    for (int s_row = 0; s_row < stride_row; s_row++) {
        for (int s_col = 0; s_col < stride_col; scol++) {
            if ((s_row + i) < M && (j + s_col) < N){ // ensure that the extra threads do not do any work
                for (int step = 0; step < K; step++) {
                    temp += A[(s_row + i)*K + step] * B[step*N + (j + s_col)];
                }
                C[(s_row + i)*N + (j + s_col)] = temp;
            }
        }
    }
}

void matmult_gpu4(int M, int N, int K, double* A, double *B, double* C) {
    // Define grid and threads per block
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    int BLOCK_SIZE = 16;
    int stride_row = 2;
    int stride_col = 2;
    dim3 blocksPerGrid(((M+BLOCK_SIZE-1) / (stride_row * BLOCK_SIZE)), ((N+BLOCK_SIZE-1) / (stride_col * BLOCK_SIZE)));
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    matmult_gpu3_kernel<<<blocksPerGrid,threadsPerBlock>>>(M, N, K, d_A, d_B, d_C, stride_row, stride_col);
    cudaDeviceSynchronize():
}
/* part 5: GPU (shared memory version) */
__global__ void matmult_gpu5(int M, int N, int K, double** A, double **B, double** C) {

}

/* part 6: DGEMM function for GPUs, NVIDIA */
__global__ void matmult_gpulib(int M, int N, int K, double **A, double **B, double **C) {
        double alpha = 1.0, beta = 0.0;
        cublasDgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,alpha,A[0],K,B[0],N,beta,C[0],N);
}


/*
The drivers still take the same command line arguments:

matmult_f.nvcc type m n k [bs]

where m, n, k are the parameters defining the matrix sizes, bs is the
optional blocksize for the block version, and type can be one of:

nat     - the native/naïve version
lib     - the library version (note that this now calls a multithreaded library)
gpu1    - the first gpu version
gpu2    - the second gpu version
gpu3    - the third gpu version
gpu4    - the fourth gpu version
gpu5    - the fifth gpu version
gpu6    - the sixth gpu version
gpulib  - the CUBLAS library version

as well as blk, mnk, nmk, ... (the permutations).
*/
int main(int argc, char *argv[]) {
    // input arguments
    if (argc != 5) {
        printf("Error: Missing command line arguements.");
        exit(1);
    }
    int M, N, K;
    char* type;
    double max_val = 10.0;
    double time_start, time_end, time_IO_1, time_IO_2;
    double time_start_compute, time_end_compute, total_time_compute;

    type = atoi(argv[1]);
    M = atoi(argv[2]);
    N = atoi(argv[3]);
    K = atoi(argv[4]);

    // warm up
    printf("Warming up GPU...\n");
    double *dummy_d;
    cudaSetDevice(device0);
    cudaMalloc((void**)&dummy_d, 0);

    double *d_A, *d_B, *d_C; // Device variables
    double *h_A, *h_B, *h_C; // host vairables
    int size_A = M*K*sizeof(double);
    int size_B = N*K*sizeof(double);
    int size_C = N*M*sizeof(double);

    /* GPU: Allocate memory on device */
    printf("Allocating memory... \n");
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_b);
    cudaMalloc((void**)&d_C, size_c);

    /* GPU: Allocate memory on host */
    cudaMallocHost((void**)&h_A, size_A);
    cudaMallocHost((void**)&h_B, size_b);
    cudaMallocHost((void**)&h_C, size_c);

    /* Initialize matrices with random data (h_A. h_b)*/
    printf("Initializing matrices... \n");
    init_2d(max_val, M, K, h_A);
    init_2d(max_val, K, N, h_B);

    /* Copying data to device */
    printf("Copying data to device... \n");
    time_start = omp_get_wtime();
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);
    time_IO_1 = omp_get_wtime() - time_start;


    /* MATRIX MULTIPLICATION */
    printf("Computing Matrix multiplication... \n");
    time_start_compute = omp_get_wtime();
    switch(type) {
    case "nat":
        printf("Executing nat version...");
        break;
    case "lib":
        printf("Executing lib version...");
        break;
    case "gpu1":
        printf("Executing gpu1 version...");
        matmult_gpu1(M, N, d_A, d_B, d_C);
        break;
    case "gpu2":
        printf("Executing gpu2 version...");
        break;
    case "gpu3":
        printf("Executing gpu3 version...");
        break;
    case "gpu4":
        printf("Executing gpu4 version...");
        break;
    case "gpu5":
        printf("Executing gpu5 version...");
        break;
    case "gpulib":
        printf("Executing gpulib version...");
        break;
    default:
        printf("Error: Not valid type.");
        exit(1);
    }
    time_end_compute = omp_get_wtime();

    /* Copying back to host the result matrix C */
    if (type != "lib") { // if using gpu
        printf("Copying results back to host... \n");
        cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }
    time_end = omp_get_wtime();
    time_IO_2 = time_end - time_end_compute;
    total_time_compute = time_start_compute- time_end_compute;


    /* generate stats to print */
    printf("Generating results stats...\n");
    double giga = 1.0e-09;
    double gflops = (double)(N * M * 2 / total_time_compute) * giga; // 2 because product and sum
    double memory = giga*(double)size_A * (double)size_B  * (double)size_C * (double)pow(sizeof(double), -2); // each size already is multiplied by sizeof(double)
    double memoryGBsec = memory/ total_time_compute;

    printf("Total time (s) | Mem. footprint (GBytes) | GBytes/s | GFLOPS | Compute time (s) | I/O time (s) \n");
    printf("%g   %g   %g   %g   %g   %g \n", time_end-time_start, memory, memoryGBsec, gflops, total_time_compute, time_IO_2+time_IO_1)
    
    /* Freeing memory */
    printf("Freeing memory... \n");
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_c);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    printf("Success! \n");
    return(0);
}
