// Asignment 3 Part 1 Matrix Multiplication

extern "C" {

#include <stdio.h>
#include <stdlib.h>
#include <math.h> // for pow()
#include <omp.h>
#include <cblas.h> // for matmult_lib()

/* Native CBLAS CPU implementation of matrix multiplication */
void matmult_lib(int M, int N, int K, double *A, double *B, double *C) {
        double alpha = 1.0, beta = 0.0;
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,alpha,A,K,B,N,beta,C, N);
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
    
    double *d_A, *d_B, *d_C; // Device variables
    double *h_A, *h_B, *h_C; // host vairables
    int size_A = M*K*sizeof(double);
    int size_B = N*K*sizeof(double);
    int size_C = N*M*sizeof(double);

    /* GPU: Allocate memory on device */
    printf("Allocating memory... \n");
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    /* GPU: Allocate memory on host */
    cudaMallocHost((void**)&h_A, size_A);
    cudaMallocHost((void**)&h_B, size_B);
    cudaMallocHost((void**)&h_C, size_C);

    /* Initialize matrices with random data (h_A. h_B)*/
    printf("Initializing matrices... \n");
    init_2d(max_val, M, K, h_A);
    init_2d(max_val, K, N, h_B);

    /* Copying data to device */
    printf("Copying data to device... \n");
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    /* MATRIX MULTIPLICATION */
    printf("Computing Matrix multiplication... \n");
    // Define grid and threads per block
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    matmult_gpu1_kernel<<<1,1>>>(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    printf("Copying results back to host... \n");
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Freeing memory */
    printf("Freeing memory... \n");
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
    double *d_A, *d_B, *d_C; // Device variables
    double *h_A, *h_B, *h_C; // host vairables
    int size_A = M*K*sizeof(double);
    int size_B = N*K*sizeof(double);
    int size_C = N*M*sizeof(double);

    /* GPU: Allocate memory on device */
    printf("Allocating memory... \n");
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    /* GPU: Allocate memory on host */
    cudaMallocHost((void**)&h_A, size_A);
    cudaMallocHost((void**)&h_B, size_B);
    cudaMallocHost((void**)&h_C, size_C);

    /* Initialize matrices with random data (h_A. h_B)*/
    printf("Initializing matrices... \n");
    init_2d(max_val, M, K, h_A);
    init_2d(max_val, K, N, h_B);

    /* Copying data to device */
    printf("Copying data to device... \n");
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    /* MATRIX MULTIPLICATION */
    printf("Computing Matrix multiplication... \n");
    // Define grid and threads per block
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    int BLOCK_SIZE = 16;
    dim3 blocksPerGrid(((M+BLOCK_SIZE-1) / BLOCK_SIZE), ((N+BLOCK_SIZE-1) / BLOCK_SIZE));
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    matmult_gpu2_kernel<<<blocksPerGrid,threadsPerBlock>>>(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    printf("Copying results back to host... \n");
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Freeing memory */
    printf("Freeing memory... \n");
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    /* MATRIX MULTIPLICATION */
    printf("Computing Matrix multiplication... \n");
    // Define grid and threads per block
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    int BLOCK_SIZE = 16;
    int stride = 2;
    dim3 blocksPerGrid(((M+BLOCK_SIZE-1) / (stride * BLOCK_SIZE)), ((N+BLOCK_SIZE-1) / BLOCK_SIZE));
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    matmult_gpu3_kernel<<<blocksPerGrid,threadsPerBlock>>>(M, N, K, d_A, d_B, d_C, stride);
    cudaDeviceSynchronize();

    printf("Copying results back to host... \n");
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Freeing memory */
    printf("Freeing memory... \n");
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_c);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    /* MATRIX MULTIPLICATION */
    printf("Computing Matrix multiplication... \n");
    // Define grid and threads per block
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    int BLOCK_SIZE = 16;
    int stride_row = 2;
    int stride_col = 2;
    dim3 blocksPerGrid(((M+BLOCK_SIZE-1) / (stride_row * BLOCK_SIZE)), ((N+BLOCK_SIZE-1) / (stride_col * BLOCK_SIZE)));
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    matmult_gpu3_kernel<<<blocksPerGrid,threadsPerBlock>>>(M, N, K, d_A, d_B, d_C, stride_row, stride_col);
    cudaDeviceSynchronize();

    printf("Copying results back to host... \n");
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Freeing memory */
    printf("Freeing memory... \n");
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_c);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
/* part 5: GPU (shared memory version) */
__global__ void matmult_gpu5(int M, int N, int K, double** A, double **B, double** C) {
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
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    /* MATRIX MULTIPLICATION */
    printf("Computing Matrix multiplication... \n");
    // Define grid and threads per block
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block


    //TODO:

    printf("Copying results back to host... \n");
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Freeing memory */
    printf("Freeing memory... \n");
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_c);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/* part 6: DGEMM function for GPUs, NVIDIA */
__global__ void matmult_gpulib(int M, int N, int K, double **A, double **B, double **C) {
        double alpha = 1.0, beta = 0.0;
        cublasDgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,alpha,A[0],K,B[0],N,beta,C[0],N);
}

/* Initialization of 2d arrays */
void init_2d (double max_val, int m, int n, double* A) {
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            A[i][j] = (double)rand()*max_val/RAND_MAX;
        }
    }
}
}
