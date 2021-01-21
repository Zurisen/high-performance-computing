// Asignment 3 Part 1 Matrix Multiplication
#include <cublas_v2.h> // for matmult_gpulib()
extern "C" {

    #include <stdio.h>
    #include <stdlib.h>
    #include <math.h> // for pow()
    #include <omp.h>
    #include <cblas.h> // for matmult_lib()

    #define stride_col 2
    #define stride_row 2
    #define stride 2
    #define BLOCK_SIZE 16

    /* Native CBLAS CPU implementation of matrix multiplication */
    void matmult_lib(int M, int N, int K, double *A, double *B, double *C) {
        double alpha = 1.0, beta = 0.0;
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,alpha,A,K,B,N,beta,C, N);
    }

    /* part 1: sequential implementation in GPU (single thread) */
    __global__ void matmult_gpu1_kernel(int M, int N, int K, double* A, double *B, double* C) {
        double temp = 0.0;

        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                temp = 0.0;
                for (int k = 0; k < K; k++) {
                    temp += A[i*K + k] * B[k*N + j];
                }
                C[i*N + j] = temp;
            }
        }
    }

    void matmult_gpu1(int M, int N, int K, double* h_A, double *h_B, double* h_C) {
        double *d_A, *d_B, *d_C; // Device variables
        int size_A = M*K*sizeof(double);
        int size_B = N*K*sizeof(double);
        int size_C = N*M*sizeof(double);

        /* GPU: Allocate memory on device */
        cudaMalloc((void**)&d_A, size_A);
        cudaMalloc((void**)&d_B, size_B);
        cudaMalloc((void**)&d_C, size_C);

        /* Copying data to device */
        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

        /* MATRIX MULTIPLICATION */
        // Define grid and threads per block
        matmult_gpu1_kernel<<<1,1>>>(M, N, K, d_A, d_B, d_C);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

        /* Freeing memory */
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    /* part 2: naive implementation in GPU (one thread per element in C) */
    __global__ void matmult_gpu2_kernel(int M, int N, int K, double* A, double* B, double* C) {
        double temp = 0.0;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i < M && j < N){ // ensure that the extra threads do not do any work
            for (int step = 0; step < K; step++) {
                temp += A[j*K + step] * B[step*N + i];
            }
            C[j*N + i] = temp;
        }
    }

    void matmult_gpu2(int M, int N, int K, double* h_A, double *h_B, double* h_C) {
        double *d_A, *d_B, *d_C; // Device variables
        int size_A = M*K*sizeof(double);
        int size_B = N*K*sizeof(double);
        int size_C = N*M*sizeof(double);

        /* GPU: Allocate memory on device */
        cudaMalloc((void**)&d_A, size_A);
        cudaMalloc((void**)&d_B, size_B);
        cudaMalloc((void**)&d_C, size_C);

        /* Copying data to device */
        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

        /* MATRIX MULTIPLICATION */
        // Define grid and threads per block
        dim3 blocksPerGrid(((N-1) / BLOCK_SIZE+1), ((M-1) / BLOCK_SIZE+1));
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

        matmult_gpu2_kernel<<<blocksPerGrid,threadsPerBlock>>>(M, N, K, d_A, d_B, d_C);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

        /* Freeing memory */
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    /* part 3: GPU (thread computes 2 elements of C) */
    __global__ void matmult_gpu3_kernel(int M, int N, int K, double* d_A, double* d_B, double* d_C) {
        double temp1 = 0.0;
        double temp2 = 0.0;

        int i = (blockIdx.x * blockDim.x + threadIdx.x);
        int j = (blockIdx.y * blockDim.y + threadIdx.y)*stride;

        if (i < M && j < N) {
            for (int k = 0; k < K; k++) {
                temp1 += d_A[(i)*K + k] * d_B[k*N + j];
                if (j+1 < N) { // only if not end
                    temp2 += d_A[(i)*K + k] * d_B[k*N + (j+1)]; // right neighbour
                }
            }
            d_C[i*N + j] = temp1;
            if (j+1 < N) { // only if not end
                d_C[(i)*N + (j+1)] = temp2;
            }
        }
    }

    void matmult_gpu3(int M, int N, int K, double* h_A, double *h_B, double* h_C) {
        double *d_A, *d_B, *d_C; // Device variables
        int size_A = M*K*sizeof(double);
        int size_B = N*K*sizeof(double);
        int size_C = N*M*sizeof(double);

        /* GPU: Allocate memory on device */
        cudaMalloc((void**)&d_A, size_A);
        cudaMalloc((void**)&d_B, size_B);
        cudaMalloc((void**)&d_C, size_C);

        /* Copying data to device */
        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

        /* MATRIX MULTIPLICATION */
        // Define grid and threads per block
        dim3 blocksPerGrid(ceil(N/BLOCK_SIZE)+1, ceil(M/BLOCK_SIZE*stride)+1);
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

        matmult_gpu3_kernel<<<blocksPerGrid,threadsPerBlock>>>(M, N, K, d_A, d_B, d_C);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

        /* Freeing memory */
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    /* part 4: GPU (thread computes >2 elements of C) */
    __global__ void matmult_gpu4_kernel(int M, int N, int K, double* d_A, double* d_B, double* d_C) {
        double temp[stride_col][stride_row];
        int sc, sr;
        for (sc = 0; sc < stride_col; sc++){
            for (sr = 0; sr < stride_row; sr++){
                temp[sc][sr] = 0.0;
            }
        }

        int j = (blockIdx.x * blockDim.x + threadIdx.x)*stride_col;
        int i = (blockIdx.y * blockDim.y + threadIdx.y)*stride_row;

        if (i < M && j < N) {
            for (int k = 0; k < K; k++) {
                for (sc = 0; sc < stride_col; sc++) {
                    if (sc + j < N) {
                        for (sr = 0; sr < stride_row; sr++) {
                            if (sr + i < M) {
                                temp[sc][sr] += d_A[(i+sr)*K + k] * d_B[k*N + (j+sc)];
                            }
                        }
                    }
                }
            }
            for (sc = 0; sc < stride_col; sc++) {
                if (sc + j < N) {
                    for (sr = 0; sr < stride_row; sr++) {
                        if (sr + i < M) {
                            d_C[(i+sr)*N + (j + sc)] = temp[sc][sr];
                        }
                    }
                }
            }
        }
    }

    void matmult_gpu4(int M, int N, int K, double* h_A, double *h_B, double* h_C) {
        double *d_A, *d_B, *d_C; // Device variables
        int size_A = M*K*sizeof(double);
        int size_B = N*K*sizeof(double);
        int size_C = N*M*sizeof(double);

        /* GPU: Allocate memory on device */
        cudaMalloc((void**)&d_A, size_A);
        cudaMalloc((void**)&d_B, size_B);
        cudaMalloc((void**)&d_C, size_C);

        /* Copying data to device */
        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

        /* MATRIX MULTIPLICATION */
        // Define grid and threads per block
        dim3 blocksPerGrid(ceil(N/BLOCK_SIZE*stride_col)+1, ceil(M/BLOCK_SIZE*stride_row)+1);
        dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

        matmult_gpu4_kernel<<<blocksPerGrid,threadsPerBlock>>>(M, N, K, d_A, d_B, d_C);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        /* Freeing memory */
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }

    /* part 5: GPU (shared memory version) */

    __global__ void matmult_gpu5_kernel(int M, int N, int K, double* A, double* B, double* C) {
        // Each thread computes one element of C
        // by accumulating results into Cvalue
        double temp = 0.0;
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        int j = blockIdx.y * blockDim.y + threadIdx.y;

        if (i < M && j < N){ // ensure that the extra threads do not do any work
            for (int step = 0; step < K; step++) {
                temp += A[i*K + step] * B[step*N + j];
            }
            C[i*N + j] = temp;
        }
    }

    // Matrix multiplication - Host code
    // Matrix dimensions are assumed to be multiples of BLOCK_SIZE
    void matmult_gpu5(int M, int N, int K, double* h_A, double* h_B, double* h_C) {
        double *d_A, *d_B, *d_C; // Device variables
        int size_A = M*K*sizeof(double);
        int size_B = N*K*sizeof(double);
        int size_C = N*M*sizeof(double);

        /* GPU: Allocate memory on device */
        cudaMalloc((void**)&d_A, size_A);
        cudaMalloc((void**)&d_B, size_B);
        cudaMalloc((void**)&d_C, size_C);

        /* Copying data to device */
        cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

        // Invoke kernel
        dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
        dim3 dimGrid(N / dimBlock.x, M / dimBlock.y);
        matmult_gpu5_kernel<<<dimGrid, dimBlock>>>(M, N, K, d_A, d_B, d_C);
        cudaDeviceSynchronize();

        cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        /* Freeing memory */
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }


    /* part 6: DGEMM function for GPUs, NVIDIA */
    void matmult_gpulib(int M, int N, int K, double* A, double *B, double* C) {

        /* Declare handle and initialize cublas */
        cublasHandle_t handle;
        cublasStatus_t status = cublasCreate(&handle);
        if (status != CUBLAS_STATUS_SUCCESS) { // check if init successful
            printf("Error: Initialization error CUBLAS. \n");
            exit(1);
        }

        double alpha = 1.0; // no prefactor
        double beta = 0.0; // C matrix not involved

    
        status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, N, &beta, C, K);
        if (status != CUBLAS_STATUS_SUCCESS) { // check no errors are outputed in the execution
            printf("Error: Execution error CUBLAS. \n");
            exit(1);
        }

        /* Destroy handle and free memory */
        status = cublasDestroy(handle);
        if (status != CUBLAS_STATUS_SUCCESS) {
            printf("Error: Error destroying CUBLAS handle. \n");
            exit(1);
        }
    }
}
