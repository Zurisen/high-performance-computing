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

void matmult_gpu1(int M, int N, int K, double* A, double *B, double* C) {
    
    double *d_A, *d_B, *d_C; // Device variables
    double *h_A, *h_B, *h_C; // host vairables
    int size_A = M*K*sizeof(double);
    int size_B = N*K*sizeof(double);
    int size_C = N*M*sizeof(double);
    double max_val = 10.0;

    /* GPU: Allocate memory on device */
    //printf("Allocating memory... \n");
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    /* GPU: Allocate memory on host */
    cudaMallocHost((void**)&h_A, size_A);
    cudaMallocHost((void**)&h_B, size_B);
    cudaMallocHost((void**)&h_C, size_C);

    /* Copying data to device */
    //printf("Copying data to device... \n");
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    /* MATRIX MULTIPLICATION */
    //printf("Computing Matrix multiplication... \n");
    // Define grid and threads per block
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    matmult_gpu1_kernel<<<1,1>>>(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    //printf("Copying results back to host... \n");
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Freeing memory */
    //printf("Freeing memory... \n");
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
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
            temp += A[i*K + step] * B[step*N + j];
        }
        C[i*N + j] = temp;
    }
}

void matmult_gpu2(int M, int N, int K, double* h_A, double* h_B, double* h_C) {
    double *d_A, *d_B, *d_C; // Device variables
    int size_A = M*K*sizeof(double);
    int size_B = N*K*sizeof(double);
    int size_C = N*M*sizeof(double);
    double max_val = 10.0;

    /* GPU: Allocate memory on device */
    //printf("Allocating memory... \n");
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    /* GPU: Allocate memory on host */
    cudaMallocHost((void**)&h_A, size_A);
    cudaMallocHost((void**)&h_B, size_B);
    cudaMallocHost((void**)&h_C, size_C);

    /* Copying data to device */
    //printf("Copying data to device... \n");
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    /* MATRIX MULTIPLICATION */
    //printf("Computing Matrix multiplication... \n");
    // Define grid and threads per block
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    int BLOCK_SIZE = 16;
    dim3 blocksPerGrid(((M+BLOCK_SIZE-1) / BLOCK_SIZE), ((N+BLOCK_SIZE-1) / BLOCK_SIZE));
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    matmult_gpu2_kernel<<<blocksPerGrid,threadsPerBlock>>>(M, N, K, d_A, d_B, d_C);
    cudaDeviceSynchronize();

    //printf("Copying results back to host... \n");
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Freeing memory */
    //printf("Freeing memory... \n");
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/* part 3: GPU (thread computes 2 elements of C) */
__global__ void matmult_gpu3_kernel(int M, int N, int K, double* A, double* B, double* C, int stride) {
    double temp = 0.0;
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

void matmult_gpu3(int M, int N, int K, double* h_A, double* h_B, double* h_C) {
    double *d_A, *d_B, *d_C; // Device variables
    int size_A = M*K*sizeof(double);
    int size_B = N*K*sizeof(double);
    int size_C = N*M*sizeof(double);
    double max_val = 10.0;

    /* GPU: Allocate memory on device */
    //printf("Allocating memory... \n");
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    /* GPU: Allocate memory on host */
    cudaMallocHost((void**)&h_A, size_A);
    cudaMallocHost((void**)&h_B, size_B);
    cudaMallocHost((void**)&h_C, size_C);

    /* Copying data to device */
    //printf("Copying data to device... \n");
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    /* MATRIX MULTIPLICATION */
    //printf("Computing Matrix multiplication... \n");
    // Define grid and threads per block
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    int BLOCK_SIZE = 16;
    int stride = 2;
    dim3 blocksPerGrid(((M+BLOCK_SIZE-1) / (stride * BLOCK_SIZE)), ((N+BLOCK_SIZE-1) / BLOCK_SIZE));
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    matmult_gpu3_kernel<<<blocksPerGrid,threadsPerBlock>>>(M, N, K, d_A, d_B, d_C, stride);
    cudaDeviceSynchronize();

    //printf("Copying results back to host... \n");
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Freeing memory */
    //printf("Freeing memory... \n");
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

/* part 4: GPU (thread computes >2 elements of C) */
__global__ void matmult_gpu4_kernel(int M, int N, int K, double* A, double* B, double* C, int stride_row, int stride_col) {
    double temp = 0.0;
    int i = (blockIdx.x * blockDim.x + threadIdx.x)*stride_row;
    int j = (blockIdx.y * blockDim.y + threadIdx.y)*stride_col;

    for (int s_row = 0; s_row < stride_row; s_row++) {
        for (int s_col = 0; s_col < stride_col; s_col++) {
            if ((s_row + i) < M && (j + s_col) < N){ // ensure that the extra threads do not do any work
                for (int step = 0; step < K; step++) {
                    temp += A[(s_row + i)*K + step] * B[step*N + (j + s_col)];
                }
                C[(s_row + i)*N + (j + s_col)] = temp;
            }
        }
    }
}

void matmult_gpu4(int M, int N, int K, double* h_A, double* h_B, double* h_C) {
    double *d_A, *d_B, *d_C; // Device variables
    int size_A = M*K*sizeof(double);
    int size_B = N*K*sizeof(double);
    int size_C = N*M*sizeof(double);
    double max_val = 10.0;

    /* GPU: Allocate memory on device */
    //printf("Allocating memory... \n");
    cudaMalloc((void**)&d_A, size_A);
    cudaMalloc((void**)&d_B, size_B);
    cudaMalloc((void**)&d_C, size_C);

    /* GPU: Allocate memory on host */
    cudaMallocHost((void**)&h_A, size_A);
    cudaMallocHost((void**)&h_B, size_B);
    cudaMallocHost((void**)&h_C, size_C);

    /* Copying data to device */
    //printf("Copying data to device... \n");
    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    /* MATRIX MULTIPLICATION */
    //printf("Computing Matrix multiplication... \n");
    // Define grid and threads per block
    // declare the number of blocks per grid and the number of threads per block
    // use 1 to 512 threads per block
    int BLOCK_SIZE = 16;
    int stride_row = 2;
    int stride_col = 2;
    dim3 blocksPerGrid(((M+BLOCK_SIZE-1) / (stride_row * BLOCK_SIZE)), ((N+BLOCK_SIZE-1) / (stride_col * BLOCK_SIZE)));
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    matmult_gpu4_kernel<<<blocksPerGrid,threadsPerBlock>>>(M, N, K, d_A, d_B, d_C, stride_row, stride_col);
    cudaDeviceSynchronize();

    //printf("Copying results back to host... \n");
    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    /* Freeing memory */
    //printf("Freeing memory... \n");
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
/* part 5: GPU (shared memory version) */
// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 16

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaMalloc(&d_A.elements, size);
    cudaMemcpy(d_A.elements, A.elements, size,
               cudaMemcpyHostToDevice);
    Matrix d_B;
    d_B.width = B.width; d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    cudaMalloc(&d_B.elements, size);
    cudaMemcpy(d_B.elements, B.elements, size,
               cudaMemcpyHostToDevice);

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    cudaMalloc(&d_C.elements, size);

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);

    // Read C from device memory
    cudaMemcpy(C.elements, d_C.elements, size,
               cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
__global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}
/* part 6: DGEMM function for GPUs, NVIDIA */

}
