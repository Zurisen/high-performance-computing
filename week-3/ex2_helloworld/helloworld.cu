#include <omp.h>
#include <stdio.h>

const int device = 0;

// kernel function
__global__ void my_kernel(){

int thread_i = threadIdx.x;
int thread_max = blockDim.x;
int block_i = blockIdx.x;

int glo_thread_i = blockDim.x * blockIdx.x + threadIdx.x;
int glo_thread_i_max = gridDim.x * blockDim.x;

    printf("Hello world! I'm thread %i out of %i in block %i. My global thread id is %i out of %i.\n", thread_i,thread_max,block_i,glo_thread_i,glo_thread_i_max);
}


int main(int argc, char *argv[]) {
    // Wake up GPU from power save state.
    printf("Warming up device %i ... ", device); fflush(stdout);
    double time = omp_get_wtime();
    cudaSetDevice(device);           // Set the device to 0 or 1.
    double *dummy_d;
    cudaMalloc((void**)&dummy_d, 0); // We force the creation of context on the
    // device by allocating a dummy variable.
        printf("time = %3.2f seconds\n", omp_get_wtime() - time);

    // program 
    int n_blk, n_threads;

    if (argc == 3 ) {
        n_blk = atoi(argv[1]);
        n_threads = atoi(argv[2]);
    }
    else {
        // use default N
        n_blk = 1;
        n_threads = 32;
    }

    printf("n_blk  %i ; n_threads %i\n",n_blk, n_threads);

    my_kernel<<<n_blk,n_threads>>>();
    cudaDeviceSynchronize();
}
