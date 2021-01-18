#include <omp.h>
#include <stdio.h>
#include <helper_cuda.h>

const int device = 0;

// kernel function
__global__ void my_kernel(){

int thread_i = threadIdx.x;
int thread_max = blockDim.x;
int block_i = blockIdx.x;

int glo_thread_i = blockDim.x * blockIdx.x + threadIdx.x;
int glo_thread_i_max = gridDim.x * blockDim.x;

// force segmentation fault in block
//if (glo_thread_i == 100){
//    int *a = (int*) 0x10000; *a = 0;
//}
//
    printf("Hello world! I'm thread %i out of %i in block %i. My global thread id is %i out of %i.\n", thread_i,thread_max,block_i,glo_thread_i,glo_thread_i_max);
}

int main(int argc, char* argv[]){

// wake up GPU

//printf("Warming up device %i ...", device);

double time = omp_get_wtime();
cudaSetDevice(device);

double* dummy_d;
cudaMalloc((void**)&dummy_d,0); //Force allocation for waking up gpu
//printf("Wake up time = %3.2f seconds\n", omp_get_wtime()-time);

int n_blk, n_threads;

if (argc == 3 ) {
    n_blk = atoi(argv[1]);
    n_threads = atoi(argv[2]);
    }
else {
    // use default N
    n_blk = 8;
    n_threads = 16;
    }

//printf("n_blk  %i ; n_threads %i\n",n_blk, n_threads);

my_kernel<<<n_blk,n_threads>>>();
cudaDeviceSynchronize();
}




