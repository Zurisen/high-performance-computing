#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "mandelgpu.h"
#include "writepng.h"

// setting GPU device
const int device = 0;

int main(int argc, char *argv[]) {

    int width, height, max_iter, *h_image_gpu, *d_image_gpu;

    width    = 10000;
    height   = 10000;
    max_iter = 400;

    // warm up:
    cudaSetDevice(device);
    double *dummy_d;
    cudaMalloc((void**)&dummy_d, 0);
    // command line argument sets the dimensions of the image
    if ( argc == 2 ) width = height = atoi(argv[1]);


    // GPU
    // Alloc mem on host and device
    cudaMallocHost((void **)&h_image_gpu, width * height * sizeof(int));
    cudaMalloc((void **)&d_image_gpu, width * height * sizeof(int));

    //   
    int bs = 16;
    dim3 block_size(bs, bs);
    dim3 grid_size(width / block_size.x, height / block_size.y);
    
    double time = omp_get_wtime();
    mandel_gpu<<<grid_size, block_size>>>(width, height, d_image_gpu, max_iter, bs);
    //mandel_gpu<<<1,1>>>(width, height, d_image_gpu, max_iter, 1);
    cudaDeviceSynchronize();
    double time_compute = omp_get_wtime();

    // Copy result back to host
    cudaMemcpy(h_image_gpu, d_image_gpu, width * height * sizeof(int), cudaMemcpyDeviceToHost);
    double time_IO = omp_get_wtime();

    printf("Compute = %3.2f seconds\n", time_compute - time);
    printf("IO =      %3.2f seconds\n", time_IO - time_compute);
    printf("Total =   %3.2f seconds\n", time_IO - time);

    writepng("mandelbrotgpu.png", h_image_gpu, width, height);

    // Cleanup
    cudaFreeHost(h_image_gpu); 
    cudaFree(d_image_gpu);

    return(0);
}


