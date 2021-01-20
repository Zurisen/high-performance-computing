#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d_gpu.h"
#include "func.h"
#include "print.h"

__global__ void jacobi_v1(double *d_u, double *d_uOld, double *d_f, int N, int N2, int iter_max, double frac, double delta2){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i>0 && i<N-1 && j>0 && j<N-1 && k>0 && k<N-1){
       	d_u[i*N2+j*N+k]	= frac*(d_uOld[(i-1)*N2+j*N+k]+d_uOld[(i+1)*N2+j*N+k]+d_uOld[i*N2+(j-1)*N+k]+d_uOld[i*N2+(j+1)*N+k]+d_uOld[i*N2+j*N+k-1]+d_uOld[i*N2+j*N+k+1]+delta2*d_f[i*N2+j*N+k]);
    }
}

int main(int argc, char *argv[]){
    
    int N = atoi(argv[1]);
    int iter_max = atoi(argv[2]);
    double start_T = atof(argv[3]);
    int output_type = 4;
    char *output_prefix = "poisson_j_gpu1";
    char *output_ext = "";
    char output_filename[FILENAME_MAX];


    int N2 = N * N;
    // Wake up gpu
    cudaSetDevice(0);
    double *d_dummy;
    cudaMalloc((void**)&d_dummy,0);

    double *d_u, *d_uOld, *d_uSwap, *d_f;
    double *h_u, *h_uOld, *h_uSwap, *h_f;
    double size = N * N * N * sizeof(double);

    // Device memory allocation 
    cudaMalloc((void**)&d_u, size);
    cudaMalloc((void**)&d_uOld, size);
    cudaMalloc((void**)&d_uSwap, size);
    cudaMalloc((void**)&d_f, size);

    // Pinning memory in host
    cudaMallocHost((void**)&h_u, size);
    cudaMallocHost((void**)&h_uOld, size);
    cudaMallocHost((void**)&h_uSwap, size);
    cudaMallocHost((void**)&h_f, size);

    // Initialization of the arrays
    u_init(h_u, N, N2, start_T); 
    u_init(h_uOld, N, N2, start_T); 
    u_init(h_uSwap, N, N2, start_T); 
    f_init(h_f, N, N2);

    // Copy initializationf from host to device
    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uOld, h_uOld, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uSwap, h_uSwap, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, h_f, size, cudaMemcpyHostToDevice);

    // kernel settings
    dim3 blocksize(10,10,10);
    dim3 gridsize( ceil((double) N/blocksize.x),ceil((double) N/blocksize.y),ceil((double) N/blocksize.z) );
    // Jacobi max iterations loop in host
    double frac = 1.0/6.0;
    double delta2 = (2.0*2.0)/N2;
        // timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    
    int it = 0;
    cudaEventRecord(start);
    while(it < iter_max){
        d_uSwap = d_uOld;
        d_u = d_uOld;
        d_uOld = d_uSwap;   
        jacobi_v1<<<gridsize,blocksize>>>(d_u, d_uOld, d_f, N, N2, iter_max, frac, delta2);
        cudaDeviceSynchronize();
        
        it++;
    }
    cudaEventRecord(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Operation finished!  GPU runtime (ms): %3.6f\n\n", milliseconds);

    // Copy back to host
    cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost);

    // dump  results if wanted
    switch(output_type) {
        case 0:
            // no output at all
            break;
        case 4:
            output_ext = ".vtk";
            sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
            fprintf(stderr, "Write VTK file to %s: ", output_filename);
            print_vtk(output_filename, N, h_u);
            break;
        default:
            fprintf(stderr, "Non-supported output type!\n");
            break;
    }

    //Free host and device memory    
    cudaFreeHost(h_f);
    cudaFreeHost(h_u);
    cudaFreeHost(h_uOld);
    cudaFreeHost(h_uSwap);
    
    cudaFreeHost(d_f);
    cudaFreeHost(d_u);
    cudaFreeHost(d_uOld);
    cudaFreeHost(d_uSwap);
   
    return(0); 
}
