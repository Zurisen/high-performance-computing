#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d_gpu.h"
#include "func.h"
#include "print.h"

__global__ void jacobi_v3_dv0(double *d_u, double *d_uOld, double *d1_uOld, double *d_f, \
    double frac, double delta2, int N, int N2) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (0<k && k<(N-1) && 0<j && j<(N-1) && 0<i) {
    if (i == (N/2-1)) {
      d_u[i*N2 + j*N + k] = frac* ( \
        d_uOld[(i-1)*N2 + j*N + k] + \
        d1_uOld[j*N + k] + \
        d_uOld[i*N2 + (j-1)*N + k] + \
        d_uOld[i*N2 + (j+1)*N + k] + \
        d_uOld[i*N2 + j*N + k-1] + \
        d_uOld[i*N2 + j*N + k+1] + \
        delta2 * d_f[i*N2 + j*N + k]);
    } else if (i < (N/2-1)) {
      d_u[i*N2 + j*N + k] =frac* ( \
        d_uOld[(i-1)*N2+ j*N + k] + \
        d_uOld[(i+1)*N2 + j*N + k] + \
        d_uOld[i*N2 + (j-1)*N + k] + \
        d_uOld[i*N2 + (j+1)*N + k] + \
        d_uOld[i*N2 + j*N + k-1] + \
        d_uOld[i*N2 + j*N + k+1] + \
        delta2 * d_f[i*N2 + j*N + k]);
    }
  }
}

__global__ void jacobi_v3_dv1(double *d1_u, double *d1_uOld, double *d_uOld, double *d1_f, \
    double frac, double delta2, int N, int N2) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int k = blockIdx.z * blockDim.z + threadIdx.z;

  if (0<k && k<(N-1) && 0<j && j<(N-1) && i<((N/2)-1)) {
    if (i == 0) {
      d1_u[i*N2 + j*N + k] = frac* ( \
        d_uOld[(N/2-1)*N2 +j*N + k] + \
        d1_uOld[(i+1)*N2 +j*N + k] + \
        d1_uOld[i*N2 + (j-1)*N + k] + \
        d1_uOld[i*N2 + (j+1)*N + k] + \
        d1_uOld[i*N2 + j*N + k-1] + \
        d1_uOld[i*N2 + j*N + k+1] + \
        delta2 * d1_f[i*N2 + j*N + k]);
    }
    else if (i > 0) {
      d1_u[i*N2 + j*N + k] = frac* ( \
        d1_uOld[(i-1)*N2 + j*N + k] + \
        d1_uOld[(i+1)*N2 + j*N + k] + \
        d1_uOld[i*N2 + (j-1)*N + k] + \
        d1_uOld[i*N2 + (j+1)*N + k] + \
        d1_uOld[i*N2 + j*N + k-1] + \
        d1_uOld[i*N2 + j*N + k+1] + \
        delta2 * d1_f[i*N2 + j*N + k]);
    }
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
    cudaSetDevice(1);
    cudaMalloc((void**)&d_dummy,0);

    double *d_u, *d_uOld, *d_f, *d1_u, *d1_uOld, *d1_f;
    double *h_u, *h_uOld, *h_f;
    double size = N * N * N * sizeof(double);
    // Pinning memory in host
    cudaMallocHost((void**)&h_u, size);
    cudaMallocHost((void**)&h_uOld, size);
    cudaMallocHost((void**)&h_f, size);

    // Initialization of the arrays
    u_init(h_u, N, N2, start_T); 
    u_init(h_uOld, N, N2, start_T); 
    f_init(h_f, N, N2);

    // Device 0
    cudaSetDevice(0);

    // Device memory allocation 
    cudaMalloc((void**)&d_u, size/2);
    cudaMalloc((void**)&d_uOld, size/2);
    cudaMalloc((void**)&d_f, size/2);

    // Copy initializationf from host to device
    cudaMemcpy(d_u, h_u, size/2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uOld, h_uOld, size/2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, h_f, size/2, cudaMemcpyHostToDevice);

    // Device 1
    cudaSetDevice(1);

    // Device memory allocation 
    cudaMalloc((void**)&d1_u, size/2);
    cudaMalloc((void**)&d1_uOld, size/2);
    cudaMalloc((void**)&d1_f, size/2);

    // Copy initializationf from host to device
    cudaMemcpy(d1_u, h_u + N*N*N/2, size/2, cudaMemcpyHostToDevice);
    cudaMemcpy(d1_uOld, h_uOld + N*N*N/2, size/2, cudaMemcpyHostToDevice);
    cudaMemcpy(d1_f, h_f + N*N*N/2, size/2, cudaMemcpyHostToDevice);
   
    // Enable peer access
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1,0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0,0);

    // kernel settings
    dim3 blocksize(8,8,8);
    dim3 gridsize( ceil((int) N/blocksize.x),ceil((int) N/blocksize.y),ceil((int) N/blocksize.z) ); 
    // Jacobi max iterations loop in host
    double frac = 1.0/6.0;
    double delta2 = (2.0*2.0)/N2;
    
    int it = 0;
    while(it < iter_max){
        swap(d_uOld, d_u);
        swap(d1_uOld, d1_u);    

        cudaSetDevice(0);
        jacobi_v3_dv0<<<gridsize,blocksize>>>(d_u, d_uOld, d1_uOld, d_f, frac, delta2, N, N2);

        cudaSetDevice(1);
        jacobi_v3_dv1<<<gridsize,blocksize>>>(d1_u, d1_uOld, d_uOld, d1_f, frac, delta2, N, N2);
        
        cudaDeviceSynchronize();
        cudaSetDevice(0);
        cudaDeviceSynchronize();
        it++;
    }

    // Copy back to host
    cudaSetDevice(0);
    cudaMemcpy(h_u, d_u, size/2, cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(h_u + N*N*N/2, d1_u, size/2, cudaMemcpyDeviceToHost);

    // dump  results if wanted
    switch(output_type) {
        case 0:
            // no output at all
            break;
        case 4:
            output_ext = ".vtk";
            sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
            //fprintf(stderr, "Write VTK file to %s: ", output_filename);
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
    
    cudaFreeHost(d_f);
    cudaFreeHost(d_u);
    cudaFreeHost(d_uOld);
   
    return(0); 
}
