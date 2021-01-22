#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d_gpu.h"
#include "func.h"
#include "print.h"
#include <omp.h>

__managed__ double sh_norm=1000000.0;

__global__ void jacobi_v1(double *d_u, double *d_uOld, double *d_f, int N, int N2, int iter_max, double frac, double delta2){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    double norm_value;
    if (i>0 && i<N-1 && j>0 && j<N-1 && k>0 && k<N-1){
       	d_u[i*N2+j*N+k]	= frac*(d_uOld[(i-1)*N2+j*N+k] + d_uOld[(i+1)*N2+j*N+k]+d_uOld[i*N2+(j-1)*N+k] + d_uOld[i*N2+(j+1)*N+k]+d_uOld[i*N2+j*N+k-1] + d_uOld[i*N2+j*N+k+1]+delta2*d_f[i*N2+j*N+k]);
        
        norm_value = ((d_u[i*N2+j*N+k]-d_uOld[i*N2+j*N+k])*(d_u[i*N2+j*N+k]-d_uOld[i*N2+j*N+k]));
        atomicAdd(&sh_norm, norm_value);
    }
}

int main(int argc, char *argv[]){
    
    int N = atoi(argv[1]);
    int iter_max = atoi(argv[2]);
    double start_T = atof(argv[3]);
    double tolerance = atof(argv[4]);
    int output_type = 4;
    char *output_prefix = "poisson_j_gpu1";
    char *output_ext = "";
    char output_filename[FILENAME_MAX];


    int N2 = N * N;
    // Wake up gpu
    cudaSetDevice(0);
    double *d_dummy;
    cudaMalloc((void**)&d_dummy,0);

    double *d_u, *d_uOld, *d_f;
    double *h_u, *h_uOld, *h_f;
    int size = N * N * N * sizeof(double);

    // Device memory allocation 
    cudaMalloc((void**)&d_u, size);
    cudaMalloc((void**)&d_uOld, size);
    cudaMalloc((void**)&d_f, size);
    // Pinning memory in host
    cudaMallocHost((void**)&h_u, size);
    cudaMallocHost((void**)&h_uOld, size);
    cudaMallocHost((void**)&h_f, size);

    // Initialization of the arrays
    u_init(h_u, N, N2, start_T); 
    u_init(h_uOld, N, N2, start_T); 
    f_init(h_f, N, N2);

    // Copy initializationf from host to device
    cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_uOld, h_uOld, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, h_f, size, cudaMemcpyHostToDevice);

    // kernel settings
    dim3 blocksize(8,8,8);
    dim3 gridsize( ceil((int) N/blocksize.x),ceil((int) N/blocksize.y),ceil((int) N/blocksize.z) );
   
    // dim3 gridsize(1,1,1);
    // dim3 blocksize(64,64,64); 
    // Jacobi max iterations loop in host
    double frac = 1.0/6.0;
    double delta2 = (2.0*2.0)/N2;
    
    int it = 0;
    double ts = omp_get_wtime();
    while(it < iter_max && sh_norm>tolerance){
        sh_norm = 0;   
        
        swap(&d_uOld, &d_u); 
        jacobi_v1<<<gridsize,blocksize>>>(d_u, d_uOld, d_f, N, N2, iter_max, frac, delta2);
        cudaDeviceSynchronize();
        it++;
    }
    double te = omp_get_wtime() - ts;
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
            //fprintf(stderr, "Write VTK file to %s: ", output_filename);
            print_vtk(output_filename, N, h_u);
            break;
        default:
            fprintf(stderr, "Non-supported output type!\n");
            break;
    }

    // Calculate effective bandwidth
    double efBW = N*N*N*sizeof(double)*4*it/te/1e3;
       // 4 -> read uold, f | read and write u
    // Calculate it/s
    double itpersec  = it/te;
    int kbytes = N*N*N*sizeof(double)*3/1000;
    //print info
    printf("%d %d %3.6f %3.6f %3.6f %3.6f\n", N, it, te, itpersec, kbytes, efBW);

    //Free host and device memory    
    cudaFreeHost(h_f);
    cudaFreeHost(h_u);
    cudaFreeHost(h_uOld);
    
    cudaFree(d_f);
    cudaFree(d_u);
    cudaFree(d_uOld);
   
    return(0); 
}
