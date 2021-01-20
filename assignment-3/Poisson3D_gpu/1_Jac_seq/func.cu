#include <math.h>

void u_init(double *u, int N, int N2, double start_T) {
    int i, j, k;
    // Initialize interior
    for (i=0; i<N; i++){
        for (j=0; j<N; j++) {
            for (k=0; k<N; k++){
                u[i*N2+j*N+k] = start_T;
            }
        }
            
    }

    // Initialize boundaries
    for (i=0; i<N; i++){
        for (j=0; j<N; j++){
            u[i*N2+(N-1)*N+j] = 20.0;
            u[i*N2+j] = 0.0;
            u[i*N+j] = 20.0;
            u[N2*(N-1)+i*N+j] = 20.0;
            u[i*N2+j*N+N-1] = 20.0;
            u[i*N2+j*N] = 20.0;
        }
    
    }

}

void f_init(double *f, int N, int N2) {
    int i_max = (int) 5*N/16;
    int j_max = (int) N/4;
    int k_min = (int) N/6;
    
    int i, j, k;
    for(i=0; i<N; i++){
        for(j=0; j<N; j++){
            for(k=0;k<N;k++){
                if(i <= i_max && j <= j_max && k >= k_min && k <= N/2){
                    f[i*N2+j*N+k] = 200.0;
                }
                else {
                    f[i*N2+j*N+k] = 0.0;
                }
            }
        }
    }
}
