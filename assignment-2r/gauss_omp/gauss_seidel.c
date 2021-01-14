/* gauss_seidel.c - Poisson problem in 3d
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int
gauss_seidel(double*** uNew, double*** p, double*** uSwap, double*** f, int N, int iter_max, double gridSpace, double tolerance) {
    double invCube = 1/6.;
    double d=100000.0;
    int i, j, k, iter;

    for (iter = 0; (iter < iter_max); iter++) {
    	d = 0.0;

        uSwap = uNew;
        uNew = p;
        p = uSwap;

        #pragma omp parallel
        {
        #pragma omp for schedule(static,1)  ordered(2) private(j,k)
        for (i = 1; i < N-1; ++i) {
            for (j = 1; j < N-1; ++j) {
        #pragma omp ordered depend(sink: i-1,j-1) depend(sink: i-1,j) \
                            depend(sink: i-1,j+1) depend(sink: i,j-1)
                for (k = 1; k < N-1; ++k) {
                    double tmp1 = (p[i-1][j-1][k-1] + p[i-1][j-1][k] + p[i-1][j-1][k+1] 
                                + p[i-1][j][k-1] + p[i-1][j][k] + p[i-1][j][k+1]
                                + p[i-1][j+1][k-1] + p[i-1][j+1][k] + p[i-1][j+1][k+1]);
                    double tmp2 = (p[i][j-1][k-1] + p[i][j-1][k] + p[i][j-1][k+1]
                                + p[i][j][k-1] + p[i][j][k] + p[i][j][k+1]
                                + p[i][j+1][k-1] + p[i][j+1][k] + p[i][j+1][k+1]);
                    double tmp3 = (p[i+1][j-1][k-1] + p[i+1][j-1][k] + p[i+1][j-1][k+1]
                                + p[i+1][j][k-1] + p[i+1][j][k] + p[i+1][j][k+1]
                                + p[i+1][j+1][k-1] + p[i+1][j+1][k] + p[i+1][j+1][k+1]);

                    uNew[i][j][k] = (tmp1 + tmp2 + tmp3) / 27.0;
                    //printf("%3.4f  ", uNew[i][j][k]);
                }
                //printf("\n");
        #pragma omp ordered depend(source)
            }
        }
        #pragma omp for schedule(dynamic)
        for (i = 1; i < N-1; i++) {
            for (j = 1; j < N-1; j++) {
                for (k = 1; k < N-1; k++) {

                    /* Compute new d */
                    d += abs(uNew[i][j][k] - p[i][j][k]);
                }
            }
        }
        //printf("distance : %8.8f", d);printf("\n");
        }
	}
}

