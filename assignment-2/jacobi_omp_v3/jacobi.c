/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int
jacobi(double*** uNew, double*** uOld, double*** uSwap, double*** f, int N, int iter_max, double gridSpace, double tolerance) {
    double invCube = 1./6.;
    double d=100000.0;
    int i, j, k, iter=0;
	#pragma omp parallel shared(uNew, uOld, uSwap, f, N, gridSpace, tolerance, iter, d) firstprivate(iter_max)
	{ 	
	while (iter < iter_max && d>tolerance){
	#pragma omp barrier
	#pragma omp critical
          {
	    d = 0.0;
            uSwap=uOld;
            uOld=uNew;
            uNew=uSwap;
          }
	   #pragma omp for private(i,j,k) reduction(+:d)
    	   for (i = 1; i < N-1; i++) {
    		  for (j = 1; j < N-1; j++) {
    		  	   for (k = 1; k < N-1; k++) {

    				    /* Compute update of uNew */
    				    uNew[i][j][k] = invCube*(uOld[i-1][j][k] + uOld[i+1][j][k] 
    					   + uOld[i][j-1][k] + uOld[i][j+1][k] + uOld[i][j][k-1] + uOld[i][j][k+1] + gridSpace*f[i][j][k]);
				    d += (uOld[i][j][k]-uNew[i][j][k])*(uOld[i][j][k]-uNew[i][j][k]);
		    }
                }
            }
	  #pragma omp single
	  {
	    iter++;
	    //printf("%i\n",d);	
	  }
    	}
    
    	}
    return iter;
}
