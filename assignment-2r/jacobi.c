/* jacobi.c - Poisson problem in 3d
 * 
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void
jacobi(double*** uNew, double*** uOld, double*** uSwap, double*** f, int N, int iter_max, double gridSpace, double tolerance) {
    double invCube = 1/6.;
    double d=100000.0;
   for (int iter = 0; (iter < iter_max || d > tolerance); iter++) {
        d = 0.0;
        for (int i = 1; i < N-1; i++) {
                for (int j = 1; j < N-1; j++) {
                        for (int k = 1; k < N-1; k++) {
				uOld[i][j][k] = uNew[i][j][k];
			}
		}
	}
	
    	for (int i = 1; i < N-1; i++) {
    		for (int j = 1; j < N-1; j++) {
    			for (int k = 1; k < N-1; k++) {

    				/* Compute update of uNew */
    				uNew[i][j][k] = invCube*(uOld[i-1][j][k] + uOld[i+1][j][k] 
    					+ uOld[i][j-1][k] + uOld[i][j+1][k] + uOld[i][j][k-1] + uOld[i][j][k+1] + gridSpace*f[i][j][k]);

    				/* Compute new d */
    				/* NOTE: maybe this can be parallelized better if 
    				splitted into a different nested loop */
				d += abs(uNew[i][j][k] - uOld[i][j][k]);
    			}
    		}
    	}
	}
}