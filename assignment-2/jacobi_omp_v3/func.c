/* helper functions func.c */

#include <stdlib.h>
#include <stdio.h>

void init_3d(double start_T, int N, double*** u) {

	/* initilaize boundaries with bound conditions
	
	NOTE: Initial conditions in the corner of the box
	are not well defined
	*/
	int i,j,k;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			u[0][i][j] = 20.0;
			u[N-1][i][j] = 20.0;
			u[i][N-1][j] = 20.0;
			u[i][j][0] = 20.0;
			u[i][j][N-1] = 20.0;
			u[i][0][j] = 0.0;
		}
	}
	
	/* Fill inside of the cube with starting temperature */
	for (i = 1; i < N-1; i++) {
		for (j = 1; j < N-1; j++) {
			for (k = 1; k < N-1; k++) {
				u[i][j][k] = start_T;
			}
		}
	}

}

void init_f(int N, double*** f) {
	double rangX, rangY, rangMinZ, rangMaxZ;
	rangX = 5*N/16;
	rangY = 0.25*N;
	rangMinZ = N/6;
	rangMaxZ = 0.5*N;
	
	int i,j,k;	
	/* Fill radiator f with zeros */
	for (i = rangX; i < N; i++) {
		for (j = rangY; j < N; j++) {
			for (k = 0; k < N; k++) {
				f[i][j][k] = 0.0;
			}
		}
	}

	/* Fill radiator f with zeros */
	for (i = 0; i < rangX; i++) {
		for (j = 0; j < rangY; j++) {
			for (k = rangMinZ; k < rangMaxZ; k++) {
				f[i][j][k] = 200.0;
			}
		}
	}

}
