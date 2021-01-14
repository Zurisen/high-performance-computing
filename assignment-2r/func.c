/* helper functions func.c */

#include <stdlib.h>
#include <stdio.h>

void init_3d(double start_T, int N, double*** u) {

	/* initilaize boundaries with bound conditions
	
	NOTE: Initial conditions in the corner of the box
	are not well defined
	*/
	for (int w = 0; w < N; w++) {
		for (int v = 0; v < N; v++) {
			u[0][w][v] = 20.0;
			u[N-1][w][v] = 20.0;
			u[w][N-1][v] = 20.0;
			u[w][v][0] = 20.0;
			u[w][v][N-1] = 20.0;
			u[w][0][v] = 0.0;
		}
	}
	
	/* Fill inside of the cube with starting temperature */
	for (int i = 1; i < N-1; i++) {
		for (int j = 1; j < N-1; j++) {
			for (int k = 1; k < N-1; k++) {
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

	/* Fill radiator f with zeros */
	for (int i = rangX; i < N; i++) {
		for (int j = rangY; j < N; j++) {
			for (int k = 0; k < N; k++) {
				f[i][j][k] = 0.0;
			}
		}
	}

	/* Fill radiator f with zeros */
	for (int r = 0; r < rangX; r++) {
		for (int s = 0; s < rangY; s++) {
			for (int t = rangMinZ; t < rangMaxZ; t++) {
				f[r][s][t] = 200.0;
			}
		}
	}
}
