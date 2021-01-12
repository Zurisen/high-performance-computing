/* helper functions func.c */

void init_3d(double start_T, int N, double*** u) {

	/* initilaize boundaries with bound conditions */
	/* NOTE: Initial conditions in the corner of the box
	are not well defined */
	for (int w = 0; w < N; w++) {
		for (int v = 0; v < N; v++) {
			u[0][w][v] = 20.0;
			u[N][w][v] = 20.0;
			u[w][0][v] = 20.0;
			u[w][N][v] = 20.0;
			u[w][v][0] = 0.0;
			u[w][v][N] = 0.0;
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