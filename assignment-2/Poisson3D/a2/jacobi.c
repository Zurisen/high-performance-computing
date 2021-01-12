#include <stdio.h>
#include <math.h>
#include "datatools.h"
#include <stdlib.h>

void jacobi(double ***f, double ***u,double ***temp, int N, int num_iterations, double error)
{

	int i, j, k;
	int it = 0;
	double dist = 100000000000.0;
	double ***u_prev = d_malloc_3d(N + 2, N + 2, N + 2);
	double delta_square = (2.0 / (N + 1)) * (2.0 / (N + 1))* (2.0 / (N + 1));

	error *= error;

	for (i = 0; i <= N + 1; i++)
	{
		for (j = 0; j <= N + 1; j++)
		{
			for (j = 0; j <= N + 1; j++)
			{
				u_prev[i][j][k] = u[i][j][k];
			}
		}
	}


	
	while (dist > error && it < num_iterations)
	{
		dist = 0.0;

		for (i = 1; i <= N; i++)
		{
			for (j = 1; j <= N; j++)
			{
				for (k = 1; k <= N; k++)
				{
					u[i][j] = (1/6) * (u_prev[i - 1][j][k] + u_prev[i + 1][j][k] + u_prev[i][j - 1][k] + u_prev[i][j + 1][k] + u_prev[i][j][k - 1] + u_prev[i][j][k + 1] + delta_square * f[i][j][k]);
				dist += (u[i][j][k] - u_prev[i][j][k]) * (u[i][j][k] - u_prev[i][j][k]);
				}
			}
		}
		temp = u;
		u = u_prev;
		u_prev = temp;

		it += 1;
	}
	free_3d(u_prev);
	printf("Iterations: %d\nDistance: %.18f\n", it, dist);
}
