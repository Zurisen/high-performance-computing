#include <stdio.h>
#include <math.h>
#include "datatools.h"
#include <stdlib.h>

void jacobi(int N, int num_iterations, double **f, double **u, double error)
{

	int i, j;
	int k = 0;
	double dist = 100000000000.0;
	double **u_prev = malloc_2d(N + 2, N + 2);
	double delta_square = (2.0 / (N + 1)) * (2.0 / (N + 1));

	error *= error;

	for (i = 0; i <= N + 1; i++)
	{
		for (j = 0; j <= N + 1; j++)
		{
			u_prev[i][j] = u[i][j];
		}
	}

	double **temp = NULL;

	
	while (dist > error && k < num_iterations)
	{
		dist = 0.0;

		for (i = 1; i <= N; i++)
		{
			for (j = 1; j <= N; j++)
			{
				u[i][j] = 0.25 * (u_prev[i - 1][j] + u_prev[i + 1][j] + u_prev[i][j - 1] + u_prev[i][j + 1] + delta_square * f[i][j]);
				dist += (u[i][j] - u_prev[i][j]) * (u[i][j] - u_prev[i][j]);
			}
		}
		temp = u;
		u = u_prev;
		u_prev = temp;

		k += 1;
	}
	free_2d(u_prev);
	
}
