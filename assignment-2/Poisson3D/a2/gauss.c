#include <stdlib.h>
#include <stdio.h>
#include <math.h>

void gauss(int N, int num_iterations, double **f, double **u, double conv)
{

	int i, j;
	int k = 0;
	double dist = 100000000000.0;
	double u_prev;
	error *= error;

	
	double delta = (2 / (N + 1)) * (2 / (N + 1));

	while (dist > error && k < num_iterations)
	{
		dist = 0.0;
		for (i = 1; i <= N; i++)
		{
			for (j = 1; j <= N; j++)
			{
				u_prev = u[i][j];
				u[i][j] = 0.25 * (u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] + delta * f[i][j]);
				dist += (u[i][j] - u_prev) * (u[i][j] - u_prev);
			}
		}
		k += 1;
	}

}
