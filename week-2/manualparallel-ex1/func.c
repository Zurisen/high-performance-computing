#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif


double piloop(int N){
	int i;
	double N_inv, tmp_denum, sum_tot;
	sum_tot = 0.0;
	N_inv = pow(N, -1);

	double *sum_tmp = malloc(N * sizeof(double));

	// parallelization of loop
	#pragma omp parallel shared(N, N_inv, sum_tmp) private (i, tmp_denum)
	#pragma omp for
	for (i = 1; i < N; i++){
		tmp_denum = 1 + pow((i-0.5)*N_inv, 2);
		sum_tmp[i] = 4 * pow(tmp_denum, -1);
	}

	// this loop cannot be parallelized
	for (i = 0; i < N; i++){
		sum_tot += sum_tmp[i];
	}
	
	sum_tot = sum_tot * N_inv;

	//free
	free(sum_tmp);

	return(sum_tot);
}
