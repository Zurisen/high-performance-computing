#include <stdio.h>
#include <math.h>
#include <stdlib.h>

double piloop(int N){
	
	double N_inv, tmp_denum, sum_tot;
	sum_tot = 0.0;
	N_inv = pow(N, -1);

	double *sum_tmp = malloc(N * sizeof(double));

	//loop
	for (int i = 1; i < N; i++){
		tmp_denum = 1 + pow((i-0.5)*N_inv, 2);
		sum_tmp[i] = 4 * pow(tmp_denum, -1);
	}

	for (int i = 0; i < N; i++){
		sum_tot += sum_tmp[i];
	}
	
	sum_tot = sum_tot * N_inv;

	//free
	free(sum_tmp);

	return(sum_tot);
}
