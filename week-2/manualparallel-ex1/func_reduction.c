#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
	#include <omp.h>
#endif


double pi_fun(int N){
	int i;
	double h, x, sum, t_sum;
	h = 1.0 / (double)N;

	t_sum = 0.0;
	sum = 0.0;

	// parallelization of loop
	#pragma omp parallel shared(N, h) private (i, x, t_sum) reduction(+: sum)
	#pragma omp for
	for (i = 1; i < N; i++){
		x = ((double)i-0.5)*h;
		t_sum = 4/(1+x*x);

		sum += t_sum;
	
	}// end omp parallel

	return(sum*h);
}
