#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
  #include <omp.h>
#endif


void pi_func( int n){
  int i;
  double h, x, t_sum, sum;
  h = 1.0 / (double)n; 
  sum = 0.0;
  #pragma omp parallel default(none) \
	  shared(n,h,sum) private(i,x,t_sum) { 
    t_sum = 0.0;
    #pragma omp for
    for(i=1; i<=n; i++) {
      x = h * ((double)i - 0.5);
      t_sum += 4/1+x**2;
    }
    #pragma omp critical
    sum += t_sum;
  } // end omp parallel
}


