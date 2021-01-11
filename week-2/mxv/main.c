/* exercise 2 week 2 - Matrix multiplication times a vector */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mxv.h"

int main(int argc, char *argv[]) {
	/* declaring */
	int m, n;
	double** a;
	double* b;
	double* c;
	clock_t start, end;
	double time_taken, max_val;

	/* definitions */
	m = 30000;
	n = 30000;
	max_val = 10.0;

	/* allocating memory */
	a = malloc_2d(m, n);
	b = malloc_1d(n);
	c = malloc_1d(n);

	if ((a == NULL) || (b == NULL) || (c == NULL)) {
		fprintf(stderr, "Memory allocation error...\n");
		exit(1);
	}

	/* Initialize data */
	init_2d(max_val, m, n, a);
	init_1d(max_val, n, b);

	/* Matrix times vector */
	start = clock();
	mxv(m, n, a, b, c);
	end = clock();
	
	/* Print results
	print_2d(m, n, a);
	printf("\n");
	print_1d(n, b);
	printf("\n");
	print_1d(n, c);
	printf("\n");
	*/

	time_taken = (double)(end-start)/(double)(CLOCKS_PER_SEC);
	printf("Time: %f \n", time_taken);
	return(0);
}
