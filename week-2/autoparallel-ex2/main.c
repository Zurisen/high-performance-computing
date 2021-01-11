/* exercise 2 week 2 - Matrix multiplication times a vector */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "func.h"

int main() {
	/* declaring */
	int m, n;
	double** A;
	double* B;
	double* C;
	double max_val;

	/* definitions */
	m = 3000;
	n = 3000;
	max_val = 10.0;

	/* allocating memory */
	A = malloc_2d(m, n);
	B = malloc_1d(n);
	C = malloc_1d(n);

	if ((A == NULL) || (B == NULL) || (C == NULL)) {
		fprintf(stderr, "Memory allocation error...\n");
		exit(1);
	}

	/* Initialize data */
	init_2d(max_val, m, n, A);
	init_1d(max_val, n, B);

	/* Matrix times vector */
	mxv(m, n, A, B, C);
	
	/* Free array allocation */
	free_2d(A);
	free_1d(B);
	free_1d(C);	
	/* Print results
	print_2d(m, n, a);
	printf("\n");
	print_1d(n, b);
	printf("\n");
	print_1d(n, c);
	printf("\n");
	*/

	return(0);
}
