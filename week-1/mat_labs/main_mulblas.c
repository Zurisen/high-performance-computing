#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "datatools.h"		/* helper functions	        */
#include "lib.h"		/* my matrix add fucntion	*/
#include <cblas-atlas.h>	/* Cblas */

#define NREPEAT 100		/* repeat count for the experiment loop */

#define mytimer clock
#define delta_t(a,b) (1e3 * (b - a) / CLOCKS_PER_SEC)

int
main(int argc, char *argv[]) {

    int    i, m, n, N = NREPEAT;
    double **A, **B, **C;
    double tcpu1; 

    clock_t t1, t2;

    int k = 5;
	m = 3;
	n = 2;

	/* Allocate memory */
	A = malloc_2d(m, k);
	B = malloc_2d(k, n);
	C = malloc_2d(m, n);
	if (A == NULL || B == NULL | C == NULL) {
	    fprintf(stderr, "Memory allocation error...\n");
	    exit(EXIT_FAILURE);
	}

	double* a = flatten_mat(A, m, k);
	double* b = flatten_mat(B, k, n);
	double* c = flatten_mat(C, m, n);

	/* initialize with useful data - last argument is reference */
	init_matA(m, k, A);
	init_matB(k, n, B);

	/* timings */
	t1 = mytimer();
	for (i = 0; i < N; i++)
	    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, k, b, n, 1, c, n);
	t2 = mytimer();
	tcpu1 = delta_t(t1, t2) / N;

	/* Print n and results  */
	printf("MATRIX X MATRIX\n");
	printf("%4d %4d %8.3f\n", m, n,tcpu1);
	printf("%8.3f\n", C[0][1]);

	/* Free memory */
	free_2d(A);
	free_2d(B);
	free_2d(C);
	free_1d(a);
	free_1d(b);
	free_1d(c);

    return EXIT_SUCCESS;
}
