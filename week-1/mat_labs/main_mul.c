#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "datatools.h"		/* helper functions	        */
#include "lib.h"		/* my matrix add fucntion	*/
#define NREPEAT 100		/* repeat count for the experiment loop */

#define mytimer clock
#define delta_t(a,b) (1e3 * (b - a) / CLOCKS_PER_SEC)

int
main(int argc, char *argv[]) {

    int    i, m, n, N = NREPEAT;
    double **A, **B, **C;
    double tcpu1; 

    clock_t t1, t2;

    int k=5;

    for (m = 200; m <= 3500; m += 300) {
	n = m + 25;

	/* Allocate memory */
	A = malloc_2d(m, k);
	B = malloc_2d(k, n);
	C = malloc_2d(m, n);
	if (A == NULL || B == NULL | C == NULL) {
	    fprintf(stderr, "Memory allocation error...\n");
	    exit(EXIT_FAILURE);
	}

	/* initialize with useful data - last argument is reference */
	init_mat(m, k, 1, 1, A);
	init_mat(k, n, 2, 2, B);

	/* timings */
	t1 = mytimer();
	for (i = 0; i < N; i++)
	    matmat(m, n, k, A, B, C);
	t2 = mytimer();
	tcpu1 = delta_t(t1, t2) / N;

	/* Print n and results  */
	printf("MATRIX X MATRIX\n");
	printf("%4d %4d %8.3f\n", m, n,tcpu1);

	/* Free memory */
	free_2d(A);
	free_2d(B);
	free_2d(C);
		
    }

    return EXIT_SUCCESS;
}