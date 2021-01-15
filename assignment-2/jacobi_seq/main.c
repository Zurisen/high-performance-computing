/* main.c - Poisson problem in 3D
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include "alloc3d.h"
#include "print.h"
#include "func.h" /* some helper functions */
#include <math.h> /* for pow() */
#include <omp.h>
/*
#ifdef _JACOBI
#include "jacobi.h"
#endif
*/
#include "jacobi.h"

/*
#ifdef _GAUSS_SEIDEL
#include "gauss_seidel.h"
#endif
*/

#include "gauss_seidel.h"


#define N_DEFAULT 100

int
main(int argc, char *argv[]) {

    int 	N = N_DEFAULT;
    int 	iter_max = 1000;
    double	tolerance;
    double	start_T;
    int		output_type = 0;
    char	*output_prefix = "poisson_res";
    char        *output_ext    = "";
    char	output_filename[FILENAME_MAX];
    double 	***u = NULL;
    double  ***uOld = NULL;
    double  ***uSwap = NULL;
    double  ***f = NULL;
    double gridSpace;

    double start, time; /* for timing omp */
    int iter;
    double memory; /* memory footprint */
    /* get the parameters from the command line */
    N         = atoi(argv[1]);	// grid size
    iter_max  = atoi(argv[2]);  // max. no. of iterations
    tolerance = atof(argv[3]);  // tolerance
    start_T   = atof(argv[4]);  // start T for all inner grid points
    if (argc == 6) {
	output_type = atoi(argv[5]);  // ouput type
    }

    gridSpace = 4.*pow(N, -2);

    // allocate memory
    u = d_malloc_3d(N, N, N);
    uOld = d_malloc_3d(N, N, N);
    uSwap = d_malloc_3d(N, N, N);
    f = d_malloc_3d(N, N, N);

    if ((u == NULL) || (uOld == NULL) || (uSwap == NULL) || (f == NULL)) {
        perror("array u: allocation failed");
        exit(-1);
    }

    /* fill in your code here */

    /* Initialize the 3d arrays */
    init_3d(start_T, N, u);
    init_3d(start_T, N, uOld);
    init_f(N, f);

    /* Jacobi method */
    start = omp_get_wtime();
    iter = jacobi(u, uOld, uSwap, f, N, iter_max, gridSpace, tolerance);
    time = omp_get_wtime() - start;
	/* Printing final u matrix
	for (int i=0; i<N; i++){
		printf("\n%d -th layer", i);
		for (int j=0; j<N; j++){
			printf("\n");
			for(int k=0; k<N; k++){
				printf("%f ", u[i][j][k]);
			}
		}
	}
	*/

    /* Gauss Seidel method */
    //gauss_seidel(u, uOld, uSwap, f, N, iter_max, gridSpace, tolerance);
    /* print stats of the run
 *     N: size of grid, iter: iterations, time: (total) time, iterations/per unit time*/
    memory = 3.0*(double)(pow(N,3))*(double)(sizeof(double))*0.001; /* kBytes */
    printf("%i %i %lf %lf %g\n",N,iter,time,(double)iter/time, memory);

    // dump  results if wanted 
    switch(output_type) {
	case 0:
        printf("killing it!");
	    // no output at all
	    break;
	case 3:
	    output_ext = ".bin";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write binary dump to %s: ", output_filename);
	    print_binary(output_filename, N, u);
	    break;
	case 4:
	    output_ext = ".vtk";
	    sprintf(output_filename, "%s_%d%s", output_prefix, N, output_ext);
	    fprintf(stderr, "Write VTK file to %s: ", output_filename);
	    print_vtk(output_filename, N, u);
	    break;
	default:
	    fprintf(stderr, "Non-supported output type!\n");
	    break;
    }

    // de-allocate memory
    free(u);
    free(uOld);
    free(uSwap);
    free(f);

    return(0);
}
