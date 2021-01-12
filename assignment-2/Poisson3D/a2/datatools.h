void init_data(int N, double **u, double **f);

int check_results(char *comment, /* comment string 		 */
									int m,				 /* number of rows               */
									int n,				 /* number of columns            */
									double **a		 /* vector of length m           */
);

double **malloc_2d(int m, /* number of rows               */
									 int n	/* number of columns            */
);

void free_3d(double ***A); /* free data allocated by malloc_2d */

#ifndef __ALLOC_3D_2
#define __ALLOC_3D_2

double ***d_malloc_3d(int m, int n, int k);

#endif /* __ALLOC_3D_2 */
