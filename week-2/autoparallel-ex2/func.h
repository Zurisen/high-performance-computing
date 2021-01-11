#ifndef __FUNC_PHI_FUNC_H
#define __FUNC_PHI_FUNC_H

/*matrix multiplication times vector */
void mxv(int m, int n, double** A, double*  B, double* C);

/* Initialization of 2d and 1d arrays */
void init_2d (double max_val, int m, int n, double** A);
void init_1d (double max_val, int k, double* A);

/* allocating memory */
double** malloc_2d(int m, int n);
double* malloc_1d(int k);

/* freeing the memory */
void free_2d(double** A);
void free_1d(double* A);

/* printing 2d and 1d arrays */
void print_2d(int m, int n, double** A);
void print_1d(int k, double* A);

#endif
