#include <stdlib.h>
#include <stdio.h>

/* matrix multiplication times vector function */
void mxv(int m, int n, double** a, double*  b, double* c) {
	int i, j;
	double sum;

	for (i = 0; i < m; i++) {
		sum = 0.0;
		for (j = 0; j < n; j++) {
			sum += a[i][j] * b[j];
		}
		c[i] = sum;
	}
}

/* Initialization of 2d and 1d arrays */
void init_2d (double max_val, int m, int n, double** A) {
	int i, j;
	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			A[i][j] = (double)rand()*max_val/RAND_MAX;
		}
	}
}

void init_1d (double max_val, int k, double* A) {
	int i;
	for (i = 0; i < k; i++) {
		A[i] = (double)rand()*max_val/RAND_MAX;
	}
}

/* allocating memory */
double** malloc_2d(int m, int n) {
	int i;

	if (m <= 0 || n <= 0)
		return NULL;

	double** A = malloc(m* sizeof(double*));
	if (A == NULL)
		return NULL;

	A[0] = malloc(m * n * sizeof(double));
	if (A[0] == NULL) {
		free(A);
		return NULL;
	}

	for (i = 1; i < m; i++)
		A[i] = A[0] + i * n;

	return A;
}

double* malloc_1d(int k) {
	if (k <= 0)
		return NULL;

	double* A = malloc(k * sizeof(double));
	if (A == NULL) {
		free(A);
		return NULL;
	}

	return A;
}

/* freeing the memory */
void free_2d(double** A) {
	free(A[0]);
	free(A);
}

void free_1d(double* A) {
	free(A);
}

/* printing 2d and 1d arrays */
void print_2d(int m, int n, double** A) {
	int i, j;

	for (i = 0; i < m; i++) {
		for (j = 0; j < n; j++) {
			printf("%8.3f ", A[i][j]);
		}
		printf("\n");
	}
}

void print_1d(int k, double* A) {
	int i;
	for (i = 0; i < k; i++) {
		printf("%8.3f \n", A[i]);
	}
}
	