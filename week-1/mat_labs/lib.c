#include "lib.h"
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

int matadd(int m, int n, double** A, double** B, double** C) {
  int i, j;

  for (i = 0; i < m; i++) {
    for (j = 0; j < n; j++) {
      C[i][j] = A[i][j] + B[i][j];
    }
  }
}

int matvec(int m, int k, double** A, double* B, double* C) {
  int i, j;

  for (i = 0; i < k; i++) {
    C[i] = 0;
  }

  for (i = 0; i < m; i++) {
    for (j = 0; j < k; j++) {
      C[j] += A[i][j] * B[j];
    }
  }
};

int matmat(int m, int n, int k, double **A, double **B, double **C) {
    double tmp_val;

    for(int i = 0; i < m; i++) {
    	for(int j = 0; j < n; j++) {
			C[i][j] = 0;
			for(int l = 0; l < k; l++) {
				C[i][j] += A[i][l] * B[l][j];
			}
    	}	
    }
}

double* malloc_1d(int k) {
  
  int i;

  if (k <= 0)
	  return NULL;

  double* A = malloc(k*sizeof(double));
  if (A == NULL) {
	  free(A);
	  return NULL;
  }

  return A;
}

void free_1d(double *A) {
    free(A);
}

void init_vec(int k, double *A) {

  int i;

  for(i = 0; i < k; i++) {
	  A[i] = 2.0;	    
  }
}

void init_mat(int m, int n, int r, int s, double **A) {
  int i, j, val;
  for( i = 0; i < m; i++) {
    for( j = 0; j < n; j++) {
	    val = i * 10 + j;
	    A[i][j] = val;
    }
  }
}

void init_matA(int m, int k, double **A) {
  int i, j, val;
  for( i = 0; i < m; i++) {
    for( j = 0; j < k; j++) {
            val = (i+1) * 10 + (j+1);
            A[i][j] = val;
    }
  }
}

void init_matB(int k, int n, double **B) {
  int i, j, val;
  for( i = 0; i < k; i++) {
    for( j = 0; j < n; j++) {
            val = (i+1) * 20 + (j+1);
            B[i][j] = val;
    }
  }
}

double* flatten_mat(double** A, int m, int n) {
  int i, j;

  double* B = malloc_1d(n * m);

  if (B == NULL) {
    free(B);
    return NULL;
  }

  for (i = 0; i < m; i++)
    for (j = 0; j < n; j++)
      B[i * m + j] = A[i][j];

  return B;
}
