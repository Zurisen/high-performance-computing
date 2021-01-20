#include <stdlib.h>
#include <stdio.h>

/* Initialization of 2d and 1d arrays */
void init_2d (double max_val, int m, int n, double* A) {
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