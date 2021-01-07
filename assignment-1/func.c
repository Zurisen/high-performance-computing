#include <math.h>
#include <cblas.h>

/* permutations of loops for matrix multiplication */
void matmult_mnk(int M, int N, int K, double **A, double **B, double **C) {
    int k, n, m;

    /* fill C matrix */
    for (m = 0; m < M; m++) {
    	for (n = 0; n < N; n++) {
    		C[m][n] = 0;
    	}
    }

    /* matrix multiplication */
    for (m = 0; m < M; m++) {
    	for (n = 0; n < N; n++) {
			for (k = 0; k < K; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}

void matmult_mkn(int M, int N, int K, double **A, double **B, double **C) {
    int k, n, m;

    /* fill C matrix */
    for (m = 0; m < M; m++) {
    	for (n = 0; n < N; n++) {
    		C[m][n] = 0;
    	}
    }

    /* matrix multiplication */
    for (m = 0; m < M; m++) {
    	for (k = 0; k < K; k++) {
			for (n = 0; n < N; n++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}

void matmult_nmk(int M, int N, int K, double **A, double **B, double **C) {
    int k, n, m;

    /* fill C matrix */
    for (m = 0; m < M; m++) {
    	for (n = 0; n < N; n++) {
    		C[m][n] = 0;
    	}
    }

    /* matrix multiplication */
    for (n = 0; n < N; n++) {
    	for (m = 0; m < M; m++) {
			for (k = 0; k < K; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}

void matmult_nkm(int M, int N, int K, double **A, double **B, double **C) {
    int k, n, m;

    /* fill C matrix */
    for (m = 0; m < M; m++) {
    	for (n = 0; n < N; n++) {
    		C[m][n] = 0;
    	}
    }

    /* matrix multiplication */
    for (n = 0; n < N; n++) {
    	for (k = 0; k < K; k++) {
			for (m = 0; m < M; m++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}

void matmult_kmn(int M, int N, int K, double **A, double **B, double **C) {
    int k, n, m;

    /* fill C matrix */
    for (m = 0; m < M; m++) {
    	for (n = 0; n < N; n++) {
    		C[m][n] = 0;
    	}
    }

    /* matrix multiplication */
    for (k = 0; k < K; k++) {
    	for (m = 0; m < M; m++) {
			for (n = 0; n < N; n++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}

void matmult_knm(int M, int N, int K, double **A, double **B, double **C) {
    int k, n, m;

    /* fill C matrix */
    for (m = 0; m < M; m++) {
    	for (n = 0; n < N; n++) {
    		C[m][n] = 0;
    	}
    }

    /* matrix multiplication */
    for (k = 0; k < K; k++) {
    	for (n = 0; n < N; n++) {
			for (m = 0; m < M; m++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}

/* Original permutation for matrix multiplication (mnk)*/
void matmult_nat(int M, int N, int K, double **A, double **B, double **C) {
	int k, n, m;

	for (m = 0; m < M; m++) {
		for (n = 0; n < N; n++) {
			C[m][n] = 0;
			for (k = 0; k < K; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}

/* Native CBLAS implementation of matrix multiplication */
void matmult_lib(int M, int N, int K, double **A, double **B, double **C) {
        double alpha = 1.0, beta = 0.0;
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,alpha,A[0],K,B[0],N,beta,C[0],N);
}

/* Matrix multiplications using batches (for best permutation mkn)*/
void matmult_blk(int M, int N, int K, double **A, double **B, double **C, int bs) 
{   

    int m0, n0, m, k, n;

	/* case batch is too large */
	bs = fmax(1, fmin(bs, K));

    /* Fill in C matrix */
    for (m0 = 0; i < M; i++) {
        for (n0 = 0; j < N; j++) {
            C[i][j] == 0;
        }
    }

    /* Matrix multiplication with batches */
	for (m0 = 0; m0 < M; m0 += bs) {
		for (k0 = 0; k0 < K; k0 += bs) {
			for (n0 = 0; n0 < N; n0 += bs) {
				for (m = m0; m < fmin(m0 + bs, M); m++) {
					for (k = k0; k < fmin(k0 + bs, K); k++) {
						for (n = n0; n < fmin(n0 + bs, N); n++)
							C[m][n] += A[m][k] * B[k][n];
					}
				}
			}
		}
	}
}