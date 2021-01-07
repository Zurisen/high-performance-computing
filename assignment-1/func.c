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

/* Matrix multiplications using batches */
void matmult_blk(int M, int N, int K, double **A, double **B, double **C, int bs) 
{
	/* for the best performing permutation (mkn) */
	bs = fmax(1, fmin(bs, K));

	for (int m0 = 0; m0 < M; m0 += bs)
	{
		for (int k0 = 0; k0 < K; k0 += bs)
		{
			for (int n0 = 0; n0 < N; n0 += bs)
			{
				for (int m = m0; m < fmin(m0 + bs, M); m++)
				{
					for (int k = k0; k < fmin(k0 + bs, K); k++)
					{
						for (int n = n0; n < fmin(n0 + bs, N); n++)
							C[m][n] += A[m][k] * B[k][n];
					}
				}
			}
		}
	}
}
