
#include <math.h>
#include <cblas.h>

void matmult_mnk(int M, int N, int K, double **A, double **B, double **C) {
	for (int l = 0; l < M*N; l++) {
		C[0][l] = 0;
	}
    	for (int m = 0; m < M; m++) {
    		for (int n = 0; n < N; n++) {
			for (int k = 0; k < K; k++) {
				C[m][n] += A[m][k] * B[k][n];
		}
    	}	
    }
}

void matmult_mkn(int M, int N, int K, double **A, double **B, double **C) {
	for (int l = 0; l < M*N; l++) {
		C[0][l] = 0;
	}
    for (int m = 0; m < M; m++) {
    	for (int k = 0; k < K; k++) {
			for (int n = 0; n < N; n++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}

void matmult_nmk(int M, int N, int K, double **A, double **B, double **C) {
	for (int l = 0; l < M*N; l++) {
		C[0][l] = 0;
	}
    for (int n = 0; n < N; n++) {
    	for (int m = 0; m < M; m++) {
			for (int k = 0; k < K; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}

void matmult_nkm(int M, int N, int K, double **A, double **B, double **C) {
	for (int l = 0; l < M*N; l++) {
		C[0][l] = 0;
	}
    for (int n = 0; n < N; n++) {
    	for (int k = 0; k < K; k++) {
			for (int m = 0; m < M; m++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}

void matmult_kmn(int M, int N, int K, double **A, double **B, double **C) {
	for (int l = 0; l < M*N; l++) {
		C[0][l] = 0;
	}
    for (int k = 0; k < K; k++) {
    	for (int m = 0; m < M; m++) {
			for (int n = 0; n < N; n++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}

void matmult_knm(int M, int N, int K, double **A, double **B, double **C) {
	for (int l = 0; l < M*N; l++) {
		C[0][l] = 0;
	}
    for (int k = 0; k < K; k++) {
    	for (int n = 0; n < N; n++) {
			for (int m = 0; m < M; m++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}


void matmult_nat(int M, int N, int K, double **A, double **B, double **C) {
	for (int m = 0; m < M; m++) {
		for (int n = 0; n < N; n++) {
			C[m][n] = 0;
			for (int k = 0; k < K; k++) {
				C[m][n] += A[m][k] * B[k][n];
			}
    	}	
    }
}

void matmult_lib(int M, int N, int K, double **A, double **B, double **C) {
	int layout, TRANSA, TRANSB, LDA, LDB, LDC;
	double alpha, beta;

    layout = 101; // rowmajor
	TRANSA = 111; // A is not transposed
	TRANSB = 111; // B is not transposed


	LDA = fmax(1,K); // leading dimension of A
	LDB = fmax(1,N); // leading dimension of B
	LDC = fmax(1,N); // leading dimension of C

	alpha = 1.0; // no scaling
	beta = 0.0; //

	cblas_dgemm(layout, TRANSA, TRANSB, M, N, K, alpha, *A, LDA, *B, LDB, beta, *C, LDC);    
}

void matmult_blk(int M, int N, int K, double **A, double **B, double **C, int bs) 
{
	// MKN

	bs = fmax(1, fmin(bs, K));

	for (int i = 0; i < M*N; i++)
		C[0][i] = 0.0;

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
