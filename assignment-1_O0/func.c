#include <math.h>
#include <cblas.h>
// Have also these lib on matmult_blk. They are introduced here just in case are necessary.
#include <stdlib.h>
#include <stdio.h>



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

/* Native CBLAS implementation of matrix multiplication */
void matmult_lib(int M, int N, int K, double **A, double **B, double **C) {
        double alpha = 1.0, beta = 0.0;
        cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,M,N,K,alpha,A[0],K,B[0],N,beta,C[0],N);
}

/* Matrix multiplications using batches (for best permutation mkn)*/
void matmult_blk(int M, int N, int K, double **A, double **B, double **C, int bs) 
{   

    int m0, n0, k0, m, k, n;

	/* case batch is too large */
	bs = fmax(1, fmin(bs, K));

    /* Fill in C matrix */
    for (m0 = 0; m0 < M; m0++) {
        for (n0 = 0; n0 < N; n0++) {
            C[m0][n0] = 0;
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

//void matmult_blk2(int m, int n, int k, double **A, double **B, double **C, int bs){
    // Matrix C Is the result of multiplying Matrix A with Matrix B
    // m : number of rows of A and number of rows of C 
    // k : number of cols of A and number of rows of B
    // n : number of cols of B and number of cols of C 
    // bs : blocks size
    // The algorithm is based on mnk permutation, we can change it as soon as
    // we check which one is the optimal.

    // As soon as we check the tier opimal block sizes, we can program them 
    //Without bucles "for" to obtain a faster performance.

    //int i, j, l, i0, j0, l0;
    //int limit_i, limit_j, limit_l;

    //for (i = 0; i < m; i += bs){
    //    for (j = 0; j < n; j += bs){
    //        for (l = 0; l < k; l += bs){
    //            // define limites for not surpase the matrix sizes.
    //            if (i + bs > m){
    //                limit_i = m;
    //            }
    //            else{
    //                limit_i = i + bs;
    //            }
    //            if (j + bs > n){
    //                limit_j = n;
    //            }
    //            else{
    //                limit_j = j + bs;
    //            }    
    //            if (l + bs > k){
    //                limit_l = k;
    //            }
    //            else{
    //                limit_l = l + bs;
    //            }

    //            // Block multiplication                  
    //            for (i0 = i; i0 < limit_i; i0++){
    //                for (l0 = l; l0 < limit_l; l0++){
    //                    for (j0 = j; j0 < limit_j; j0++){
    //                        C[i0][j0] += A[i0][l0] * B[l0][j0];
    //                    }
    //                }
    //            }
    //        }
    //    }
    //}
