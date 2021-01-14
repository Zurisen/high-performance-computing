/* gauss_seidel.h - Poisson problem
 *
 */
#ifndef _GAUSS_SEIDEL_H
#define _GAUSS_SEIDEL_H


int gauss_seidel(double*** uNew, double*** p, double*** uSwap, double*** f, int N, int iter_max, double gridSpace, double tolerance);


#endif
