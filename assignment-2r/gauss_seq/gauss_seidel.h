/* gauss_seidel.h - Poisson problem
 *
 */
#ifndef _GAUSS_SEIDEL_H
#define _GAUSS_SEIDEL_H


void gauss_seidel(double*** uNew, double*** uOld, double*** uSwap, double*** f, int N, int iter_max, double gridSpace, double tolerance);


#endif