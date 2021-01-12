/* jacobi.h - Poisson problem 
 *
 * $Id: jacobi.h,v 1.1 2006/09/28 10:12:58 bd Exp bd $
 */

#ifndef _JACOBI_H
#define _JACOBI_H

void jacobi(double*** uNew, double*** uOld, double*** f, int N, int iter_max, double* iter, double gridSpace, double tolerance);

#endif
