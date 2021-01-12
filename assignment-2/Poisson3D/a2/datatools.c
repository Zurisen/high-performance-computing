/* datatools.c - support functions for the matrix examples
 *
 * Author:  Bernd Dammann, DTU Compute
 * Version: $Revision: 1.2 $ $Date: 2015/11/10 11:03:12 $
 */
#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>

#include "datatools.h"

void init_data(int N, double **u, double **f)
{
    int i, j;
    double x, y;
    double h = 2.0 / (N + 1);

    for (i = 1; i <= N; i++)
    {
        for (j = 1; j <= N; j++)
        {
            u[i][j] = 10.0;
        }
    }
    for (i = 0; i <= N + 1; i++)
    {
        u[i][0] = 20.0;
        u[0][i] = 20.0;
        u[i][N + 1] = 20.0;
        u[N + 1][i] = 0.0;
    }
    for (i = 0; i <= N + 1; i++)
    {
        for (j = 0; j <= N + 1; j++)
        {
            f[i][j] = 0.0;
            x = -1 + j * h;
            y = 1 - i * h;
            // 0≤x≤1/3, −2/3≤y≤−1/3
            if (x >= 0 && x <= 1.0 / 3 && y >= -2.0 / 3 && y <= -1.0 / 3)
            {
                f[i][j] = 200.0;
            }
        }
    }
}

int check_results(char *comment, int m, int n, double **A)
{

    double relerr;
    double *a = A[0];
    double ref = 3.0;
    int i, errors = 0;
    char *marker;
    double TOL = 100.0 * DBL_EPSILON;

    if ((marker = (char *)malloc(m * n * sizeof(char))) == NULL)
    {
        perror("array marker");
        exit(-1);
    }

    for (i = 0; i < m * n; i++)
    {
        relerr = fabs((a[i] - ref));
        if (relerr <= TOL)
        {
            marker[i] = ' ';
        }
        else
        {
            errors++;
            marker[i] = '*';
        }
    }
    if (errors > 0)
    {
        printf("Routine: %s\n", comment);
        printf("Found %d differences in results for m=%d n=%d:\n",
               errors, m, n);
        for (i = 0; i < m * n; i++)
            printf("\t%c a[%d]=%f ref[%d]=%f\n", marker[i], i, a[i], i, ref);
    }
    return (errors);
}

/* Routine for allocating two-dimensional array */
double **
malloc_2d(int m, int n)
{
    int i;

    if (m <= 0 || n <= 0)
        return NULL;

    double **A = malloc(m * sizeof(double *));
    if (A == NULL)
        return NULL;

    A[0] = malloc(m * n * sizeof(double));
    if (A[0] == NULL)
    {
        free(A);
        return NULL;
    }
    for (i = 1; i < m; i++)
        A[i] = A[0] + i * n;

    return A;
}

void free_3d(double ***A)
{
    free(A[0]);
    free(A);
}

double ***
d_malloc_3d(int m, int n, int k) {

    if (m <= 0 || n <= 0 || k <= 0)
        return NULL;

    double ***array3D = (double ***)malloc(m * sizeof(double **) +
                                           m * n * sizeof(double *) +
                                           m * n * k * sizeof(double));
    if (array3D == NULL) {
        return NULL;
    }

    for(int i = 0; i < m; i++) {
        array3D[i] = (double **) array3D + m + i * n ;
    }

    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            array3D[i][j] = (double *) array3D + m + m * n + i * n * k + j * k;
        }
    }

    return array3D;
