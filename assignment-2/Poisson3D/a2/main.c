#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "datatools.h"
#include "jacobi.h"
#include "gauss.h"

void print_matrix(int n, double **Matrix)
{
    int j, i;
    for (j = 0; j < n; j++)
    {
        for (i = 0; i < n; i++)
        {
            printf("%.2lf ", Matrix[i][j]);
        }
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    int N = 1000;
    int model = 0;
    double threshold = 1e-3;
    double **u, **f;
    int num_iterations = 10000;
    
    if (argc == 2)
    {
        model = atoi(argv[1]);
    }
    else if (argc == 3)
    {
        model = atoi(argv[1]);
        N = atoi(argv[2]);
    }
    else if (argc == 4)
    {
        model = atoi(argv[1]);
        N = atoi(argv[2]);
        num_iterations = atoi(argv[3]);
    }
    else if (argc == 5)
    {
        model = atoi(argv[1]);
        N = atoi(argv[2]);
        num_iterations = atoi(argv[3]);
        threshold = atoi(argv[4]);
    }
    
    u = malloc_2d(N + 2, N + 2);
    f = malloc_2d(N + 2, N + 2);
    if (u == NULL || f == NULL)
    {
        printf("Memory allocation error...\n");
        exit(EXIT_FAILURE);
    }
    
    init_data(N, u, f);
   
    if (model == 0)
    {
        printf("Jacobi method \n");
        printf("N: %d\n", N);
        jacobi(N, num_iterations, f, u, threshold);
    }
    else
    {
        printf("Gausss method \n");
        printf("N: %d\n", N);
        gauss(N, num_iterations, f, u, threshold);
    }
    free_2d(u);
    free_2d(f);
    return 0;
}
