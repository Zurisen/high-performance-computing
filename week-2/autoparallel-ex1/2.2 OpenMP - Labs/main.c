#include <stdio.h>
#include <stdlib.h>
#include "pi_func.h"

int main ( int n)   {
	double Pi;

	Pi = my_phi_func(n);

	printf("phi: %f\n", Pi);

	return(0);
}



