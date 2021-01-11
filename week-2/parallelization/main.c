#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include "func.h"

int main (int argc, char *argv[] ) {
	int N;
	double res;

	if (argc == 2) {
		N = atoi(argv[1]);
	}
	else {
		// default
		N = 1000000000;
	}

	res = piloop(N);
	
	
	printf("pi = %f\n", res);

	return(0);
}
