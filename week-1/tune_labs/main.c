#include <stdio.h>
#include <stdlib.h>
#include "data.h"
#include "xtime.h"
#include "init_data.h"
#include "distance.h"
#include "distcheck.h"

int
main(int argc, char *argv[]) {

    #ifdef ARRAY_OF_STRUCTS
    particle_t 	*parts = NULL;
    #else
    particle_t	parts;
    #endif
    double 	total_length, check_length;
    int 	i, loops;
    int		nparts = NUM_OF_PARTS;
    double	ts, te_init, te_dist, te_check, ts_main, te_main;
    double	mf_dist, mf_check, mf_main;
    double	memory, mflops;

    if ( argc >= 2 ) {
	loops = atoi(argv[1]);
    } else {
	loops = 1;
    }
    if ( argc == 3 ) {
	nparts = atoi(argv[2]);
    }

#ifndef DATA_ANALYSIS
    #ifdef ARRAY_OF_STRUCTS
    fprintf(stderr, "Running AOS version with %d particles and %d loops.\n", 
    #else
    fprintf(stderr, "Running SOA version with %d particles and %d loops.\n", 
    #endif
            nparts, loops);
#endif


    // allocate memory
    #ifdef ARRAY_OF_STRUCTS
    if ( (parts = calloc( nparts, sizeof(particle_t) )) == NULL ) {
	perror("main(__LINE__), allocation failed");
	exit(1);
    }
    #else
    if ( (parts.x = calloc( nparts, sizeof(double) )) == NULL ) {
	perror("main(__LINE__), allocation failed");
	exit(1);
    }
    if ( (parts.y = calloc( nparts, sizeof(double) )) == NULL ) {
	perror("main(__LINE__), allocation failed");
	exit(1);
    }
    if ( (parts.z = calloc( nparts, sizeof(double) )) == NULL ) {
	perror("main(__LINE__), allocation failed");
	exit(1);
    }
    if ( (parts.ptype = calloc( nparts, sizeof(char) )) == NULL ) {
	perror("main(__LINE__), allocation failed");
	exit(1);
    }
    if ( (parts.dist = calloc( nparts, sizeof(double) )) == NULL ) {
	perror("main(__LINE__), allocation failed");
	exit(1);
    }
    #endif

    init_timer();

    ts = xtime();
    init_data(parts, nparts);
    te_init = xtime() - ts;

    te_dist  = 0.0;
    te_check = 0.0;
    ts_main = xtime();

    for( i = 0; i < loops; i++ ) {
	ts = xtime();
	total_length = distance(parts, nparts);
	te_dist += (xtime() - ts);

	ts = xtime();
	check_length = distcheck(parts, nparts);
	te_check += (xtime() - ts);
    }

    te_main = xtime() - ts_main;


    #ifdef ARRAY_OF_STRUCTS
    memory = nparts * sizeof(particle_t);
    #else
    memory = nparts * (sizeof(double) * 4 + sizeof(char));
    #endif
    memory /= 1024.0;	// in kbytes

    mflops   = 1.0e-06 * nparts * loops;
    mf_dist  = DIST_FLOP  * mflops / te_dist;
    mf_check = CHECK_FLOP * mflops / te_check;
    mf_main =  (DIST_FLOP + CHECK_FLOP) * mflops / te_main;

#ifndef DATA_ANALYSIS

    printf("Times (secs):\n");
    printf("\tInitialize : %lf\n", te_init);
    printf("\tCalculation: %lf\n", te_dist);
    printf("\tChecks     : %lf\n", te_check);
    printf("\tTotal      : %lf\n", te_main);

    printf("\nTotal length: %lf\n", total_length);
    printf("Check length: %lf\n", check_length);

    printf("Memory footprint (kbytes): %7.2lf\n", memory);
    printf("Size of particle_t (bytes): %d\n", sizeof(particle_t));

#else 

    /*
    printf("%7.2lf %le %le %le %le\n", 
	   memory, te_init, te_dist, te_check, te_main);
    */
    printf("%10.2lf %le %le %le %le\n", 
	   memory, mf_dist, mf_check, mf_main, te_main);
#endif

    #ifdef ARRAY_OF_STRUCTS
    free(parts); 
    #else
    free(parts.x);
    free(parts.y);
    free(parts.z);
    free(parts.ptype);
    free(parts.dist);
    #endif

    return(0);
}
