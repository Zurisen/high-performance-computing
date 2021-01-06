#include "data.h"
#include "distcheck.h"
#include <unistd.h>

#ifdef ARRAY_OF_STRUCTS
double 
distcheck(particle_t *p, int n) {

    double dist = -99.0;
    int i;
    for (i=0; i<n; i++){
        dist += p[i].dist;
    }
    
    return dist;
}
#else
double 
distcheck(particle_t p, int n) {

    double dist = -99.0;
    int i;
    for (i=0; i<n; i++){
        dist += p.dist[i];
    }
    return dist;
}
#endif
