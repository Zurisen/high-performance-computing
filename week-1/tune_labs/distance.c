#include "data.h"
#include "distance.h"
#include "unistd.h"
#include "math.h"

#ifdef ARRAY_OF_STRUCTS
double 
distance(particle_t *p, int n) {
    
    double dist = -99.0;
    int i;
    for (i=0; i<n; i++){
	p[i].dist = sqrt(pow(p[i].x,2) + pow(p[i].y,2) + pow(p[i].z,2));
	dist += p[i].dist; 	
    }	    

    return dist;
}
#else
double 
distance(particle_t p, int n) {

    double dist = -99.0;
    int i;
    for (i=0; i<n; i++){
        p.dist[i] = sqrt(pow(p.x[i],2) + pow(p.y[i],2) + pow(p.z[i],2));
        dist += p.dist[i];
    }

    return dist;
}
#endif
