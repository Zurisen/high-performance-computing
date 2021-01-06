#include <stdio.h>
#include <stdlib.h>
#include "data.h"
#include "init_data.h"

#ifdef ARRAY_OF_STRUCTS
void
init_data(particle_t *data, int nparts) {

    for(int i = 0; i < nparts; i++ ) {
	data[i].x = drand48();
	data[i].y = drand48();
	data[i].z = drand48();
    }
}
#else
void
init_data(particle_t data, int nparts) {

    for(int i = 0; i < nparts; i++ ) {
	data.x[i] = drand48();
	data.y[i] = drand48();
	data.z[i] = drand48();
    }
}
#endif
