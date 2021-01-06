#ifndef __INIT_DATA_H
#define __INIT_DATA_H

#include "data.h"

#ifdef ARRAY_OF_STRUCTS
void init_data(particle_t *, int);
#else
void init_data(particle_t, int);
#endif

#endif 
