#ifndef __DISTCHECK_H
#define __DISTCHECK_H

#include "data.h"

#ifdef ARRAY_OF_STRUCTS
double distcheck(particle_t *, int);
#else
double distcheck(particle_t, int);
#endif

#define CHECK_FLOP 1	// put the right value here!
#endif
