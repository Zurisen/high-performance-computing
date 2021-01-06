#ifndef __DATA_H
#define __DATA_H

/* define defaults for some parameters 
 * overwrite at compile time with -D... 
 * or make them dynamic in the source
 */
#define NUM_OF_PARTS 1000000

/* definition of the data structure */
#ifdef ARRAY_OF_STRUCTS
typedef struct particle {
    double x;
    double y;
    double z;
    char   ptype;
    double dist;
} particle_t;

#else
typedef struct particle {
    double *x;
    double *y;
    double *z;
    char   *ptype;
    double *dist;
} particle_t;

#endif

#endif /* _DATA_H */
