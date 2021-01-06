#!/bin/sh

CC=${1-"gcc"}

NPARTS="2000 3000 4000 5000 7500 10000 20000 40000 80000 200000 400000 800000 1200000 1600000 3000000"
LOOPS=1000
LOGEXT=$CC.dat

/bin/rm -f aos.$LOGEXT soa.$LOGEXT
for particles in $NPARTS
do
    ./aos.${CC} $LOOPS $particles | grep -v CPU >> aos.$LOGEXT
    ./soa.${CC} $LOOPS $particles | grep -v CPU >> soa.$LOGEXT
done

# time to say 'Good bye' ;-)
#
exit 0

