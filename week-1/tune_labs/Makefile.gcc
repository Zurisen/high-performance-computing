CC	= gcc
OPT	= -g -O3
CHIP	= 
ISA	= 
DEFS	= -DDATA_ANALYSIS 
LIBS	= -lm
CFLAGS  = $(OPT) $(CHIP) $(ISA) $(DEFS) $(XOPT)

SOURCES = init_data.c main.c distance.c distcheck.c xtime.c
OBJECTS = $(SOURCES:.c=.o)

TARGET_AOS = aos.$(CC)
TARGET_SOA = soa.$(CC)
TARGETS = $(TARGET_AOS) $(TARGET_SOA)

all: 
	$(MAKE) $(MAKEFLAGS) clean
	$(MAKE) $(MAKEFLAGS) $(TARGET_AOS)
	$(MAKE) $(MAKEFLAGS) clean
	$(MAKE) $(MAKEFLAGS) $(TARGET_SOA)

$(TARGET_AOS): DEFS += -DARRAY_OF_STRUCTS

$(TARGET_AOS): clean $(OBJECTS)
	$(CC) -o $@ $(CFLAGS) $(OBJECTS) $(LIBS)

$(TARGET_SOA): clean $(OBJECTS)
	$(CC) -o $@ $(CFLAGS) $(OBJECTS) $(LIBS)

clean:
	@/bin/rm -f $(OBJECTS) core

realclean: clean
	@/bin/rm -f $(TARGETS)


# DO NOT DELETE

init_data.o: data.h init_data.h
main.o: data.h init_data.h distance.h distcheck.h xtime.h
distance.o: data.h distance.h
distcheck.o: data.h distcheck.h
