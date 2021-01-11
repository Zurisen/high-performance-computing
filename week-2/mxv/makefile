# Makefile for Week2 exercise 2
CC = suncc
OPT = -g -fast -xrestrict -xautopar -xloopinfo
WARN = -Wall
CFLAGS = $(OPT) $(WARN)
OBJECTS = main.c mxv.o

#define project
program : $(OBJECTS)
	$(CC) $(CFLAGS) -o mxv_run $(OBJECTS)

clean :
	@rm -f *.o compiler

realclean : clean
	@rm -f mxv_run

# file dependencies
mvx.o : mxv.c mxv.h
main.o : main.c mxv.h 
