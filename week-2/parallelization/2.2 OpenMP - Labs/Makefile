
#!/bin/sh
#BSUB -J Lab2_1
#BSUB -o Lab2_1_%J.out
#BSUB -q hpcintro
#BSUB -W 2
#BSUB -R "rusage[mem=512MB]"

# makefile for the hpc course
CC      = suncc
OPT     = -g -fast -xrestrict -xopenmp -xloopinfo
WARN    = -Wall
CFLAGS  = $(OPT) $(WARN) # the compiler flags
OBJECTS = main.o pi_func.o

# define project
program : $(OBJECTS)
	$(CC) $(CFLAGS) -o program $(OBJECTS)

clean : 
	@rm -f *.o compiler

realclean : clean
	@rm -f program

# define file dependencies
func.o : func.c pi_func.h
main.o : main.c pi_func.h


