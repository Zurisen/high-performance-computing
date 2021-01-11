CC = suncc
OPT = -g -fast -xrestrict -xopenmp -xloopinfo
WARN = -Wall
CFLAGS = $(OPT) $(WARN)
OBJECTS = main.o func.o

program : $(OBJECTS)
	$(CC) $(CFLAGS) -o paralelizer $(OBJECTS)

clean : 
	@rm -f *.o compiler

realclean :
	@rm -f paralelizer

# dependencies
func.o : func.c func.h
main.o : main.c func.h
