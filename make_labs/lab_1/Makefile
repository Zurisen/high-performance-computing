CC = gcc
WARN = -Wall
OPT = -g -O3
CFLAGS = $(OPT) $(WARN)
OBJECTS = main.o data.o io.o

exer1: $(OBJECTS)
	$(CC) $(CFLAGS) -o exer1 $(OBJECTS)

clean: 
	@rm -f *.o 
realclean:
	@rm -f exer1

# dependencies
io.o: io.h io.c
data.o: data.h data.c
main.o: main.c data.h io.h
