CC = gcc
WARN = -Wall
OPT = -g -O3
CFLAGS = $(OPT) $(WARN)
OBJECTS = main.o data.o io.o

exer2: $(OBJECTS)
	$(CC) $(CFLAGS) -o exer1 $(OBJECTS)

clean: 
	@rm -f *.o 
realclean:
	@rm -f exer2

# dependencies
io.o: io.h
data.o: data.h data.c
main.o: main.c data.h io.h
