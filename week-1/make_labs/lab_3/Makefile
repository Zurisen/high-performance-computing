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
	@rm -f exer3

# DO NOT DELETE

data.o: /usr/include/stdio.h data.h
io.o: /usr/include/stdio.h io.h
main.o: /usr/include/stdio.h /usr/include/stdlib.h io.h data.h
