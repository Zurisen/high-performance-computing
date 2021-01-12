CC      	:= gcc
C_FLAGS 	:= -Wall -Wextra -g -Ofast -fopenmp
EXECUTABLE	:= Poisson.$(CC)

INCLUDE 	:= include
LIB     	:= lib

LIBRARIES	:=


all: $(EXECUTABLE)

run: clean all
	./$(EXECUTABLE)

$(EXECUTABLE): *.c
	$(CC) $(C_FLAGS) -I$(INCLUDE) -L$(LIB) $^  -o $@ $(LIBRARIES)

clean:
	-rm -r $(EXECUTABLE)
