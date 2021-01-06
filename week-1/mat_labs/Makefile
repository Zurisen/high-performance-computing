SRCS	= main_mulblas.c datatools.c lib.c
OBJS	= $(SRCS:.c=.o)
TARGET	= madd.$(CC)

CC	= gcc
INCS	= 
DBG	= -g
OPT	= -O3
PARA	= 
CFLAGS	= $(DBG) $(OPT) $(PARA) $(XOPTS)

LIBS	= -L /usr/lib64/atlas-lsatlas

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $(OBJS) $(LIBS)

clean:
	@/bin/rm -f $(OBJS) *~ core*

realclean: clean
	@/bin/rm -f $(TARGET)
