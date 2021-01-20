TARGET = poisson_j
SRCS	 =  
OBJS	 = main.o func.o print.o

OPT	= -g
PIC	=
XOPTS = -Xptxas=-v
ARCH  = -arch=sm_80
OMP   = -fopenmp

CXX	= nvcc
CXXFLAGS= --compiler-options "$(OPT) $(PIC) $(OMP)" $(ARCH) $(XOPTS) 

CUDA_PATH ?= /appl/cuda/11.1
INCLUDES = -I$(CUDA_PATH)/include -I$(CUDA_PATH)/samples/common/inc

SOFLAGS =
XLIBS	= 

$(TARGET): $(OBJS)
	$(CXX) -o $@ $(CXXFLAGS) $(SOFLAGS) $(INCLUDES) $^ $(XLIBS) 

.SUFFIXES: .cu
.cu.o:
	$(CXX) -o $*.o -c $*.cu $(CXXFLAGS) $(SOFLAGS) $(INCLUDES)

clean:
	@/bin/rm -f $(TARGET) $(OBJS) 
