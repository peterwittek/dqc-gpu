ATLAS_PATH = /usr
MAGMA_PATH = ../magma_1.1.0
CUDA_PATH = /usr/local/cuda

CC = gcc
CCFLAGS = -g -Wall
OPTS = -O3 -DADD_ -fopenmp
NVCC=$(CUDA_PATH)/bin/nvcc
NVCCFLAGS=-I $(CUDA_PATH)/include -I$(MAGMA_PATH)/include -g --compiler-options -rdynamic

INCLUDE=-I$(CUDA_PATH)/include -I$(MAGMA_PATH)/include
LIBS=-L$(ATLAS_PATH)/lib -L$(CUDA_PATH)/lib64 -L$(MAGMA_PATH)/lib \
     -lmagmablas -lmagma -lcuda \
     $(ATLAS_PATH)/lib/liblapack.a -lcblas -lblas -latlas -lcublas -lm

dqc: obj/dqc.o obj/common.o obj/cpuRoutines.o obj/gpuRoutines.cu.o obj/gpuRoutines.o
	$(CC) $(OPTS) obj/dqc.o obj/common.o obj/cpuRoutines.o obj/gpuRoutines.o obj/gpuRoutines.cu.o $(LIBS) -o dqc
	
obj/dqc.o:
	mkdir -p obj
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDE) -c dqc.cpp -o obj/dqc.o

obj/common.o:
	$(CC) $(OPTS) $(CCFLAGS) $(INCLUDE) -c common.cpp -o obj/common.o

obj/cpuRoutines.o:
	$(CC) $(OPTS) $(CCFLAGS)  $(INCLUDE) -c cpuRoutines.cpp -o obj/cpuRoutines.o

obj/gpuRoutines.cu.o: 
	$(NVCC) $(NVCCFLAGS) gpuRoutines.cu -c -o obj/gpuRoutines.cu.o

obj/gpuRoutines.o:
	$(CC) $(OPTS) $(CCFLAGS)  $(INCLUDE) -c gpuRoutines.cpp -o obj/gpuRoutines.o

clean:
	rm -rf obj/*o
