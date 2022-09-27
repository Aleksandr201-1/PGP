#CC = /usr/local/cuda/bin/nvcc
CC = nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
SOURSES = lab3.cu ImageReader.cu
BIN = lab3

all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)