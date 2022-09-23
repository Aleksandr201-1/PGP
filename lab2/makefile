CC = /usr/local/cuda/bin/nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
SOURSES = lab2.cu ImageReader.cu
BIN = lab2

all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)