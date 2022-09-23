CC = /usr/local/cuda/bin/nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
SOURSES = lab1.cu
BIN = lab1

all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)