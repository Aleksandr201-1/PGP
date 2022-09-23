# #CC = /usr/local/cuda/bin/nvcc
# CC = nvcc
# CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm

# SOURSES = lab2.cu ImageReader.cu
# OBJ = $(SOURSES:.cpp=.o)
# BIN = lab2

# all: $(BIN)

# $(BIN): $(OBJ)
# 	$(CC) $(CFLAGS) $(OBJ) -o $(BIN)

# .cu.o:
# 	$(CC) $(CFLAGS) -c $< -o $@

# lab2.cu: ImageReader.cuh
# ImageReader.cu: ImageReader.cuh

# clean:
# 	rm -f *.o $(BIN)
CC = nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
SOURSES = lab2.cu ImageReader.cu
BIN = lab2

all:
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)