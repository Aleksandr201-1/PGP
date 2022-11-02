#CC = /usr/local/cuda/bin/nvcc
CC = nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
SOURSES = lab4.cu Matrix.cu
OBJ = $(SOURSES:.cu=.o)
BIN = lab4

all: $(BIN)

$(BIN):
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)

clean:
	rm -f *.o $(BIN)