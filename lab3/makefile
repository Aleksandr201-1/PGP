#CC = /usr/local/cuda/bin/nvcc
CC = nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm
SOURSES = lab3.cu Vec3.cu Matrix.cu ImageReader.cu
OBJ = $(SOURSES:.cu=.o)
BIN = lab3

all: $(BIN)

$(BIN):
	$(CC) $(CFLAGS) -dc -c $(SOURSES)
	$(CC) $(CFLAGS) -o $(BIN) $(OBJ)

clean:
	rm -f *.o $(BIN)