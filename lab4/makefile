#CC = /usr/local/cuda/bin/nvcc
CC = nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm --expt-extended-lambda
SOURSES = lab4.cu Matrix.cu
OBJ = $(SOURSES:.cu=.o)
BIN = lab4

all: $(BIN)

$(BIN):# $(OBJ)
#	$(CC) $(CFLAGS) -dc -c $(SOURSES)
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)

#$(OBJ):
#	$(CC) $(CFLAGS) -dc -c $(SOURSES)

clean:
	rm -f *.o $(BIN)