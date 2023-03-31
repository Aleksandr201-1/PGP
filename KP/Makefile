#CC = /usr/local/cuda/bin/nvcc
CC = nvcc
CFLAGS = --std=c++11 -Werror cross-execution-space-call -lm -O3
#SOURSES = kp.cu Vec3.cuh Vec3.cu Triangle.cuh Triangle.cu Scene.cuh Scene.cu FigureInit.cuh FigureInit.cu
SOURSES = kp.cu Vec3.cu Scene.cu FigureInit.cu
OBJ = $(SOURSES:.cu=.o)
BIN = kp

all: $(BIN)

$(BIN):
	$(CC) $(CFLAGS) -dc -c $(SOURSES)
	$(CC) $(CFLAGS) -o $(BIN) $(OBJ)

clean:
	rm -f *.o $(BIN)