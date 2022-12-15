CC = /usr/local/bin/mpic++
#CC = mpic++
CFLAGS = --std=c++11 -Werror -lm
SOURSES = lab7.cpp DirihleTask.cpp
OBJ = $(SOURSES:.cpp=.o)
BIN = lab7

all: $(BIN)

$(BIN):
	$(CC) $(CFLAGS) -o $(BIN) $(SOURSES)

clean:
	rm -f *.o $(BIN)