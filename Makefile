CC      = gcc
CFLAGS  = -O3 -march=native -std=c11 -Wall -Wextra \
          -Iinclude \
          -ffast-math \
          -funroll-loops \
          -fomit-frame-pointer
LDFLAGS = -lm

SRC = src/tensor.c   \
      src/quant.c    \
      src/kvcache.c  \
      src/ops.c      \
      src/scheduler.c \
      src/engine.c   \
      src/main.c

OBJ = $(SRC:.c=.o)
BIN = llm_inference

.PHONY: all clean debug asan

all: $(BIN)

$(BIN): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

# Debug build — no optimizations, AddressSanitizer
debug:
	$(CC) -g -O0 -fsanitize=address -std=c11 -Wall -Iinclude \
	      $(SRC) -o $(BIN)_debug $(LDFLAGS)

# Profile build
profile:
	$(CC) -O2 -pg -std=c11 -Wall -Iinclude \
	      $(SRC) -o $(BIN)_prof $(LDFLAGS)

clean:
	rm -f $(OBJ) $(BIN) $(BIN)_debug $(BIN)_prof
