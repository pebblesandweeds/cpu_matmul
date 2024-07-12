# Compiler
CC = gcc

# Compiler flags
CFLAGS = -fopenmp -O2 -march=native

# Executable name
EXEC = matmul

# Source files
SRC = c_matmul.c

# Default target
all: $(EXEC)

# Rule to build the executable
$(EXEC): $(SRC)
	$(CC) $(CFLAGS) -o $@ $^

# Clean rule
clean:
	rm -f $(EXEC)

# Run the executable
run: $(EXEC)
	./$(EXEC)

.PHONY: all clean run
