# Compiler
CC = gcc

# Compiler flags
CFLAGS = -fopenmp -O2 -march=native -I./include

# Executable name
EXEC = matmul

# Source files
SRC_DIR = src

# Object directory
OBJ_DIR = obj

# Source files
SRC = $(SRC_DIR)/main.c $(SRC_DIR)/matmul_lib.c $(SRC_DIR)/time_utils.c $(SRC_DIR)/check_utils.c

# Object files
OBJ = $(patsubst $(SRC_DIR)/%.c,$(OBJ_DIR)/%.o,$(SRC))

# Default target
all: $(EXEC)

# Rule to build the executable
$(EXEC): $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^

# Rule to build object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(OBJ_DIR)
	$(CC) $(CFLAGS) -c -o $@ $<

# Clean rule
clean:
	rm -rf $(OBJ_DIR) $(EXEC)

# Run the executable
run: $(EXEC)
	./$(EXEC)

.PHONY: all clean run
