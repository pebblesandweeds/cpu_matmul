# Matrix Multiplication on CPUs

This repository explores matrix multiplication on CPUs using Python/Numpy and C, with a focus on trying to implement C code from scratch to match Python/Numpy GFLOPS performance.   

## Getting Started

### Prerequisites

* Python 3.x
* Numpy (install using `pip install -r python/requirements.txt`)
* GCC compiler
    - On Linux: `gcc`
    - Naive C and scalar implementation works on Apple MacOS: `gcc-14` (install with Homebrew)
    - Vectorized code uses immintrin.h so will not work on Apple silicon, would require a new function written with NEON

### Installation

1. Clone the repository.
2. Change your directory `cd cpu_matmul`
3. Install the required Python packages using `pip install -r python/requirements.txt`.

## Usage

### Running the Python Script

Run the Python script using `python python/numpy_matmul.py` 

### Running the C Code

1. Change directories `cd c`
2. Compile and run the benchmark `make && ./matmul` 

### Common Performance Output

Both the Python and C scripts output the elapsed time for matrix multiplication, total FLOP, and GFLOPS performance.

## Project Structure

* `/`: Root directory
* `c/`: Future implementation of matrix multiplication in C.
    + `Makefile`: Makefile for compiling the C code.
    + `src/`: Source files for C implementation.
        - `main.c`: The main program to run matrix multiplication.
        - `matmul_lib.c`: Library containing naive and optimized blocked matrix multiplication functions.
        - `time_utils.c`: Utility library for timing functions.
        - `check_utils.c`: Library for spot-checking matrix multiplication results.
    + `include/`: Header files for C implementation.
        - `matmul_lib.h`: Header for matrix multiplication functions.
        - `time_utils.h`: Header for timing utility functions.
        - `check_utils.h`: Header for spot-checking matrix multiplication results.
* `python/`: Python script for matrix multiplication using Numpy.
    + `numpy_matmul.py`: The Python script.
    + `requirements.txt`: List of required Python packages.
* `docs/`: Documentation (coming soon).
* `README.md`: This file.
* `LICENSE`: License information.

## Performance Metrics

The following performance metrics were achieved using the simple Python/Numpy script and C source code provided in this repository. Note that these results are a work in progress and will be improved with each implementation.

### Python/Numpy Performance Metrics
* AMD EPYC 9R14: 2 sockets, 64 cores per socket, 128 cores total, no SMT (AWS c7a.32xlarge): ~3500+ GFLOPS @ 32-bit precision

### Naive C Implementation Performance Metrics
* AMD EPYC 9R14: 2 sockets, 64 cores per socket, 128 cores total, no SMT (AWS c7a.32xlarge): ~25+ GFLOPS

### Optimized scalar C Implementation Performance Metrics
* AMD EPYC 9R14: 2 sockets, 64 cores per socket, 128 cores total, no SMT (AWS c7a.32xlarge): ~270+ GFLOPS

### Vectorized C Implementation Performance Metrics
* AMD EPYC 9R14: 2 sockets, 64 cores per socket, 128 cores total, no SMT (AWS c7a.32xlarge): ~2700+ GFLOPS

## License

This project is released under the MIT License.

Note: This project is still in development, and the C implementation is not yet complete. Stay tuned for updates!
