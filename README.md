# Matrix Multiplication on CPUs
==========================

This repository explores matrix multiplication on CPUs using Python/Numpy and C, with a focus on trying to implement C code from scratch to match Python/Numpy GFLOPS performance.   

## Getting Started
---------------

### Prerequisites

* Python 3.x
* Numpy (install using `pip install -r python/requirements.txt`)
* GCC compiler
    - On Linux: `gcc`
    - Initial naive C implementation will work on Apple MacOS: `gcc-14` (install with Homebrew)

### Installation

1. Clone the repository.
2. Install the required Python packages using `pip install -r python/requirements.txt`.

## Usage
-----

### Running the Python Script

Run the Python script using `python python/numpy_matmul.py` (assuming you are in the root of the repo)

### Running the C Code

Compile the code with the Makefile using `make -C c` (assuming you are in the root of the repo)
Run the code using `./c/matmul` (assuming you are in the root of the repo)

### Common Performance Output

Both the Python and C scripts output the elapsed time for matrix multiplication, total FLOP, and GFLOPS performance.

## Project Structure
-----------------

* `/`: Root directory
* `c/`: Future implementation of matrix multiplication in C.
    + `c_matmul.c`: The C code for matrix multiplication.
    + `Makefile`: Makefile for compiling the C code.
* `python/`: Python script for matrix multiplication using Numpy.
    + `numpy_matmul.py`: The Python script.
    + `requirements.txt`: List of required Python packages.
* `docs/`: Documentation (coming soon).
* `README.md`: This file.
* `LICENSE`: License information.

## Performance Metrics
--------------------

The following performance metrics were achieved using the simple Python/Numpy script provided in this repository. Note that these results are preliminary and will be improved upon with the upcoming C implementation.

### Python/Numpy Performance Metrics
* Apple M2: ~1000+ GFLOPS @ 32-bit precision (matrix size: 16384x16384)
* 2x AMD EPYC 7713 64-Core (128 cores total): ~3000+ GFLOPS @ 32-bit precision (matrix size: 16384x16384)

### Naive C Implementation Performance Metrics
* Apple M2: ~1 GFLOPS
* 2x AMD EPYC 7713 64-Core (128 cores total): ~15+ GFLOPS

## License
-------

This project is released under the MIT License.

Note: This project is still in development, and the C implementation is not yet complete. Stay tuned for updates!
