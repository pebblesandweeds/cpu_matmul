# Matrix Multiplication on CPUs
==========================

This repository explores matrix multiplication on CPUs using Python/Numpy and C, with a focus on achieving high GFLOPS performance.

## Getting Started
---------------

### Prerequisites

* Python 3.x
* Numpy (install using `pip install -r python/requirements.txt`)

### Installation

1. Clone the repository.
2. Install the required Python packages using `pip install -r python/requirements.txt`.
3. Run the Python script using `python python/numpy_matmul.py`.

## Usage
-----

### Running the Python Script

The script outputs the elapsed time for matrix multiplication, total FLOP, and GFLOPS performance.

## Project Structure
-----------------

* `/`: Root directory
* `c/`: Future implementation of matrix multiplication in C (coming soon).
* `python/`: Python script for matrix multiplication using Numpy.
    + `numpy_matmul.py`: The Python script.
    + `requirements.txt`: List of required Python packages.
* `docs/`: Documentation (coming soon).
* `README.md`: This file.
* `LICENSE`: License information.

## Performance Metrics
--------------------

The following performance metrics were achieved using the simple Python/Numpy script provided in this repository. Note that these results are preliminary and will be improved upon with the upcoming C implementation.

* Apple M2: ~1000+ GFLOPS @ 32-bit precision (matrix size: 16384x16384)
* 2x AMD EPYC 7713 64-Core (128 cores total): ~3000+ GFLOPS @ 32-bit precision (matrix size: 16384x16384)

## License
-------

This project is released under the MIT License.

Note: This project is still in development, and the C implementation is not yet available. Stay tuned for updates!
