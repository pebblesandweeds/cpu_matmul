.. _matrix-multiplication:

CPU Matrix Multiplication: A Deep Dive
======================================

.. admonition:: Overview

   This blog post delves into the importance of matrix multiplication in deep learning. We explore its role as a foundational element in neural networks, why starting with CPU implementations is beneficial for learning, and how we can optimize matrix multiplication in C. We aim to match the performance of Python's NumPy by implementing efficient algorithms, preparing you to transition these skills to GPU optimizations in future projects.

Introduction
------------

Matrix multiplication is a fundamental operation that serves as the backbone of many deep learning models. Whether you're building neural networks, large language models (LLMs), or transformer architectures, mastering matrix multiplication is essential for success in these domains.

Why is Matrix Multiplication Important?
---------------------------------------

Matrix multiplication is used extensively in the operations that underlie deep learning, such as the dot products in neural network layers. Without a solid grasp of matrix multiplication, it becomes challenging to understand and optimize these complex models.

Most matrix operations in deep learning are performed on GPUs for efficiency, but it's crucial to understand the basics by starting with CPUs. Once you've mastered matrix multiplication on CPUs, you can transition to optimizing it on GPUs.

Why Start with CPUs?
--------------------

While GPUs are the go-to for high-performance matrix operations, learning how to implement these operations on CPUs is a great way to build foundational knowledge. By starting on a CPU, you can understand the mechanics of matrix multiplication and optimization techniques at a granular level.

The benchmark for optimal performance on a CPU is often set by Python's NumPy library. Our goal is to develop an understanding of matrix multiplication by implementing it in C and striving to reach the performance levels of NumPy.

Benchmarking with Large Matrices
--------------------------------

To effectively benchmark and compare performance, we use a large matrix size of N = 8192. Using a large N helps in observing consistent performance characteristics and avoiding anomalies that can occur with smaller matrices. Small matrices may fit entirely within the CPU cache, leading to performance variations that don't scale to larger, real-world applications.

Additionally, using a power of 2 like 8192 aligns with optimal memory access patterns on many systems.

Matrix Multiplication Flow
==========================

The following diagram illustrates how matrix \( A \) is multiplied by matrix \( B \) to form matrix \( C \) using the naive matrix multiplication approach.

.. plantuml::

    @startuml

    package "Matrix A" {
        [A11]
        [A12]
        [A1N]
        [AN1]
        [ANN]
    }

    package "Matrix B" {
        [B11]
        [B12]
        [B1N]
        [BN1]
        [BNN]
    }

    package "Matrix C" {
        [C11]
        [C12]
        [C1N]
        [CN1]
        [CNN]
    }

    @enduml

Explanation
-----------

In this diagram, the rows of matrix \( A \) are multiplied by the columns of matrix \( B \), resulting in the elements of matrix \( C \). Each element of matrix \( C \) is the dot product of a row from matrix \( A \) and a column from matrix \( B \).


Naive Matrix Multiplication in C
--------------------------------

Our first implementation is a naive matrix multiplication approach, which is straightforward but not optimized for performance. The code below demonstrates this basic method:

.. code-block:: c

   #include "../include/matmul_lib.h"
   #include <stdlib.h>
   #include <omp.h>
   #include <math.h>
   #include <immintrin.h>
   #include <stdio.h>

   void matmul(float A[N][N], float B[N][N], float C[N][N]) {
       #pragma omp parallel for collapse(2)
       for (int i = 0; i < N; i++) {
           for (int j = 0; j < N; j++) {
               for (int k = 0; k < N; k++) {
                   C[i][j] += A[i][k] * B[k][j];
               }
           }
       }
   }

This method multiplies matrices A and B to produce matrix C using three nested loops, which is simple but not efficient for large matrices.

Optimizing Matrix Multiplication
--------------------------------

To improve performance, we employ techniques such as tiling, blocking, and vectorization. These techniques help make better use of the CPU cache and parallel processing capabilities.

Tiling and Blocking
~~~~~~~~~~~~~~~~~~~

Tiling and blocking break down the matrices into smaller submatrices (tiles) and process them to reduce cache misses and improve data locality. Here's how we apply these techniques in our optimized matrix multiplication function:

.. code-block:: c

   void matmul_scalar(float A[N][N], float B[N][N], float C[N][N]) {
       #pragma omp parallel for collapse(3)
       for (int i = 0; i < N; i += BLOCK_SIZE) {
           for (int j = 0; j < N; j += BLOCK_SIZE) {
               for (int k = 0; k < N; k += BLOCK_SIZE) {
                   // Further tile within blocks
                   for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ii += TILE_SIZE) {
                       for (int jj = j; jj < j + BLOCK_SIZE && jj < N; jj += TILE_SIZE) {
                           for (int kk = k; kk < k + BLOCK_SIZE && kk < N; kk += UNROLL_FACTOR) {
                               float c_temp = C[ii][jj];
                               for (int iii = ii; iii < ii + TILE_SIZE && iii < i + BLOCK_SIZE && iii < N; iii++) {
                                   for (int jjj = jj; jjj < jj + TILE_SIZE && jjj < j + BLOCK_SIZE && jjj < N; jjj++) {
                                       c_temp += A[iii][kk] * B[kk][jjj];
                                   }
                                   C[iii][jjj] = c_temp;
                               }
                           }
                       }
                   }
               }
           }
       }
   }

Scalar vs. Vectorized Matrix Multiplication
-------------------------------------------

**Scalar** operations process one data element at a time, while **vectorized** operations process multiple data elements simultaneously using SIMD (Single Instruction, Multiple Data) instructions. Vectorization can significantly enhance performance by utilizing the CPU's full capabilities.

Our vectorized implementation leverages AVX instructions for efficient computation:

.. code-block:: c

   void matmul_vectorized(float A[N][N], float B[N][N], float C[N][N]) {
       float (*B_col)[N] = aligned_alloc(32, N * N * sizeof(float));
       if (B_col == NULL) {
           fprintf(stderr, "Memory allocation failed\n");
           exit(1);
       }
       #pragma omp parallel for collapse(2)
       for (int j = 0; j < N; j += 32) {
           for (int k = 0; k < N; k++) {
               for (int jj = 0; jj < 32 && j + jj < N; jj++) {
                   B_col[j+jj][k] = B[k][j+jj];
               }
           }
       }
       #pragma omp parallel
       {
           #pragma omp for
           for (int j = 0; j < N; j += 32) {
               for (int i = 0; i < N; i += 32) {
                   __m256 c[32][32];
                   for (int ii = 0; ii < 32; ii++) {
                       for (int jj = 0; jj < 32; jj++) {
                           c[ii][jj] = _mm256_setzero_ps();
                       }
                   }
                   for (int k = 0; k < N; k += 32) {
                       if (k + 128 < N) {
                           for (int ii = 0; ii < 32; ii++) {
                               _mm_prefetch((char*)&A[i+ii][k + 128], _MM_HINT_T1);
                               _mm_prefetch((char*)&B_col[j+ii][k + 128], _MM_HINT_T1);
                           }
                       }
                       __m256 a[32][4], b[32][4];
                       for (int ii = 0; ii < 32; ii++) {
                           for (int kk = 0; kk < 4; kk++) {
                               a[ii][kk] = _mm256_loadu_ps(&A[i+ii][k+kk*8]);
                               b[ii][kk] = _mm256_load_ps(&B_col[j+ii][k+kk*8]);
                           }
                       }
                       for (int ii = 0; ii < 32; ii++) {
                           for (int jj = 0; jj < 32; jj++) {
                               c[ii][jj] = _mm256_fmadd_ps(a[ii][0], b[jj][0], c[ii][jj]);
                               c[ii][jj] = _mm256_fmadd_ps(a[ii][1], b[jj][1], c[ii][jj]);
                               c[ii][jj] = _mm256_fmadd_ps(a[ii][2], b[jj][2], c[ii][jj]);
                               c[ii][jj] = _mm256_fmadd_ps(a[ii][3], b[jj][3], c[ii][jj]);
                           }
                       }
                   }
                   for (int ii = 0; ii < 32 && i + ii < N; ii++) {
                       for (int jj = 0; jj < 32 && j + jj < N; jj++) {
                           __m256 sum = c[ii][jj];
                           __m128 sum_high = _mm256_extractf128_ps(sum, 1);
                           __m128 sum_low = _mm256_castps256_ps128(sum);
                           __m128 sum_all = _mm_add_ps(sum_high, sum_low);
                           sum_all = _mm_hadd_ps(sum_all, sum_all);
                           sum_all = _mm_hadd_ps(sum_all, sum_all);
                           float result = _mm_cvtss_f32(sum_all);
                           C[i+ii][j+jj] += result;
                       }

Conclusion
----------

This post explored the implementation of matrix multiplication in C. In future posts, we’ll dive deeper into optimizations and applications.

References
----------

- `Matrix Multiplication on Wikipedia <https://en.wikipedia.org/wiki/Matrix_multiplication>`_
- `Linear Algebra Essentials <https://www.khanacademy.org/math/linear-algebra>`_
