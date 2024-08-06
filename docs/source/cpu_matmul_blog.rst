.. _matrix-multiplication:

CPU Matrix Multiplication: A Deep Dive
======================================

.. admonition:: Overview

 Matrix multiplication is a critical operation that serves as the backbone for deep learning and neural networks, enabling complex computations and transformations. This blog post demonstrates how to implement matrix multiplication from scratch using the C programming language, aiming to achieve performance comparable to Python's NumPy. By leveraging C's low-level capabilities, we can perform detailed performance optimizations that are not typically possible in higher-level languages. 

Introduction
------------

Matrix multiplication is a fundamental operation in fields like physics, chemistry, engineering, and computer science. It is crucial for solving systems of linear equations and applying transformations such as rotations, scaling, and translations, which are essential for manipulating data and models. These capabilities make matrix multiplication a vital tool across various scientific and technical disciplines.

In machine learning, matrix multiplication underlies key algorithms, including neural networks, where it is used to compute activations and propagate information through layers. Transformer architectures, which are pivotal in natural language processing, rely heavily on matrix multiplication to process and transform data efficiently.

While libraries like NumPy provide highly optimized implementations that abstract away the complexities of matrix multiplication, implementing these operations in a low-level language like C offers valuable insights. By developing our own algorithms, we can better understand the mechanics of matrix multiplication and explore optimization strategies that improve performance and efficiency.

This blog discusses implementing matrix multiplication from scratch in C, emphasizing efficient algorithms and optimization techniques. The process involves executing matrix multiplication at a low level and gaining insights into performance tuning and the principles behind optimized libraries like NumPy, demonstrating practical strategies for achieving high performance in matrix computations.

Why is Matrix Multiplication Important?
---------------------------------------

Matrix multiplication is central to the training and inference processes in a wide range of machine learning models, including those used in natural language processing, computer vision, and audio processing. It is the core computational operation used to transform input data, compute activations, and update model parameters. In transformer architectures, matrix multiplication is crucial for the self-attention mechanisms and feedforward neural networks that drive the models' ability to analyze and generate complex data.

Models like GPT-2 and GPT-3 rely heavily on matrix multiplication within their transformer architectures, which feature multiple layers, attention heads, and parallel processing capabilities. These models come in various configurations, with different numbers of parameters and layers, ranging from small to large (e.g., GPT-2: 1.5 billion parameters, GPT-3: up to 175 billion parameters). As model size increases, so does the number of matrix multiplications required to compute attention scores, transform data, and update weights across layers. This scalability enables complex tasks, but also underscores the need for efficient matrix operations to manage the significant computational demands of large-scale models.

Matrix multiplication plays a critical role in the training process of these models. With millions of operations performed across batches and epochs, the computational cost of these operations represents a significant portion of the total training cost. Each training step involves both forward passes, where predictions are calculated, and backward passes, where gradients are computed, with matrix multiplications playing a key role in propagating information through the network. Optimizing these operations is essential for reducing training time and resource consumption, making it a key area of focus for improving the efficiency of large-scale machine learning models.

These factors highlight why matrix multiplication is vital, even in older models like GPT-2 and GPT-3, underpinning their capacity to process and generate language efficiently. Understanding and optimizing matrix multiplication is essential for handling the extensive computations required by these and other machine learning models.

Why Start with CPUs?
--------------------

While GPUs are the go-to for high-performance matrix operations, learning how to implement these operations on CPUs is a great way to build foundational knowledge. By starting on a CPU, you can understand the mechanics of matrix multiplication and optimization techniques at a granular level.

The benchmark for optimal performance on a CPU is often set by Python's NumPy library. Our goal is to develop an understanding of matrix multiplication by implementing it in C and striving to reach the performance levels of NumPy.

Benchmarking with Large Matrices
--------------------------------

To effectively benchmark and compare performance, we use a large matrix size of N = 8192. Using a large N helps in observing consistent performance characteristics and avoiding anomalies that can occur with smaller matrices. Small matrices may fit entirely within the CPU cache, leading to performance variations that don't scale to larger, real-world applications.

Additionally, using a power of 2 like 8192 aligns with optimal memory access patterns on many systems.

Matrix Multiplication Flow
--------------------------

The following diagram illustrates how matrix :math:`A` is multiplied by matrix :math:`B` to form matrix :math:`C` using the naive matrix multiplication approach.

.. image:: /_static/matrix_multiplication_8x8_precise_loop.gif
   :alt: 8x8 Matrix Multiplication Animation
   :align: center

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
