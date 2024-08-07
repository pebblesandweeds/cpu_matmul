.. _matrix-multiplication:

CPU Matrix Multiplication: A Deep Dive
======================================

.. admonition:: Overview

 Matrix multiplication is fundamental in deep learning, enabling complex computations in neural networks. This blog post explores matrix multiplication using C to achieve performance comparable to Python's NumPy, delving into detailed optimizations that leverage C's low-level capabilities.

Introduction
------------

Matrix multiplication is at the heart of modern computational models, serving as a critical component in deep learning systems. It is integral to neural networks, where it drives activations and the flow of information through network layers, enabling these models to both "learn" and make predictions.

In machine learning, especially within neural networks and transformer architectures crucial for natural language processing, matrix multiplication is fundamental. It allows for efficient data handling and transformation, which are essential for training and deploying sophisticated models.

Using a low-level language like C to implement matrix multiplication from scratch can offer valuable insights into its operational mechanics. This approach helps in understanding and optimizing performance at a granular level, which is often obscured by high-level libraries like NumPy.

This blog is dedicated to discussing matrix multiplication implemented in C, emphasizing the development of efficient algorithms and optimization techniques that enhance performance and understanding beyond typical high-level implementations.


Why is Matrix Multiplication Important?
---------------------------------------

Matrix multiplication is crucial in training and inference for a wide range of machine learning models, including those in natural language processing, computer vision, and audio processing. It is the core operation for transforming input data, computing activations, and updating model parameters. In transformer architectures, it powers self-attention mechanisms and feedforward neural networks, enabling models to analyze and generate complex data.

Models like GPT-2 and GPT-3 rely heavily on matrix multiplication within their transformer architectures, which consist of multiple layers, attention heads, and parallel processing capabilities. As model sizes grow, the number of matrix multiplications required to compute attention scores, transform data, and update weights increases significantly, emphasizing the need for efficient matrix operations to manage computational demands.

Matrix multiplication is a critical component in the training process, with millions of operations performed across batches and epochs. This accounts for a large portion of the total computational cost. During each training step, both forward passes (calculating predictions) and backward passes (computing gradients) rely on matrix multiplication to propagate information through the network. Optimizing these operations is essential for reducing training time and resource consumption, making it a key focus area for improving the efficiency of large-scale models.

These factors underscore the importance of matrix multiplication even in older models like GPT-2 and GPT-3, where it supports efficient language processing and generation. Understanding and optimizing matrix multiplication is vital for handling the extensive computations required by these and other machine learning models.

Why Start with CPUs? Why Use C?
-------------------------------

Although GPUs are preferred for high-performance matrix operations, we start here with CPUs to establish a strong foundation in matrix multiplication and optimization techniques. Implementing matrix operations on CPUs first helps us understand the mechanics and challenges of performance optimization.

We choose C over higher-level languages for several reasons. C provides direct access to hardware resources, enabling granular performance optimization. It also allows us to better understand the memory hierarchy and optimize access patterns, which is crucial for high-performance matrix operations. By comparing our C implementation to Python’s NumPy, we can benchmark performance and measure progress, equipping us with essential skills for tackling more advanced topics in matrix multiplication and optimization.

Matrix Configuration and Benchmarking 
-------------------------------------

To establish a solid foundation for our matrix multiplication implementation, we have made several key decisions regarding matrix configuration and benchmarking:

*Matrix Configuration*
^^^^^^^^^^^^^^^^^^^^^^

Our implementation uses square matrices (N x N) for both matrices A and B. This choice simplifies the implementation by focusing on the core matrix multiplication algorithm. While common, it also allows for future extensions to support non-square matrices.

We have set N to a static size of 8192. Defining N as a constant enables more aggressive compiler optimizations, and in C, we define N as a `const` to ensure its value remains unchanged at runtime. This static size helps standardize our approach and aligns with optimal memory access patterns, making it ideal for performance benchmarking.

*Benchmarking Strategy*
^^^^^^^^^^^^^^^^^^^^^^^

We use a large matrix size of N = 8192 to ensure that our benchmarking reflects realistic performance characteristics. This size is chosen to avoid anomalies that occur when smaller matrices fit entirely within the CPU cache, thus giving a clearer picture of performance under typical conditions. With N set to 8192 and using float32 data types, the memory requirement for a single matrix is approximately 268 MB, resulting in a total of around 804 MB for three matrices (A, B, and C).

For performance testing, we utilize an AWS c7a.32xlarge instance, which provides the necessary computational resources to handle large matrix operations. This instance features an AMD EPYC 9R14 processor with 2 sockets, 64 cores per socket, and a total of 128 cores, all without simultaneous multithreading (SMT). While this is a very large instance, its capacity is essential for managing the high demands of our large N value and obtaining accurate performance metrics.


Naive Matrix Multiplication 
---------------------------

To begin our exploration, we start with a naive matrix multiplication approach using C, which is visualized and detailed through both a mathematical formula and a straightforward implementation. This initial method, while simple, serves as a foundation for understanding the inefficiencies that come with straightforward algorithmic approaches.

*Visual and Formulaic Representation*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The process is visually demonstrated in the following animation, which shows an 8x8 matrix multiplication. Each frame captures the computation of the elements in matrix :math:`C` as the sum of products of corresponding elements in matrices :math:`A` and :math:`B`.

.. image:: /_static/matrix_multiplication_8x8_precise_loop.gif
   :alt: 8x8 Matrix Multiplication Animation
   :align: center

The corresponding mathematical operation is succinctly described by the formula:

.. math::
    C_{ij} = \sum_{k=1}^{N} A_{ik} B_{kj}

*Naive Implementation in C*
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Following this formula, our C code implementation employs three nested loops to perform the matrix multiplication. This basic method is straightforward but not optimized for performance, particularly with large matrices where the computational overhead becomes significant.

.. code-block:: c

   void matmul(float A[N][N], float B[N][N], float C[N][N]) {
       for (int i = 0; i < N; i++) {
           for (int j = 0; j < N; j++) {
               for (int k = 0; k < N; k++) {
                   C[i][j] += A[i][k] * B[k][j];
               }
           }
       }
   }

*Naive Matrix Multiplication Performance* 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This naive approach effectively illustrates the link between algorithmic simplicity and computational inefficiency. With N set to 8192, the computation involves approximately 1,099.51 billion floating-point operations. Despite the large workload, our AWS c7a.32xlarge instance achieves a performance of around 25 GFLOPS. This demonstrates the significant gap between the naive method's potential and the optimizations needed to harness the full computational power of our hardware. This setup provides a clear starting point for exploring more advanced optimization techniques in subsequent sections.

Optimizing Matrix Multiplication
--------------------------------

To improve performance, we employ techniques such as tiling, blocking, and vectorization. These techniques help make better use of the CPU cache and parallel processing capabilities.

Tiling and Blocking
^^^^^^^^^^^^^^^^^^^

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
