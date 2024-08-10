.. _matrix-multiplication:

CPU Matrix Multiplication from Scratch in C
===========================================

.. admonition:: Highlights 

 Matrix multiplication is fundamental in deep learning, enabling most of the computations performed in neural networks. This blog post explores matrix multiplication using C to achieve performance comparable to Python's NumPy, exploring detailed optimizations that leverage C's low-level capabilities.

 We demonstrate the performance evolution on an `AWS c7a.32xlarge EC2 instance <https://aws.amazon.com/ec2/instance-types/c7a/>`_ through the following implementations using 32-bit precision:

 - **Baseline with Python/NumPy**: **~3500 GFLOPS**, using `this implementation <https://github.com/pebblesandweeds/cpu_matmul/blob/main/python/numpy_matmul.py>`_ for comparison with our C code.
 - **Naive C Approach**: **~25 GFLOPS**, starting with a simple `scalar implementation <https://github.com/pebblesandweeds/cpu_matmul/blob/main/c/src/matmul_lib.c#L28>`_.
 - **Optimized C Techniques**: **~500 GFLOPS**, optimizing the `naive scalar implementation <https://github.com/pebblesandweeds/cpu_matmul/blob/main/c/src/matmul_lib.c#L39>`_ using tiling, blocking, and loop unrolling.
 - **Vectorized C Operations**: **~3000 GFLOPS**, leveraging `vectorized SIMD instructions <https://github.com/pebblesandweeds/cpu_matmul/blob/main/c/src/matmul_lib.c#L64>`_ for enhanced performance.

 These results demonstrate the effectiveness of implementing matrix multiplication from scratch in C, achieving performance levels close to those of Python's NumPy.

 Get the code here -> `cpu_matmul repository <https://github.com/pebblesandweeds/cpu_matmul>`_.

Introduction
------------

Matrix multiplication is a core computational operation in machine learning, particularly in deep learning and neural networks. Forward and backward propagation in neural networks use matrix multiplication to compute activations and gradients millions of times during model training and inference, allowing models to recognize patterns and make predictions. 

This blog explores building C implementations of CPU matrix multiplication, focusing on efficient algorithms and low-level optimizations. By working with C, we gain insights into performance enhancements often hidden by high-level libraries, critical for scaling matrix multiplication calculations for transformer based models to handle large volumes of computations.  


Why is Matrix Multiplication Important?
---------------------------------------

Matrix multiplication is fundamental to neural network computations, particularly in forward and backward propagation. In forward propagation, it's used to combine input data with learned weights, producing activations that flow through the network. During backward propagation, matrix multiplication helps calculate gradients, enabling the network to update its weights and learn from errors. This operation's efficiency directly impacts the speed and scalability of neural network training and inference.

Matrix multiplication is crucial in various machine learning domains, including natural language processing, computer vision, and audio processing. It forms the core of transforming input data, computing activations, and updating model parameters. In transformer architectures, such as those used in GPT models, matrix multiplication powers self-attention mechanisms and feedforward networks, enabling the analysis and generation of complex data.

As model sizes grow, the computational demands increase significantly. Modern architectures like GPT-2 and GPT-3 perform millions of matrix multiplications across multiple layers, attention heads, and parallel processing units. This operation accounts for a large portion of the total computational cost during training and inference, emphasizing the need for efficient implementation.

Optimizing matrix multiplication is therefore critical for improving the efficiency of large-scale models. It directly affects training time, resource consumption, and the ability to scale to larger datasets and model sizes. Understanding and optimizing this operation is vital for advancing the capabilities of machine learning models while managing computational resources effectively.

Why Matrix Multiplication on CPUs and in C?
-------------------------------------------

While GPUs are preferred for high-performance matrix operations, we start here with CPUs to establish a strong foundation in matrix multiplication and optimization techniques. Implementing matrix operations on CPUs first helps us understand the mechanics and challenges of performance optimization that are applicable to both CPU and GPU implementations.

We choose C over higher-level languages for several reasons. C provides direct access to hardware resources, enabling granular performance optimization. It also allows us to better understand the memory hierarchy and optimize access patterns, which is crucial for high-performance matrix operations. By comparing our C implementation to Python’s NumPy, we can benchmark performance and measure progress, equipping us with the skills for tackling more advanced topics in matrix multiplication and optimization.

Benchmarking Setup and Code Organization
----------------------------------------

*Matrix Configuration and Benchmarking Strategy*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our implementation uses square matrices (N x N) for both matrices A and B, with N set to a static size of 8192. This choice simplifies the implementation while allowing for future extensions to non-square matrices. Defining N with a preprocessor C macro enables aggressive compiler optimizations and ensures consistent runtime behavior.

With N = 8192 and using float32 (4 bytes per element), each matrix contains 67,108,864 elements. The size of each matrix (A, B, and C) is calculated as follows:

67,108,864 * 4 bytes = 268,435,456 bytes ≈ 268 MB 

This results in a total memory requirement of approximately 805 MB for all three matrices.

We chose this large N value to reflect realistic performance characteristics and avoid cache-related anomalies. Modern high-end CPUs typically have L3 cache sizes ranging from 16MB to 512MB. Our chosen matrix size exceeds these cache capacities, ensuring that:

* The total working set is much larger than the cache, forcing memory accesses to main RAM.
* Cache evictions and reloads occur frequently during computation.

This setup provides a clear picture of performance under typical conditions, exercising the full memory hierarchy and highlighting the importance of efficient memory access patterns.

For benchmarking, we use an AWS c7a.32xlarge instance featuring an AMD EPYC 9R14 processor (2 sockets, 64 cores per socket, 128 cores total) without simultaneous multithreading. This instance has a 512MB L3 cache, which is smaller than our total working set size. This large matrix size and higher-end AWS EC2 instance help us obtain accurate performance metrics that reflect real-world scenarios for large-scale matrix multiplication.

*Code Structure and Organization*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code examples in this blog are primarily from our `matmul_lib.c <https://github.com/pebblesandweeds/cpu_matmul/blob/dev/c/src/matmul_lib.c>`_ file, which contains the core matrix multiplication functions. Our `main.c <https://github.com/pebblesandweeds/cpu_matmul/blob/dev/c/src/main.c>`_  file serves as the entry point, calling these functions to perform the matrix operations.

We've organized our code into separate modules for clarity and maintainability. For detailed information about our project structure, please refer to our `README.md <https://github.com/pebblesandweeds/cpu_matmul/blob/dev/README.md#project-structure>`_ file.

As we explore different optimization techniques, we'll focus on the relevant functions from our `matmul_lib.c` file, discussing how they implement different ways of performing matrix multiplication with the associated performance gains.  Note that the code snippets below ommit the `#pragma` propressor directoves in our code for simplicity, the repo contains parallel instructions that are out of scope for our conversations in this blog. 

Naive Matrix Multiplication 
---------------------------

To begin our exploration, we start with a naive matrix multiplication approach using C, which is visualized and detailed through both a mathematical formula and a straightforward implementation. This initial method, while simple, serves as a foundation for understanding the inefficiencies that come with straightforward algorithmic approaches.

*Visual and Formulaic Representation*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The process is visually demonstrated in the following animation, which shows an 8x8 matrix multiplication. Each frame captures the computation of the elements in matrix :math:`C` as the sum of products of corresponding elements in matrices :math:`A` and :math:`B`.

.. image:: /_static/matrix_multiplication_8x8_precise_loop.gif
   :alt: 8x8 Matrix Multiplication Animation
   :align: center

The corresponding mathematical operation is described by the formula:

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

This naive approach effectively illustrates the link between algorithmic simplicity and computational inefficiency. With N set to 8192, the computation involves approximately 1,099.51 billion floating-point operations. Despite the high-end CPU we have, our AWS c7a.32xlarge instance achieves a performance of **~25 GFLOPS**. This demonstrates the significant gap between the naive method's potential and the optimizations needed to harness the full computational power of our hardware. This setup provides a clear starting point for exploring more advanced optimization techniques in subsequent sections.

Optimizing Matrix Multiplication
--------------------------------

While the naive matrix multiplication implementation provides a clear understanding of the algorithm, it is not efficient for large matrices. The naive approach processes matrices row by row and column by column, which can lead to frequent cache misses and inefficient memory access patterns. This inefficiency arises because accessing matrix elements in this order does not align well with how data is cached in memory, resulting in slow performance.

To address these inefficiencies, we employ optimization techniques such as tiling, blocking, and loop unrolling. These techniques improve data locality and make better use of CPU caches, significantly enhancing performance. You can learn more about these techniques through the following links: `Tiling and Blocking <https://en.wikipedia.org/wiki/Loop_nest_optimization#Tiling>`_ and `Loop Unrolling <https://en.wikipedia.org/wiki/Loop_unrolling>`_.

*Optimized Implementation in C*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our optimized matrix multiplication implementation leverages these techniques to minimize cache misses and maximize computational throughput. The following C code demonstrates the use of blocking and tiling to improve performance:

.. code-block:: c

   #define BLOCK_SIZE 64
   #define TILE_SIZE 32
   #define UNROLL_FACTOR 4

   void matmul_scalar(float A[N][N], float B[N][N], float C[N][N]) {
    for (int i = 0; i < N; i += BLOCK_SIZE) {
    for (int j = 0; j < N; j += BLOCK_SIZE) {
    for (int k = 0; k < N; k += BLOCK_SIZE) {
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

*Optimized Matrix Multiplication Performance*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By optimizing matrix multiplication, we achieve a significant performance boost. On the AWS c7a.32xlarge instance, the optimized implementation achieves approximately **500 GFLOPS**, which represents more a 20x increase over the naive approach. This performance gain demonstrates the effectiveness of optimization techniques in harnessing the full computational power of modern hardware.

This exploration into optimized matrix multiplication illustrates how strategic algorithmic improvements can dramatically enhance performance, providing a solid foundation for further exploration and learning in high-performance computing.

Vectorized Matrix Multiplication
--------------------------------

*Scalar vs. Vectorized Operations*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scalar operations process data one element at a time, performing calculations sequentially. In contrast, vectorized operations use a Single Instruction, Multiple Data (SIMD) approach, processing multiple data elements simultaneously. This parallelism is implemented on CPUs through SIMD instructions, which leverage hardware capabilities to execute the same operation on multiple data points in a single instruction cycle.

To write vectorized code, several elements are necessary:

1. **SIMD Instructions**: Using SIMD instructions like AVX for parallel computation. Learn more about SIMD from `Wikipedia <https://en.wikipedia.org/wiki/SIMD>`_.

2. **Data Alignment**: Ensuring data is aligned in memory for efficient SIMD processing. Check out `Data Alignment <https://en.wikipedia.org/wiki/Data_structure_alignment>`_.

3. **Loop Unrolling**: Unrolling loops to increase the efficiency of vector operations. More on this at `Loop Unrolling <https://en.wikipedia.org/wiki/Loop_unrolling>`_.

4. **Prefetching**: Fetching data into cache before it's needed to minimize cache misses. Learn about `Prefetching <https://en.wikipedia.org/wiki/Cache_prefetching>`_.

5. **Transposition**: Efficiently managing data layout for improved access patterns, especially in matrix operations. See `Matrix Transposition <https://en.wikipedia.org/wiki/Transpose>`_.

*Vectorized Implementation in C*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is the C implementation of matrix multiplication using vectorization techniques to enhance performance:

.. code-block:: c

   void matmul_vectorized(float A[N][N], float B[N][N], float C[N][N]) {
       float (*B_col)[N] = aligned_alloc(32, N * N * sizeof(float));
       if (B_col == NULL) {
           fprintf(stderr, "Memory allocation failed\n");
           exit(1);
       }
       for (int j = 0; j < N; j += 32) {
           for (int k = 0; k < N; k++) {
               for (int jj = 0; jj < 32 && j + jj < N; jj++) {
                   B_col[j+jj][k] = B[k][j+jj];
               }
           }
       }
       {
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
                   }
               }
           }
       }
       free(B_col);
   }

*Performance Improvement*
^^^^^^^^^^^^^^^^^^^^^^^^^

The vectorized implementation significantly enhances performance by taking full advantage of CPU capabilities. On the AWS c7a.32xlarge instance, this approach achieves approximately **3000 GFLOPS**, representing a 6x performance increase over the previously optimized matrix multiplication. This demonstrates the power of vectorized operations in maximizing computational efficiency and speed in large-scale matrix operations.

Conclusion
----------

This exploration of matrix multiplication demonstrates the substantial gains possible through strategic optimizations in C. By transitioning from a naive implementation to a highly optimized vectorized approach, we achieved a 100x improvement in performance. These results underscore the importance of understanding and applying advanced techniques such as tiling, blocking, and SIMD vectorization.

The journey through these optimizations highlights the potential of C in unlocking the full computational capabilities of modern hardware. As machine learning models grow increasingly complex, mastering these techniques becomes crucial for developing efficient and scalable solutions. This foundational work provides a stepping stone for future explorations into more sophisticated algorithms and hardware accelerations.

References
----------

- `Matrix Multiplication on Wikipedia <https://en.wikipedia.org/wiki/Matrix_multiplication>`_
- `Linear Algebra Essentials <https://www.khanacademy.org/math/linear-algebra>`_
