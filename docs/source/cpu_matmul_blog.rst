.. _matrix-multiplication:

CPU Matrix Multiplication from Scratch in C
===========================================

.. admonition:: Highlights 

 Matrix multiplication is fundamental in deep learning, acting as the primary mechanism for executiong computations in neural networks. This blog post explores matrix multiplication using C to achieve performance comparable to Python's NumPy and explores detailed optimizations that leverage C's low-level capabilities.

 We demonstrate the performance evolution on an `AWS c7a.32xlarge EC2 instance <https://aws.amazon.com/ec2/instance-types/c7a/>`_ through the following implementations using 32-bit precision:

 - **Baseline with Python/NumPy**: **~3,500 GFLOPS**, using `this implementation <https://github.com/pebblesandweeds/cpu_matmul/blob/main/python/numpy_matmul.py>`_ for comparison with our C code.
 - **Naive C Approach**: **~25 GFLOPS**, starting with a simple `scalar implementation <https://github.com/pebblesandweeds/cpu_matmul/blob/main/c/src/matmul_lib.c#L28>`_.
 - **Optimized Scalar C**: **~500 GFLOPS**, optimizing the `naive scalar implementation <https://github.com/pebblesandweeds/cpu_matmul/blob/main/c/src/matmul_lib.c#L39>`_ using tiling, blocking, and loop unrolling.
 - **Vectorized C Operations**: **~3,,000 GFLOPS**, leveraging `vectorized instructions <https://github.com/pebblesandweeds/cpu_matmul/blob/main/c/src/matmul_lib.c#L64>`_ for enhanced performance.

 These results demonstrate the effectiveness of implementing matrix multiplication from scratch in C, achieving performance levels close to those of Python's NumPy.

 Get all of the code `in this repo <https://github.com/pebblesandweeds/cpu_matmul>`_.

Introduction
------------

Matrix multiplication is a core computational operation in machine learning, particularly in deep learning and neural networks.  Multiplying matrices is important for translating and scaling input data across various computational structures, a key step in processing interactions of data features within models architectures.  Optimizing computational efficiency and accuracy during both model training and inference phases directly relies on fast matrix multiplication.  

This blog explores building C implementations of CPU matrix multiplication from scratch, focusing on efficient algorithms and low-level optimizations. By working with C, we gain insights into performance enhancements often hidden by high-level libraries, critical for scaling matrix multiplication calculations for transformer based models to handle large volumes of computations.  


Why is Matrix Multiplication Important?
---------------------------------------

Matrix multiplication is essential to neural network computations, particularly in forward and backward propagation.

* **Forward Propagation:** Matrix multiplication combines input data with learned weights, producing activations that flow through the network.

* **Backward Propagation:** Matrix multiplication helps calculate gradients, enabling the network to update its weights and learn from errors.

The efficiency of matrix multiplication directly impacts the speed and scalability of neural network training and inference.

Matrix multiplication is vital in many machine learning domains, including natural language processing, computer vision, and audio processing. It is central to transforming input data, computing activations, and updating model parameters. These domains heavily rely on transformer architectures, where matrix multiplication powers self-attention mechanisms and feedforward networks, enabling the efficient computation of attention weights and data transformation through network layers.

As model sizes grow, computational demands increase significantly. Though no longer at the cutting edge, architectures like GPT-2 and GPT-3 still involve millions of matrix multiplications across multiple layers, attention mechanisms, embedding layers, position-wise feedforward networks, and output linear layers, highlighting the significant computational load requred to process and propagate inputs, activations, and gradients through these complex networks.

Optimizing matrix multiplication is therefore critical for improving the efficiency of large-scale models. It directly affects training time, resource consumption, and the ability to scale to larger datasets and model sizes. Understanding and optimizing this operation is key to advancing the capabilities of machine learning models while managing computational resources effectively.

Matrix Multiplication on CPUs Written in C
------------------------------------------

**Why aren't we using GPUs?**

* Starting with CPUs helps establish a foundational understanding of matrix multiplication and optimization techniques.
* Implementing matrix operations on CPUs helps us better understand performance optimization challenges and prepares us for future GPU implementations.

**Why aren't we using Python?**

* C provides low-level access to system resources and control over execution, ideal for optimizing performance in computationally intensive tasks.
* Building an implementation of matrix multiplication in C improves understanding of the memory hierarchy and optimizes access patterns, which are important for high-performance matrix operations.
* We can compare our C implementation to Python's NumPy to benchmark performance to enhance our understanding of matrix multiplication and optimization techniques.

Benchmarking Setup and Code Organization
----------------------------------------

*Matrix Configuration and Benchmarking Strategy*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our implementation performs matrix multiplication using the formula C = A x B, where both matrices A and B are square matrices of size N × N. We set N to a static size of 8,192, simplifying the implementation and laying the groundwork for future extensions to non-square matrices. By defining N with a preprocessor C macro (``#define N 8192``), we can enable aggressive compiler optimizations and ensure consistent runtime behavior.

In this setup, we are not implementing separate kernels with varying block sizes because each matrix is fixed at N × N. A kernel typically refers to an optimized code block designed for flexible execution across varying data sizes or hardware conditions. Since N is static in our implementation, the complexity of multiple kernels is unnecessary, allowing us to focus on optimizing for a single, fixed configuration.

Memory Requirements
'''''''''''''''''''

With N = 8,192, each matrix contains 67,108,864 elements. Using 32-bit floating-point precision (often referred to as "single precision" or "FP32"), the size of each matrix (A, B, and C) is calculated as follows:

.. math::

   67,108,864 \times 4 \text{ bytes} = 268,435,456 \text{ bytes} \approx 268 \text{ MB}

This results in a total memory requirement of approximately 805 MB for all three matrices.

Computational Complexity
''''''''''''''''''''''''

Calculating the computational effort for matrix multiplication involves determining the total number of floating point operations (FLOPs) needed. When multiplying two :math:`N \times N` matrices, the resulting matrix is also :math:`N \times N` (:math:`N^2` elements). Each element is the result of a dot product between a row from the first matrix and a column from the second matrix. This involves:

- **Multiplications:** Each element requires multiplying :math:`N` pairs of numbers (one from the row and one from the column).

- **Additions:** The products from the multiplications are then summed together, requiring :math:`N - 1` additions (adding two numbers requires one addition, adding three numbers requires two additions, etc).

Thus, the total number of FLOPs is calculated as:

.. math::

   \text{Total FLOPs} = 2N^3 - N^2

For large matrices, the :math:`2N^3` term contributes primarily to the total FLOPs, so it is often used to estimate the computational effort. This simplifies to:

.. math::

   \text{Total FLOPs} = 2N^3

This simplification highlights how the computational effort grows with the size of the matrices. For our chosen matrix size of 8192 x 8192, this results in:

.. math::

   2 \times 8192^3 = 1,099,511,627,776 \approx 1.1 \text{ TFLOPs}

This large number of operations underscores the computational intensity of large-scale matrix multiplication and highlights the importance of our optimization efforts. It is also important to note the distinction between FLOPs, which measure the total operations required, and FLOPS (Floating Point Operations Per Second), which indicate the system's performance capability.

Cache Considerations
''''''''''''''''''''

We chose this large N value (8,192) to represent a realistic problem size for our matrix multiplication.  With our matrix size of approximately 268MB each, the entire problem (all three matrices) doesn't fit in L3 cache simultaneously, but significant portions of the working set can potentially reside in cache during computation. This creates a scenario where careful cache management becomes crucial for performance. Our setup allows us to:

* Explore the effects of cache blocking and tiling optimizations
* Observe how different algorithms balance cache utilization and main memory access
* Understand performance characteristics that bridge cached and non-cached operations
* Investigate how implementations handle a problem that doesn't neatly fit entirely in cache, but is also not so large as to make cache optimizations irrelevant

This approach provides insight into algorithm design for real-world, cache-sensitive computations.

Benchmarking Environment
''''''''''''''''''''''''

Building on our choice of matrix size, we conducted our benchmarks on an AWS c7a.32xlarge instance. This instance features an AMD EPYC 9R14 processor with 2 sockets, 64 cores per socket, totaling 128 cores without simultaneous multithreading. Notably, it has a 512MB L3 cache, which, while substantial, is still smaller than our total working set size of about 805MB (three 268MB matrices).  This configuration allows us to observe the interplay between cache utilization and main memory access, as significant portions of the working set can reside in cache during computation, but not the entire problem simultaneously. By using this high-performance computing infrastructure, we can obtain performance metrics that reflect real-world scenarios for large-scale matrix multiplication, where careful cache management becomes crucial for optimal performance. 

*Code Structure and Organization*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Our matrix multiplication code is organized into separate modules for clarity and maintainability. The primary files are:

* `matmul_lib.c <https://github.com/pebblesandweeds/cpu_matmul/blob/dev/c/src/matmul_lib.c>`_: Contains the core matrix multiplication functions.
* `main.c <https://github.com/pebblesandweeds/cpu_matmul/blob/dev/c/src/main.c>`_: Serves as the entry point, calling functions from ``matmul_lib.c``.
* `Makefile <https://github.com/pebblesandweeds/cpu_matmul/blob/main/c/Makefile>`_: Specifies teh build process using the ``gcc`` compiler with optimization flags ``CFLAGS = -mavx2 -fopenmp -O3 -march=native -I./include``

For a detailed overview of our project structure, please refer to our `README.md <https://github.com/pebblesandweeds/cpu_matmul/blob/dev/README.md#project-structure>`_.

As we explore different optimization techniques, we'll focus on relevant functions from ``matmul_lib.c``. We'll discuss how these functions implement various matrix multiplication methods and their associated performance gains.

The code snippets in this blog omit the ``#pragma`` preprocessor directives for simplicity. The full code in our repository includes parallel instructions, which are out of scope for our discussions here.

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

   #define BLOCK_SIZE 64 // Optimizes memory across L1/L2/L3; fetch data in chunks 
   #define TILE_SIZE 32 // Improves CPUs data processing; balances CPU resources and data caching
   #define UNROLL_FACTOR 4 // Increases parallel operations w/out overwhelming memory

   void matmul_scalar(float A[N][N], float B[N][N], float C[N][N]) {
   // Outer loops for block-wise operations
    for (int i = 0; i < N; i += BLOCK_SIZE) {
    for (int j = 0; j < N; j += BLOCK_SIZE) {
    for (int k = 0; k < N; k += BLOCK_SIZE) {
        // Inner loops for tile-wise operations within blocks
        for (int ii = i; ii < i + BLOCK_SIZE && ii < N; ii += TILE_SIZE) {
        for (int jj = j; jj < j + BLOCK_SIZE && jj < N; jj += TILE_SIZE) {
        // Loop unrolling for innermost loop
        for (int kk = k; kk < k + BLOCK_SIZE && kk < N; kk += UNROLL_FACTOR) {
            float c_temp = C[ii][jj]; // Temp variable for accumulation
            // Compute on tiles
            for (int iii = ii; iii < ii + TILE_SIZE && iii < i + BLOCK_SIZE && iii < N; iii++) {
            for (int jjj = jj; jjj < jj + TILE_SIZE && jjj < j + BLOCK_SIZE && jjj < N; jjj++) {
                // Matrix multiplication within a tile
                c_temp += A[iii][kk] * B[kk][jjj];
            }
            C[iii][jjj] = c_temp; // Store accumulated results
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

By optimizing matrix multiplication, we achieve a significant performance boost. Our approach employs three key strategies: dividing matrices into cache-friendly blocks, further subdividing into efficiently processable tiles, and using loop unrolling for parallel operations. These techniques work together to ensure optimal data availability and CPU resource utilization.

On the AWS c7a.32xlarge instance, this optimized implementation achieves approximately **500 GFLOPS**, representing more than a *20x increase* over the naive approach. This dramatic improvement stems from better use of the CPU's cache hierarchy, reduced memory access times, and increased instruction-level parallelism. While further scalar optimizations are possible, we're approaching the limits of what can be achieved without leveraging more advanced hardware features. The next step in boosting performance is to utilize vectorized operations, which we'll explore in the following section.

Vectorized Matrix Multiplication
--------------------------------

*Scalar vs. Vectorized Operations*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scalar operations process data one element at a time, performing calculations sequentially. In contrast, vectorized operations use a Single Instruction, Multiple Data (SIMD) approach, processing multiple data elements simultaneously. This parallelism is implemented on CPUs through SIMD instructions, which leverage hardware capabilities to execute the same operation on multiple data points in a single instruction cycle.

To write vectorized code, several elements are necessary:

1. **SIMD Instructions**: Using SIMD instructions like AVX for parallel computation, including operations like Fused Multiply-Add (FMA) for combining addition and multiplication. Learn more about SIMD from `Wikipedia <https://en.wikipedia.org/wiki/SIMD>`_.

2. **Data Alignment**: Ensuring data is aligned in memory for efficient SIMD processing. Check out `Data Alignment <https://en.wikipedia.org/wiki/Data_structure_alignment>`_.

3. **Loop Unrolling**: Unrolling loops to increase the efficiency of vector operations. More on this at `Loop Unrolling <https://en.wikipedia.org/wiki/Loop_unrolling>`_.

4. **Prefetching**: Fetching data into cache before it's needed to minimize cache misses. Learn about `Prefetching <https://en.wikipedia.org/wiki/Cache_prefetching>`_.

5. **Transposition**: Efficiently managing data layout for improved access patterns, especially in matrix operations. See `Matrix Transposition <https://en.wikipedia.org/wiki/Transpose>`_.

*Vectorized Implementation in C*
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Below is the C implementation of matrix multiplication using vectorization techniques to enhance performance:

.. code-block:: c

   void matmul_vectorized(float A[N][N], float B[N][N], float C[N][N]) {
       // Data alignment (allocate memory for B_col)
       float (*B_col)[N] = aligned_alloc(32, N * N * sizeof(float));
       if (B_col == NULL) {
           fprintf(stderr, "Memory allocation failed\n");
           exit(1);
       }
       // Transposition (transpose B into B_col for better memory access patterns)
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
                   // SIMD instructions (__m256 for 256-bit for SIMD operations)
                   __m256 c[32][32];
                   for (int ii = 0; ii < 32; ii++) {
                       for (int jj = 0; jj < 32; jj++) {
                           c[ii][jj] = _mm256_setzero_ps();
                       }
                   }
                   for (int k = 0; k < N; k += 32) {
                       // Prefetching (fetch data into cache before we use it)
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
                       // Loop unrolling (unroll inner loop for vector operations) and FMA (fused multiply-add)
                       for (int ii = 0; ii < 32; ii++) {
                           for (int jj = 0; jj < 32; jj++) {
                               c[ii][jj] = _mm256_fmadd_ps(a[ii][0], b[jj][0], c[ii][jj]);
                               c[ii][jj] = _mm256_fmadd_ps(a[ii][1], b[jj][1], c[ii][jj]);
                               c[ii][jj] = _mm256_fmadd_ps(a[ii][2], b[jj][2], c[ii][jj]);
                               c[ii][jj] = _mm256_fmadd_ps(a[ii][3], b[jj][3], c[ii][jj]);
                           }
                       }
                   }
                   // SIMD Instructions (final matrix multiplication reduction using SIMD)
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

The vectorized implementation significantly enhances performance by leveraging several key techniques. Data alignment ensures efficient memory access for SIMD operations, while transposition optimizes data layout to improve access patterns crucial for matrix operations. Using SIMD instructions and the 256-bit YMM `registers <https://en.wikipedia.org/wiki/Processor_register>`_, the implementation enables parallel processing of multiple data points, allowing the simultaneous processing of eight single-precision floating-point numbers and maximizing data throughput. Prefetching minimizes cache misses by loading data before it's needed, and loop unrolling increases the throughput of vector operations by reducing loop overhead and enabling more parallel execution of instructions. Together, these techniques fully utilize the CPU's vectorization capabilities, resulting in a dramatic performance boost.

On the AWS c7a.32xlarge instance, this vectorized approach achieves approximately **3000 GFLOPS**, representing a *6x performance increase* over the previously optimized scalar implementation. This stark contrast highlights the fundamental difference between scalar and vectorized operations. While scalar operations process data sequentially, one element at a time, vectorized operations leverage SIMD to process multiple data elements simultaneously. This parallelism at the instruction level allows for significantly higher throughput, demonstrating the power of vectorized operations in maximizing computational efficiency and speed in large-scale matrix operations.

Conclusion
----------

Our journey through matrix multiplication optimization showcases the remarkable potential for performance enhancement in computational tasks. Starting from a naive implementation that achieved only 25 GFLOPS, we progressed to an optimized scalar version reaching 500 GFLOPS—a 20x improvement. The final leap to vectorized operations propelled our performance to an impressive 3000 GFLOPS, marking a 120x total increase from our starting point. This progression underscores the power of thoughtful optimization techniques such as cache-friendly blocking, efficient tiling, and SIMD vectorization in maximizing CPU capabilities.

Notably, our vectorized C implementation from scratch nearly achieves parity with NumPy's performance of 3500 GFLOPS. This accomplishment not only demonstrates the effectiveness of low-level optimizations but also provides valuable insights into the inner workings of high-performance libraries. By building this foundation on CPUs, we've gained a deep understanding of memory hierarchies, cache utilization, and parallelism strategies. These lessons form a solid basis for future GPU implementations, where concepts like data locality and parallel processing are equally crucial, albeit applied differently. As we look towards GPU optimization, we're well-equipped to tackle the unique challenges and opportunities presented by massively parallel architectures, armed with a comprehensive understanding of performance optimization principles.

Further Reading
---------------

* `GEMM Optimization Tutorial <https://github.com/flame/how-to-optimize-gemm>`_ and `BLISlab Tutorial <https://github.com/flame/blislab/blob/master/tutorial.pdf>`_
* `Beating NumPy in 150 lines of C Code <https://salykova.github.io/matmul-cpu>`_ plus the `repo <https://github.com/salykova/matmul.c>`_
* George Hotz's six hour video stream `Can You Mutliply a Matrix? <https://youtu.be/VgSQ1GOC86s?si=HP1VB1UDF384_xQt>`_ and `gemm.c code <https://github.com/tinygrad/tinygrad/blob/master/extra/gemm/gemm.c>`_
