.. _rccl-matrix-multiplication:

Scaling Matrix Multiplication Across Multiple AMD GPUs with RCCL and rocBLAS
============================================================================

.. admonition:: Highlights

 Scaling matrix multiplication beyond a single GPU presents both opportunities and challenges in deep learning. This blog post demonstrates how to scale our `previous single-GPU implementation <https://blog.pebblesandweeds.com/gpu_matmul_blog.html>`_ to efficiently utilize multiple GPUs in a single server through AMD's RCCL library, showing how coordination of communication and computation can achieve near-linear performance scaling.

 - **Scaling Efficiency**: Using baseline performance from our previous single-GPU implementation, we achieve equivalent per-GPU throughput when distributed across 8 GPUs (~35 TFLOPS per GPU, ~280 TFLOPS aggregate). This demonstrates that RCCL's communication primitives impose minimal overhead, as each GPU maintains the baseline performance while coordinating through broadcast and allGather operations.

 - **Memory Distribution**: We performed multiplication of 32,768 x 32,768 single precision matrices by horizontally chunking matrix A across eight (8) GPUs while broadcasting matrix B. This reduces per-GPU memory requirements from 12.87 GB to ~5.36 GB while enabling parallel computation of the results.

 - **RCCL Communication**: Implemented single-host, multi-GPU coordination through RCCL collective operations, broadcasting matrix B across GPUs and combining partial results through allGather. These high-level primitives handle the complex low-level details of efficient inter-GPU data transfer.

 - **PyTorch Validation**: Implemented simple distributed `Pytorch <https://github.com/pebblesandweeds/rccl_gpu_matmul/blob/dev/pytorch/pytorch_rccl.py>`_ code using torch.distributed primitives that achieved matching multi-GPU performance (34.6-35.7 TFLOPS per GPU), validating our low-level C and RCCL implementation against PyTorch's established distributed computing framework.

 This implementation demonstrates how proper coordination between RCCL communication and rocBLAS computation enables efficient scaling across multiple GPUs while maintaining high performance. Our C implementation provides insight into distributed GPU computing concepts while achieving performance parity with PyTorch's optimized framework.

Introduction
------------

In our `previous blog post <https://blog.pebblesandweeds.com/gpu_matmul_blog.html>`_, we implemented matrix multiplication in C using AMD's `rocBLAS <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/>`_ library, specifically utilizing the `rocblas_sgemm <https://rocm.docs.amd.com/projects/rocBLAS/en/latest/reference/level-3.html#rocblas-xgemm-batched-strided-batched>`_ API to leverage AMD's fast GPU `matrix cores <https://www.amd.com/en/technologies/cdna.html>`_. The implementation demonstrated that carefully written C code using rocBLAS could match PyTorch's highly optimized matrix operations, allowing us to achieve the same performance with a lower-level implementation.

While our previous work focused on single-GPU matrix multiplication, this operation is inherently parallelizable - computations can be efficiently distributed across multiple GPUs with minimal dependencies between parallel tasks. Modern servers and supercomputers systems support this parallelism by providing multiple GPUs per node, enabling significant computational speedups through parallel execution. While our `single-GPU implementation <https://github.com/pebblesandweeds/gpu_matmul>`_ demonstrated basic rocBLAS capabilities, the parallel nature of matrix multiplication makes it an ideal candidate for multi-GPU execution.

This post extends our previous work by distributing matrix multiplication across multiple GPUs within a single host using `RCCL <https://github.com/ROCmSoftwarePlatform/rccl>`_ (ROCm Communication Collectives Library). `RCCL provides <https://rocm.docs.amd.com/projects/rccl/en/latest/>`_ efficient communication primitives between GPUs, similar to NVIDIA's NCCL, enabling us to coordinate computation across all available devices to maximize hardware utilization and computational throughput. Our goal is to show how to extend our single-GPU rocBLAS implementation in C to utilize RCCL for coordinating matrix multiplication across multiple GPUs in a single host system.

Scaling Matrix Multiplication: From Single to Multi-GPU Systems
----------------------------------------------------------------

Single-GPU Matrix Multiplication
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The rocBLAS ``rocblas_sgemm`` API implements high-performance single precision (fp32) matrix multiplication using AMD's matrix core accelerators (detailed formula and optimizations are covered in our `previous post <https://blog.pebblesandweeds.com/gpu_matmul_blog.html#matrix-multiplication-formulas>`_). The core workflow involves transferring input matrices A and B to GPU memory, executing the multiplication, and transferring result matrix C back to host memory.

While this appears straightforward, achieving peak performance requires careful orchestration of memory transfers, matrix layouts, and compute scheduling. Thankfully, rocBLAS abstracts away many of these complexities - it handles matrix padding and alignment to maximize memory throughput, manages optimal blocking strategies for AMD's matrix cores, and provides batching capabilities for efficient execution of multiple multiplications. This allows developers to focus on high-level algorithm design while the library manages the hardware-specific optimizations.

Even though this single-GPU approach delivers good performance for matrices that fit within GPU memory, it is ultimately constrained by both memory capacity and computational throughput of a single device. A modern GPU can deliver impressive TFLOP/s for matrix operations, but most AI workloads demand higher computational capabilities than a single GPU can deliver. These performance demands, combined with memory limitations, motivate exploration of multi-GPU approaches that can harness both the aggregate compute power and memory capacity of multiple devices.

.. figure:: _static/single-gpu-flow.png
  :alt: Single GPU Matrix Multiplication Workflow
  :align: center

  Simple matrix multiplication on single GPU

Distributed Matrix Multiplication 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Extending beyond a single device, we can leverage multiple GPUs within a host system to dramatically increase both computational throughput and available memory. The key lies in efficiently partitioning the workload while minimizing data transfers between devices.

Our distributed implementation employs a horizontal partitioning strategy that balances computational efficiency with communication overhead through several key mechanisms:

* **Matrix Distribution** - Matrix A is split horizontally across GPUs while matrix B is broadcast in its entirety to each device using RCCL, allowing independent processing of matrix partitions using rocBLAS primitives.

* **Result Consolidation**: The system combines partial results from each device through RCCL's allGather operation, constructing the final output matrix

* **Performance Optimization**: The approach maximizes efficiency through balanced computational load from the horizontal split of A, minimizing inter-GPU communication through a single broadcast of B, and requiring only one collective operation during result collection via allGather

Through these design choices, we transform our earlier single-GPU implementation into a scalable distributed system that preserves the computational efficiency of rocBLAS while extending across multiple devices.

.. figure:: _static/matmul_rccl_workflow.png
   :alt: Distributed Matrix Multiplication Workflow
   :align: center

   Distributed matrix multiplication across multiple GPUs

Broadcasting matrix B instead of partitioning it optimizes our approach for deep learning workloads. While this requires more memory per GPU, it significantly reduces communication overhead based on how matrices A and B are used in practice:

* Matrix B contains model weights that remain constant across many computations
* Matrix A holds the activations or embeddings that change with each forward pass
* Matrix multiplication requires each row of A to interact with every column of B. Partitioning B by columns would force GPUs to exchange partial results, since computing a single output row needs access to all of B's columns

Given modern GPU memory capacities and the characteristic reuse of parameter matrices in deep learning workloads, the higher memory cost of broadcasting B is outweighed by the reduced communication overhead.

Implementing Multi-GPU Matrix Multiplication
--------------------------------------------

Implementation Libraries
^^^^^^^^^^^^^^^^^^^^^^^^
Our implementation leverages two core AMD libraries:

**rocBLAS for Matrix Computation**

The ``rocblas_sgemm`` API handles matrix multiplication on each GPU. We covered the single-GPU implementation in our `previous blog <https://blog.pebblesandweeds.com/gpu_matmul_blog.html#rocblas-sgemm-api>`_, the multi-GPU version works similarly - each device executes its own matrix multiplication after receiving its portion of matrix A and a complete copy of matrix B. rocBLAS optimizes these computations for AMD's matrix cores, managing memory layouts and compute scheduling automatically.

**RCCL for GPU Communication**

RCCL (ROCm Communication Collectives Library) provides efficient primitives for moving data between GPUs. While this is AMD's library, it maintains API compatibility with NVIDIA's NCCL - hence the ``nccl`` prefix in function names like ``ncclBroadcast``. Our implementation uses two key RCCL operations:

* ``ncclBroadcast`` distributes matrix B to all GPUs during initialization
* ``ncclAllGather`` combines partial results from each GPU's computation into the final output matrix

RCCL handles the complexity of optimal data transfer paths between GPUs, utilizing direct GPU-to-GPU communication when available and automatically selecting the most efficient transfer methods based on system topology.

The interaction between these libraries follows a clear pattern: RCCL first distributes the input data across devices, rocBLAS performs local computations on each GPU, and finally RCCL consolidates the results. This separation of tasks - RCCL for communication and rocBLAS for computation - allows each library to optimize its specific role while working together for efficient distributed processing.

Memory Requirements
^^^^^^^^^^^^^^^^^^^

Let's examine the memory distribution patterns across GPUs in our matrix multiplication implementation. For this discussion, we'll use 32K × 32K matrices with single precision floating point values (fp32, 4 bytes per element). Each complete matrix occupies:

.. math::

   32,768 \times 32,768 \times 4 \text{ bytes} \approx 4.29 \text{ GB}

While modern enterprise GPUs can handle much larger matrices, this size provides a practical example for demonstrating how distributed computation reduces memory requirements per device.

**Single-GPU Memory Footprint**

When running matrix multiplication on a single GPU using rocBLAS, we need all three matrices to reside in device memory. With each matrix requiring 4.29 GB, our total VRAM usage is ~12.87 GB for matrices A, B, and C. While this memory footprint is within the capabilities of modern GPUs, distributing these matrices across devices we can reduce the per-GPU memory requirements, allowing us to perform larger computations and to process multiple matrix multiplications in parallel (batches).

**Distributed Memory Layout**

Our 8-GPU implementation reduces per-device memory usage through selective matrix distribution. Each GPU stores:

* 1/8th chunk of matrix A: 4.29 GB ÷ 8 ≈ 536 MB
* Complete copy of matrix B: 4.29 GB
* 1/8th chunk of output matrix C: 536 MB

This distribution strategy requires ~5.36 GB per GPU compared to the 12.87 GB needed for single-GPU execution. The reduction stems from dividing matrices A and C across devices while broadcasting B to each GPU. While in this example our memory savings are modest, this pattern becomes increasingly important when scaling to larger matrices or processing multiple matrix multiplications in parallel.

It's worth noting that in real world deep learning applications, we typically process batches of matrix multiplications rather than single operations. While batched operations are beyond the scope of this blog post, the memory distribution strategy demonstrated here (chunking A and C while broadcasting B) provides a foundation for handling these larger workloads using less VRAM.

RCCL Implementation Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When distributing matrix multiplication across multiple GPUs, several factors influence overall system performance:

**Communication Overhead and Hardware**

Distributing computation across multiple GPUs introduces unavoidable overhead from both communication costs and the inherent challenges of parallel workloads. While a single GPU might achieve :math:`X` teraflops of performance, scaling to :math:`N` GPUs will not yield :math:`N \times X` teraflops due to these distributed computing overheads. Our goal is to minimize this scaling efficiency loss through careful management of the three main communication costs:

* Initial distribution of matrix chunks across devices
* Broadcasting matrix B to all GPUs
* Final gathering of results using ncclAllGather

The impact of these transfers depends on the system's GPU interconnect topology since different interconnects offer varying bandwidth and latency characteristics. PCIe and vendor-specific interconnects provide different performance tradeoffs, which RCCL leverages by automatically selecting transfer paths that minimize communication overhead based on the specific hardware topology.

**Stream Management and Execution Flow**

Our implementation creates independent HIP streams per GPU to manage asynchronous operations. The streams coordinate:

* Asynchronous memory transfers between host and device
* RCCL collective operations (broadcasts and gathers)
* rocBLAS matrix multiplication kernels

The code uses RCCL's group start end semantics to batch communication operations, with explicit synchronization through hipStreamSynchronize and hipDeviceSynchronize ensuring completion at critical points.

**Workload Distribution Strategy**

The implementation divides matrix A into equal-sized chunks across available GPUs, with each device processing an equal portion of rows. Matrix B is broadcast in full to all devices. Each GPU computes its portion of the final result matrix C, which is then gathered using ncclAllGather to reconstruct the complete output.

Through this design, we minimize the overhead inherent in distributed computation while maximizing hardware utilization. The approach scales efficiently with additional GPUs while preserving the computational benefits of rocBLAS's optimized matrix operations on each device.

Code Walkthrough
^^^^^^^^^^^^^^^^
Let's walk through the key components of our multi-GPU matrix multiplication implementation, examining how RCCL coordination, memory management, and computation work together to achieve high performance.

The first step involves setting up the RCCL context and allocating memory across our GPU array. Each GPU needs its own chunk of matrix A, a full copy of matrix B, and space for its portion of the result matrix C:

.. code-block:: c

    // Initialize RCCL context
    RCCLContext* rccl_ctx = rccl_init(num_gpus);
    for (int i = 0; i < num_gpus; i++) {
        CHECK_HIP(hipSetDevice(i));
        CHECK_HIP(hipMalloc(&d_A_chunks[i], chunk_bytes));
        CHECK_HIP(hipMalloc(&d_B[i], full_size));
        CHECK_HIP(hipMalloc(&d_C_chunks[i], chunk_bytes));
        CHECK_HIP(hipMalloc(&d_C_final[i], full_size));
        // Copy data to devices
        CHECK_HIP(hipMemcpyAsync(d_A_chunks[i],
                               h_A + (i * chunk_size * N),
                               chunk_bytes,
                               hipMemcpyHostToDevice,
                               rccl_ctx->streams[i]));
    }

The ``CHECK_HIP`` macro below wraps all HIP API calls to provide error handling. The macro checks the returned `hipError_t` status code and terminates execution with an error message if the operation fails:

.. code-block:: c

    #define CHECK_HIP(stmt) do {
        hipError_t err = stmt;
        if (err != hipSuccess) {
            printf("HIP error: %s\n", hipGetErrorString(err));
            exit(1);
        }
    } while(0)

Next, we use RCCL to broadcast matrix B to all GPUs before performing our computation. The ``ncclGroupStart`` and ``ncclGroupEnd`` functions create a collective communication group that allows multiple NCCL operations to be executed together for improved performance, while the ``ncclBroadcast`` function copies data from a source GPU (specified by rank 0) to all other GPUs in the communicator, ensuring each device has an identical copy of matrix B:

.. code-block:: c

   void rccl_broadcast_matrix(RCCLContext* ctx, float** send_data, size_t elements) {
       CHECK_NCCL(ncclGroupStart());
       for (int i = 0; i < ctx->num_gpus; i++) {
           CHECK_HIP(hipSetDevice(i));
           CHECK_NCCL(ncclBroadcast(send_data[i], send_data[i], elements,
                                   ncclFloat, 0, ctx->comms[i], ctx->streams[i]));
       }
       CHECK_NCCL(ncclGroupEnd());
   }

Once the broadcast is complete, each GPU performs matrix multiplication on its assigned chunk of matrix A while utilizing its full copy of matrix B. We pass matrix B as the first input matrix to the rocBLAS API instead of matrix A. This works because our matrices are in row-major order while rocBLAS expects column-major order. When passing row-major matrices to rocBLAS's column-major API, each matrix is implicitly transposed. So passing :math:`(B,A)` in row-major becomes :math:`B^T * A^T` in column-major, which equals :math:`(A * B)^T`. When we read the result back in row-major, it's transposed again, giving us :math:`A * B`. This lets us avoid explicit transpose operations while getting correct results:

.. code-block:: c

  void perform_matrix_multiplication(
      rocblas_handle* handles,
      float** d_A_chunks,
      float** d_B,
      float** d_C_chunks,
      int N,
      int chunk_size,
      int num_gpus,
      hipStream_t* streams,
      int NUM_RUNS) {
      const float alpha = 1.0f;
      const float beta = 0.0f;
      for (int i = 0; i < num_gpus; i++) {
          CHECK_HIP(hipSetDevice(i));
          CHECK_ROCBLAS(rocblas_sgemm(handles[i],
                                     rocblas_operation_none,
                                     rocblas_operation_none,
                                     N, chunk_size, N,
                                     &alpha,
                                     d_B[i], N,
                                     d_A_chunks[i], N,
                                     &beta,
                                     d_C_chunks[i], N));
      }
  }

After the multiplication, we collect the computed chunks using ncclAllGather - each GPU contributes its portion ``chunks[i]`` and every GPU receives a complete copy in ``result[i]``. While each GPU ends up with an identical copy of the full result, we only copy GPU[0] version back to host memory:

.. code-block:: c

   void rccl_gather_matrix_chunks(RCCLContext* ctx, float** chunks, float** result,
                                size_t chunk_elements) {
       CHECK_NCCL(ncclGroupStart());
       for (int i = 0; i < ctx->num_gpus; i++) {
           CHECK_HIP(hipSetDevice(i));
           CHECK_NCCL(ncclAllGather(chunks[i], result[i], chunk_elements,
                                   ncclFloat, ctx->comms[i], ctx->streams[i]));
       }
       CHECK_NCCL(ncclGroupEnd());
   }

   // In main(), we only copy GPU 0's result back to host
   printf("Copying results back to host\n");
   CHECK_HIP(hipSetDevice(0));
   CHECK_HIP(hipMemcpy(h_C, d_C_final[0], full_size, hipMemcpyDeviceToHost));

To track performance across all GPUs, we use HIP events to measure computation time and calculate achieved TFLOPS for each device. Each GPU handles a portion of the matrix multiplication - since the input is evenly divided, each GPU does an equal share of the total floating point operations. The code records the start and stop times using HIP events, calculates how long each GPU took in milliseconds, and converts this timing into TFLOPS (trillions of floating point operations per second) to show each GPU's computational speed:

.. code-block:: c

   hipEvent_t starts[num_gpus], stops[num_gpus];
   for (int i = 0; i < num_gpus; i++) {
       CHECK_HIP(hipEventCreate(&starts[i]));
       CHECK_HIP(hipEventRecord(starts[i], streams[i]));
       // Perform computation
       CHECK_HIP(hipEventRecord(stops[i], streams[i]));
       float compute_time;
       CHECK_HIP(hipEventElapsedTime(&compute_time, starts[i], stops[i]));
       double tflops = (chunk_flops / (compute_time / 1000.0)) / 1e12;
       printf("GPU %d: Time: %.2f ms, Performance: %.2f TFLOPS\n",
              i, compute_time, tflops);
   }

This implementation shows how we can scale matrix multiplication across multiple GPUs by combining RCCL's inter-GPU communication with rocBLAS's optimized computation. By dividing work evenly, coordinating data movement with ``ncclBroadcast`` and ``ncclAllGather`` operations, and letting each GPU process its chunk independently, we maintain the high performance of rocBLAS while distributing the computational load across the available hardware.

Performance Analysis
--------------------

We evaluated our distributed matrix multiplication implementation by first establishing a baseline using our previous `single-GPU implementation <https://github.com/pebblesandweeds/gpu_matmul>`_, then comparing it against our new multi-GPU RCCL code running on the same hardware. This approach allowed us to directly measure any overhead introduced by RCCL communication when scaling from single to multi-GPU execution.

Benchmark Configuration
^^^^^^^^^^^^^^^^^^^^^^^
Our test environment consisted of:

* **Hardware**
   * AMD Instinct MI250X GPUs (1-8 GPUs)
   * GPU Clock: 1700 MHz
* **Test Parameters**
   * Matrix Dimensions: 32,768 x 32,768 (FP32)
   * 25 consecutive multiplication runs per configuration
   * ROCm 6.0.2
* **Implementations Tested**
   * Single GPU: Single-GPU `rocBLAS C implementation <https://github.com/pebblesandweeds/gpu_matmul>`_ 
   * Multi-GPU: Mult-GPU `RCCL-based C implementation <https://github.com/pebblesandweeds/rccl_gpu_matmul>`_
   * PyTorch: Distributed implementation for validation

Multi-GPU Scaling Analysis
^^^^^^^^^^^^^^^^^^^^^^^^^^
Our single-GPU baseline implementation achieved 34.58-35.87 TFLOPS for matrix multiplication, establishing our performance target for per-GPU throughput in the distributed system. When scaling to 8 GPUs using our new RCCL implementation, we observed per-GPU performance of 34.7-35.7 TFLOPS, resulting in aggregate system throughput of approximately 280 TFLOPS. The consistent per-GPU performance between single and multi-GPU execution demonstrates that RCCL's broadcast and allGather operations impose minimal overhead with our horizontal partitioning strategy.

* **Single GPU Baseline**: 34.58-35.87 TFLOPS (using previous gpu_matmul implementation)
* **Multi-GPU Range**: 34.7-35.7 TFLOPS per GPU (using new RCCL implementation)
* **Aggregate Performance**: ~280 TFLOPS across 8 GPUs
* **Scaling Efficiency**: >98% per-GPU performance maintained when scaling to 8 GPUs

PyTorch Implementation Comparison
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
To validate our C implementation, we developed an equivalent distributed PyTorch version that performs the same matrix broadcast and multiplication operations using torch.distributed primitives. The PyTorch implementation achieved similar performance characteristics after warm-up, matching our C code's performance envelope. This verification demonstrates that our low-level RCCL and rocBLAS implementation achieves comparable efficiency to PyTorch's optimized framework while providing direct control over the distributed computation pattern.

* **Per-GPU Range**: 34.6-35.7 TFLOPS
* **Aggregate Performance**: ~280 TFLOPS
* **Implementation**: Uses torch.distributed for matrix broadcast and distributed computation

Conclusion
----------

Our exploration of multi-GPU matrix multiplication using AMD's RCCL and rocBLAS libraries demonstrated how to efficiently scale matrix operations across multiple devices while maintaining high per-GPU performance. Starting with our previous single-GPU implementation that achieved 34.58-35.87 TFLOPS, we showed that distributing 32,768 x 32,768 matrices across 8 GPUs could deliver ~280 TFLOPS of aggregate performance while maintaining equivalent per-GPU throughput (34.7-35.7 TFLOPS). This near-linear scaling emphasizes the efficiency of our RCCL-based coordination approach for large-scale computations.

Both the PyTorch and C implementations produced nearly identical performance results, with both reaching approximately 280 TFLOPS. This confirms that while high-level frameworks like PyTorch simplify distributed programming, low-level programming with RCCL and rocBLAS offers comparable efficiency while providing deeper insight into GPU communication patterns and distributed memory management. Most importantly, our horizontal partitioning strategy proved effective, reducing per-GPU memory requirements from 12.87 GB to ~5.36 GB while maintaining the baseline computational throughput of our original single-GPU implementation - demonstrating the practical benefits of distributed GPU computing for handling large-scale matrix operations in deep learning workloads.

Thanks for reading! For more details, check out our GitHub repository. Stay tuned for future blogs where we'll explore more advanced topics in distributed GPU computing.

