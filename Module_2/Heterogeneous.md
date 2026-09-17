# Heterogeneous

## Basics of Parallelism

### Threads

- **Definition:** Sequence of instructions that can execute independently within a program
  - Specific Information:
    - **threads within a process share same memory**
    - **each thread has its own execution state**
    - **multiple threads can execute concurrently and divide a larger task**
  - CPU Thread != GPU Thread
  - **Thread represents an independent sequence of work**
- Parallelization: divide independent work among mulitple threads
- Thread ID: determines which data elements the thread processes
  - Data Parallelism

### Data Parallelism

- **Task Parallelism:**
  - different operations performed on same or different data elements
  - a modest number of tasks unleashing a modest amount of parallelism
- **Data Parallelism**
  - same operation applies to many different data elements
  - massive amounts of data unleashing massive amounts of parallelism
  - GPU Programming often begins by finding this repeated operation

### CUDA Processor and Memory Space

- Host(CPU)
  - Runs sequential control and launches work
- Device(GPU)
  - Runs many parallel threads
  - explicit-memory model
    - data must be moved between host and device memory

### CUDA Program Five-Stage Lifecycle

- Allocate device memory
- Copy inputs host -> device
- Launch the computation **kernel** on the GPU
- Copy results from device -> host
- Free device memory

### Device Memory Management

- Allocating Memory:

```cpp
    cudaError_t cudaMalloc(void **devPtr, size_t size){}
    - devPtr: pointer to pointer to allocated device memory
    - size: requested allocation size in bytes
```

- Deallocating Memory:

```cpp
    cudaError_t cudaFree(void *devPtr)
    - devPtr: pointer to device memory to free
```

- Return Type: helps with error checking
  
- **cudaMemcpy:** makes transfer direction explict
  - cudaMemcpy(dest, src, byte_count, direction)
  - If you reverse the direction or byte count (it is a correctness bug too)
  - **Direction:** transfer type
    - cudaMemcpyHostToHost
    - cudaMemcpyHostToDevice
    - cudaMemcpyDeviceToHost
    - cudaMemcpyDeviceToDevice

### Abstracting Out of a Loop With a Grid of Threads

- Serial: one thread performs N iterations
- Parallel: N logical workers (threads) perform one iteration
- The program must create enough threads and give each thread a unique element index
- EX: Parallel Vector Addition
  - each GPU thread gets a vector element
- **Grid:** thread population launched by 1 kernel
  - every thread in grid executes same kernel function (watdatmean)
  - Indices: allow identical code to perform operations on different data
  - Single-Program Mulitple Data (SPMD): same code, different Thread IDs (TIDs), different data
    - programming model that expresses parallelism
  - **Thread Blocks:** an array of threads that can cooperate efficiently within a grid (an array of a block of threads)
    - TIDs are local to each block which means they can help each other better than if they were in different blocks
  - **Launch Configuration**
    - Threads in same grid execute same kernel function
    - Grid can be launched by calling kernel and configuring it with number of threads and block sizes
    - Launch shape is host-side decision (watdatmean)
    - Block sizes are often multiples of 32 (e.g. 256)

### Kernel Information

- Similar to a C/C++ function
- preceded by \__global__\ keyword to indicate that it is a GPU kernel
  - \__global__\ ensures that a device's code is callable by the host
- All threads execute the same instructions but see different built-in indices
- All threads execute the same instructions but see different built-in indices
- A thread's index determines which data elements it owns
  - **Special Keywords** distinguish different threads from each other

#### Example Functions

- Grid Dimension: gridDim.x reports number of blocks in grid
  - gridDim.x describes entire thread launch in blocks
    - shared by every thread in 1D grid
- Block Index: blockIdx.x indentifies a block within the grid
  - ranges from 0 to gridDim.x - 1
  - Every thread in the same block sees the same block index
- Block Dimension: blockDim.x reports the number of threads in each block
  - All blocks use the same block dimensions
  - blockDim.x converts a block index into starting offset of that block's data
- Thread Index: threadIdx.x identifies a thread within its block
  - ranges from 0 to blockDim.x - 1
  - Combine the block offset with the local thread index to obtain a general thread index within the grid (global)
- 1 equation to map every thread to one vector element
  - i = blockIdx.x \* blockDim.x \* threadIdx.x
    - identifies thread's global index and uses it index the arrays
- Ceiling Division launches enough threads for every element
  - if N is not a multiple of block size, floor division doesn't launch enough blocks
  - Integer ceiling divsion: numBlocks = (N + blockSize - 1) / blockSize
    - Extra threads are expected and must be **guarded**(watdatmean)
      - Final block may contain threads with no valid element
        - AKA no data for certain launched threads
        - Ex: if global index at least N
        - NOT OPTIMIZATION
      - **Boundary Guard:** makes kernel correct for any N
        - Compute global index, then guard
        - if(i < N) z[i] = x[i] + y[i]
      - Let invalid threads exit safely

### Compilation Notes

- Host C/C++ targets the CPU toolchain
- Device code targets GPU instruction path
  - PTX: Parallel Thread Execution
  - SASS: Shader Assembly
    - native GPU machine instruction code
  - CUDA source code combines both roles
    - NVIDIA CUDA Compiler (NVCC) coordinates compilation and thinking
- **Function Qualifiers:** specify where code executes and who can call it
  - \__host__\: executes on host, called from host
  - \__global__\: executes on device, called from host
  - \__device__\: executes on device, called from device
    - Qualifier is part of the function's execution contract
- **Helper Functions:** can be compiled for both host and device
  - only a few of these functions
  - function must still use operations supported by each target
  - Use deliberately because duplicated compilation can produce target specific constraints
- **Kernel launches:** asynchronous wrt the host
  - GPU kernel calls are asynchronous by default
    - host continues after issuing a launch
    - useful for overlapping GPU and CPU computations
    - changes to timing and error optimizations
    - synchronize with ```cudaError_t cudaDeviceSynchronize()```
      - blocks progress until previous requested tasks are completed
      - Synchronization is only for correctness or measurement since it can destroy concurrency

### Measuring GPU Speedup

- **Define the Comparison**
- **Warm Up** before measuring steady-state perfromance
- **Account for asynchronous launches** with CUDA events or explicit synchronization
- **Include transfers and allocation** when they belong to the real workflow
- **Verify correctness** before trusting performance
- **Report the Context:** problem size, hardware, software, and methodology
