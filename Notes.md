# Notes

## 8/24 (Module 1)

### Projects

**Project 1:** Make a correct GPU kernel run; Basic GPU Programming (Vector Addition) - 5%

**Project 2:** Design and Optimize Matrix - 10%

**Project 3:** Profile and improve a scientific or AI workload - 10%

**Project 4:** Communicate and scale across multiple GPUs - 5%

**Project 5 (Final Project):** Integrate, evaluate, explain, and defend a complete application - 5%

Show and explain output, performance measurement (avg latency, problem size, hardware version, etc.), figures, tables (bars, curves, etc.) ^ discuss what you have learned

### Grades

**Attendance Quiz:** in class discussion or in class programming -> HW may be less important than class discussion

**Final Exam:**

- On paper
- up to 4 double-sided handwritten A4 paper notes (must be handwritten)
- No electronics
- No collaboration

**GPU Toolchain:**

- NVIDIA CUDA Toolkit and nvcc
- CUDA Libraries, including cuBLAS and cuDNN
- GNU C/C++ and IDE
- NVIDIA Nsight Systems and Nsight Compute
- HiPerGator access; no hw purchase required

**Thinking Response (QA):**

- Moore's law
- Clock frequency in 2000s flatten because the heat and power has come in and then cooling became a bigger problem
- 2005 frequency stopped due to power wall because higher frequency increases energy and heat
- Single thread performance growth slowed but architecture and compiler advances continue but the old frequency driven trajector was gone

## 8/26 (Module 1)

### Thinking Response (QA)

**Why did parallel computing become mainstream?**

- **Dennard Scaling Law:**smaller transistors enabled higher clock frequencies while keeping power density constant
  - Under same power, smaller transistors give better clock frequency until 2000s (freq flattened out)
  - Didn't need parallelism because sequential code doubled every so often
    - However, parallelism is now the only solution
  - *More transistors don't mean proportionally sequential performance increase*
- **Metrics:** transistors, frequency, power all slowed down
  - Single-threaded performance growth slowed AKA **Single-threaded performance bump**
  - Solution: bigger cache, bigger loading, compute closer, overall layout of chips to help -> small improvements and bounded by the physical limitation
  - Multicore and Parallel Computing became more important around 2005 when the number of logical cores started increasing
    - Each single core has a lower frequency so heat is down and each can divide the threads to automatically do concurrent tasks
    - CPU -> Latency Oriented GPU -> Throughput Oriented
    - CPU -> More complex control, Cache, ALU (smaller amount)
    - GPU -> Simpler control, Cache, ALU (a lot)

### 8/28 (Module 1)

- DLRM (Deep learning based Recommendation Systems) -> GPU Computing and HPC behind it
- [TOP500 Link](https://top500.org/)
  - Updates every 5 months (June and December)
  - Run Evals to see peak performance they can achieve in terms of FLOPS

#### Parallel Computing Pitfall

- Seq Exec T: 100s Parallel Fraction: 90% parallelizable is 1000x faster
- New time = (1 - 0.9) x 100 + (0.9 x 100) / 1000 = 10.09s
- Overall Speedup: 100/10.09 = 9.91x
- Sequential Portion that slows down the process: Initialize the environment, global states
- Moore's Law and Dennard Scaling

#### Amdahl's Scaling

- For an application with:
  - t = original sequential execution time
  - p = fraction of execution that is parallelizable
  - s = speedup achieved on parallelizable part
- New time: ((1-p) + p/s) * t
- Overall Speedup: 1 / ((1 - p) + p/s)
  - As s approaches infinity, maximum speedup approaches 1/(1-p)
  - Max Speedup < 1/(1 - p)
- Reduce transfers, synch, launch overhead, and serial setup

### 9/2 (Module 2)

- **Question for Current Module**
  - Where does data parallelism appear in a serial loop?
  - How do the CPU and GPU divide responsiblity?
  - How are thousands of threads organized?
  - How does one thread find its data element?
  - How do we handle boundaries, errors, and synchronous execution?
  - How should we measure whether the program is actually faster?

- **Data Parallelism**
  - **Task Parallelism (Different Operation)**
    - Different operations performed on same or different data elements
    - Usually, a modest number of tasks unleashing a modest amount of parallelism
  - **Data Parallelism (Same Operation, many elements)**
    - The same operation applies to many different data elements
    - Potentially massive amounts of data unleashing massive amounts of parallelism
    - GPU Programming often begins by finding this repeated operation
    - GPU programming is typically **data parallelism**
- **Cude Program Coordinates Two Processors and Two Memory Spaces**
  - Host - CPU
    - Runs sequential control and launches work
  - Device - GPU
    - Runs many parallel threads
    - in explicit-memory model, data must be moved between host and device memory
  - The CPU and GPU have separate memories and cannot access each others' memories
    - There is an advanced feature in modern systems that can
- **First CUDA Program follows a five-stage lifecycle**
  - Allocate device memory
  - Copy inputs host -> device
  - Launch the computaiton **kernel** on the GPU
  - Copy results device -> host
  - Free device memory
  - **Correctness requires every stage**
- **cudeMalloc and cudaFree**
  - `cudaMalloc` receives the address of the device pointer and a size in bytes
  - `cudaFree` releases the device allocation
  - Always calculate bytes explicitly: N * size (elements type)

```cpp
size_t bytes = N * sizeof(float);
cudaMalloc((void**)&x_d, bytes);

// use x_d on the device

cudaFree(x_d)
```

```cpp
// Actual Function Signature 
cudeError_t cudaMalloc(void **devPtr, size_t size);
```

- devPtr: pointer to pointer to allocated device memory
- size: requestion allocation size in bytes, typical errors could be OOM errors

```cpp
cudaError_t cudaFree(void *devPtr);
```

- devPtr: pointer to device memory to free if code is complex then double free could happen

```cudeMemcpy(destination, source, byte count, direction)
```

- HostToDevice copies input to the GPU
- DeviceToHost return results to the CPU
- Current research is on how to balance/partition to help with parallelism
- There is a hierarchy to be able to scale
