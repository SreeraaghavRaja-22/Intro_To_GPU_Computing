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

- 
