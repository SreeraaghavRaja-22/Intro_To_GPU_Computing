# Lab 2 Undergraduate Section

## Part A: Matrix Multiplication

### Summary
This section of the lab was mainly oriented towards understanding how to implement matrix multiplication on the GPU. The purpose was to look at the speed up between different block sizes and understand how to work with HiPerGator on a larger project. 

### Build and Run Instructions
- Log into Node with:
    - bash gpu_alloc.sh
    - bash node_nav.sh <JOBID>
- Run all Tests with script run_tests.sh:
    - sbatch run_tests.sh
    - squeue -u gatorlink (gets jobid)
    - cat lab2_tests_jobid.out (prints to terminal)

### Results (from lab2_tests_46774194.out)

Results Table

| M | K | N | Result |
|---|---|---|---|
| 1 | 1 | 1 | PASS |
| 3 | 5 | 2 | PASS |
| 16 | 16 | 16 | PASS |
| 31 | 17 | 50 | PASS |
| 65 | 33 | 129 | PASS |

Compute Sanitizer Check:
Part = mm, M = 31, K = 17, N = 50, block = 16x16, grid = 4x2, PASS
Median kernel Time = 0.0886 ms over 22 tests (min 0.0840, max 0.1004)

### Timing Table

Note: The M, K, and N values are all multiples of 16 or 32 so that there aren't extra, unused threads that are allocated for these tests. 

| Block shape | M | K | N | Median kernel time (ms) | Min (ms) | Max (ms) | Reps |
|---|---|---|---|---|---|---|---|
| 16 × 16 | 2048 | 1024 | 1536 | 4.2813 | 3.5095 | 4.4462 | 22 (after 1 warm-up) |
| 32 × 8  | 2048 | 1024 | 1536 | 4.0432 | 3.4785 | 4.4506 | 22 (after 1 warm-up) |

**Speedup (32 × 8 over 16 × 16):** 4.2813 / 4.0432 = **1.06×** (the 32 x 8 is about 6% faster than 16 x 16)

### Environment (found in the record log)

GPU: NVIDIA L4,  NVCC Version: CUDA 13.2 (V13.2.78),  compiled with `nvcc -O2 -lineinfo -std=c++17`


### Write-Up

The timing differences that I noticed between the different block sizes is that the rectangular block size (32 x 8) had a faster median kernel time than the square block size (16 x 16). What's unique about the result is that the spread between the min and max time for the rectangular block size is larger than the spread of times for the square block case. This would suggest that the square block size has a more consistent kernel launch time. However, these results are so close together that the difference between the block sizes should be considered as marginal.

### Part B: Image Blur

### Summary:
Due to time contraints, I did not attmept this part of the lab. However, I will finish it in my own time when I am free. 

### AI Disclosure

I used Claude to help me understand how to complete certain parts of this lab. I had it help me understand how to randomly fill vectors with values to optimize my accuracy tests. The most significant use I had for it was to explain how to create a script to automatically run all of the tests. Additionally, I used it to help me understand how Cuda Events worked since I couldn't find resources for it in the class slides. Finally, I used it to help create the timing table in the README as well. Overall, I used Claude more for understanding and speeding up processes that were not directly tied to the material I had to learn from this class. 