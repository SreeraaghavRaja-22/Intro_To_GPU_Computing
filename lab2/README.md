# Lab 2 Undergraduate Section

## Part A: Matrix Multiplication

### Summary
This section of the lab was mainly oriented towards understanding how to implement matrix multiplication on the GPU. The purpose was to look at the speed up between different block sizes and understand how to work with HiPerGator on a larger project. 

### Build and Run Instructions
- Log into Node with:
    - bash gpu_alloc.sh
    - bash node_nav.sh <JOBID>
- Build manually:
    - module load cuda
    - nvcc -O2 -lineinfo -std=c++17 lab2.cu -o lab2
    - ./lab2 --part mm --M 31 --K 17 --N 50
    - ./lab2 --part blur --input images/irr_grayscale_test_image.pgm --R 2
- Run all Tests with script run_tests.sh:
    - sbatch run_tests.sh
    - squeue -u gatorlink (gets jobid)
    - cat lab2_tests_jobid.out (prints to terminal)

### Results (from lab2_tests_<jobid>.out)

Results Table

| M | K | N | Result |
|---|---|---|---|
| 1 | 1 | 1 | PASS |
| 3 | 5 | 2 | PASS |
| 16 | 16 | 16 | PASS |
| 31 | 17 | 50 | PASS |
| 65 | 33 | 129 | PASS |

Compute Sanitizer Check (M = 31, K = 17, N = 50): PASS, ERROR SUMMARY: 0 errors

### Timing Table

Note: The M, K, and N values are all multiples of 16 or 32 so that there aren't extra, unused threads that are allocated for these tests. 

| Block shape | M | K | N | Median kernel time (ms) | Min (ms) | Max (ms) | Reps |
|---|---|---|---|---|---|---|---|
| 16 × 16 | 2048 | 1024 | 1536 | 4.2404 | 3.5256 | 4.4012 | 22 (after 1 warm-up) |
| 32 × 8  | 2048 | 1024 | 1536 | 4.1359 | 3.5021 | 4.4513 | 22 (after 1 warm-up) |

**Speedup (32 × 8 over 16 × 16):** 4.2404 / 4.1359 = **1.03×** (the 32 x 8 is about 3% faster than 16 x 16)

### Environment (found in the record log)

GPU: NVIDIA L4,  NVCC Version: CUDA 13.2 (V13.2.78),  compiled with `nvcc -O2 -lineinfo -std=c++17`


### Write-Up

The timing differences that I noticed between the different block sizes is that the rectangular block size (32 x 8) had a faster median kernel time than the square block size (16 x 16). In the 32 x 8 block, each warp of 32 threads covers one row of 32 consecutive columns, so its reads of B and writes to C are contiguous, while in the 16 x 16 block each warp is split across two rows. The spread between the min and max times is similar for both block sizes. However, these results are so close together that the difference between the block sizes should be considered as marginal.

## Part B: Image Blur

### Summary:
This part deals with image blurring and focuses on the algorithm for that and compares the speedup using the median time of the kernel launch for different block sizes.

### Results Table

| Image | Size (W × H) | R | Result |
|---|---|---|---|
| test_1x1.pgm | 1 × 1 | 1 | PASS |
| test_3x5.pgm | 3 × 5 | 1 | PASS |
| test_17x19.pgm | 17 × 19 | 2 | PASS |
| irr_grayscale_test_image.pgm | 899 × 719 | 2 | PASS |

Compute Sanitizer Check (irr_grayscale_test_image.pgm, R = 2): PASS, ERROR SUMMARY: 0 errors

### Timing Table
| Block shape | Image | Size (W × H) | R | Grid | Median kernel time (ms) | Min (ms) | Max (ms) | Reps |
|---|---|---|---|---|---|---|---|---|
| 16 × 16 | irr_grayscale_test_image.pgm | 899 × 719 | 2 | 57 × 45 | 0.0236 | 0.0225 | 0.0256 | 22 (after 1 warm-up) |
| 32 × 8  | irr_grayscale_test_image.pgm | 899 × 719 | 2 | 29 × 90 | 0.0236 | 0.0234 | 0.0254 | 22 (after 1 warm-up) |

**Speedup (32 × 8 over 16 × 16):** 0.0236 / 0.0236 ≈ **1.00×** (no measurable difference)

### Write-Up

There was no measurable difference between the time that the kernel ran for both block sizes. This means that the image blur program is unaffected by block size.

### AI Disclosure

I used Claude to help me understand how to complete certain parts of this lab. I had it help me understand how to randomly fill vectors with values to optimize my accuracy tests. The most significant use I had for it was to explain how to create a script to automatically run all of the tests. Additionally, I used it to help me understand how Cuda Events worked since I couldn't find resources for it in the class slides. Finally, I used it to help create the timing table in the README as well. Overall, I used Claude more for understanding and speeding up processes that were not directly tied to the material I had to learn from this class. 

Claude also generated code that I reviewed and integrated: the run_tests.sh script, the CUDA event timing block, the argument parsing skeleton and input validation, the random fill and host buffers, the file/line CUDA_CHECK helper, fixes to my verifier print statements, the run_mm/run_blur structure, most of run_blur, and the try/catch for the blur call. I verified everything with the Part A and Part B public tests (all PASS) and Compute Sanitizer (0 errors for both parts).