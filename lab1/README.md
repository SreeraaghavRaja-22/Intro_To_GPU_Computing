# README

## NVCC Version
- nvcc: Copyright (c) 2005-2026 NVIDIA Corporation
Built on Thu_Mar_19_11:12:51_PM_PDT_2026
Cuda compilation tools, release 13.2, V13.2.78
Build cuda_13.2.r13.2/compiler.37668154_0
- 

## Build and Run Configurations 
- Build: nvcc -O2 -lineinfo vector_add.cu -o vector_add
- Run: ./vector_add 1000
- Run: ./vector_add 1011 | tee "results/n1011.txt" -- This was used for the majority of this lab
- Build: nvcc -O0 -g -G vector_add.cu -o vector_add_debug
- Run: compute-sanitizer --tool memcheck ./vector_add_debug 1000
