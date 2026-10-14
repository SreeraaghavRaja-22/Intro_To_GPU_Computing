#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Macro to make cuda runtime errors visible
inline void cuda_check(cudaError_t e){
    if(e != cudaSuccess){
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

#define CUDA_CHECK(call) cuda_check(call)