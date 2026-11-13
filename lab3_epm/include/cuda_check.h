#pragma once
#include <cstdio>
#include <cstdlib>
// Read-only helper. Wrap every CUDA runtime call: CHECK(cudaMalloc(...));
#define CHECK(call)                                                          \
  do {                                                                       \
    cudaError_t e_ = (call);                                                 \
    if (e_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(e_),   \
              __FILE__, __LINE__);                                           \
      exit(2);                                                               \
    }                                                                        \
  } while (0)
