#include <cuda_runtime.h> 
#include <cmath> 
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <random>


// Helper function that reports the location of any CUDA error after aborting the program
void cuda_check(cudaError_t e, const char* file, int line){
    if(e != cudaSuccess){
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(e));
        std::exit(1);
    }
}

#define CUDA_CHECK(call) cuda_check((call), __FILE__, __LINE__)


// CPU reference model
void matmul_cpu(const float* A, const float* B, float* C, int M, int K, int N){
    for(int row = 0; row < M; ++row){
        for(int col = 0; col < N; ++col){
            float sum = 0.0f;
            for(int k = 0; k < K; ++k){
                sum += A[row *  K + k] * B[k * N + col];
            }
            C[row * N + col] = sum;
        }
    }
}

// CUDA Kernel
__global__ void matmul(const float* A, const float* B, float* C, int M, int K, int N){
    int row = blockIdx.y * blockDim.y  + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < M && col < N){
        float sum = 0.0f; 
        for(int k = 0; k < K; ++k){
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Element-wise Verifier
bool close_enough(float got, float ref){
    if(!std::isfinite(got)) return false;

    float tolerance = 1.0e-3f + 1.0e-3f * std::fabs(ref);
    return std::fabs(got - ref) <= tolerance;
}



// Verification function with tolerances 
// needs to have element count to get the values from both models
bool verify(const float* got, const float* ref, int M, int N){
    for(int i = 0; i < M * N; ++i){
        if(!(close_enough(got[i], ref[i]))){
            std::fprintf(stderr, "Index Mismatch: Index - %d (Row %d, Col %d), Got: %f, Ref: %f, Error: %e\n", i, i / N, i % N , got[i], ref[i], std::fabs(got[i] - ref[i]));
            return false;        
        } 
    }
    return true; 
}

int main(int argc, char** argv){
}