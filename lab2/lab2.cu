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
void vecadd_cpu(const float* x, const float* y, float* z, int N){
    for(int i = 0; i < N; ++i) z[i] = x[i] + y[i];
}



// Verification function with tolerances 
// Used inline here to tell linker that multiple copies of the function across multiple files are the same definition
bool verify(const float* got, const float* ref, int N, float tol){
    for(int i = 0; i < N; ++i){
        if(std::fabs(got[i] - ref[i]) > tol) return false;
    }
    return true; 
}

// Macro to make cuda runtime errors visible
void cuda_check(cudaError_t e){
    if(e != cudaSuccess){
        std::fprintf(stderr, "CUDA: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

#define CUDA_CHECK(call) cuda_check(call)

__global__ void vecadd_kernel(const float* x, const float* y, float* z, int N){
    // in built c++ variables
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N) z[i] = x[i] + y[i];
}
 
int main(int argc, char** argv){
    // num Threads for the CPU just a small fixed size
    int T = 8;
    // Set problem size 
    int N = (argc > 1) ? std::atoi(argv[1]) : 1000; 
    // Set byte size of a single array
    int bytes = size_t(N) * sizeof(float);
    // allocate the space for x and y arrays
    std::vector<float> x(N, 1.0f), y(N, 2.0f);
    std::vector<float> got(N), ref(N), ref_mt(N);

    // start mt cpu timer
    auto cpu_mt_start = std::chrono::high_resolution_clock::now();
    // call multithreaded add function
    vecadd_cpu_mt(x.data(), y.data(), ref_mt.data(), N, T);
    // end mt cpu timer
    auto cpu_mt_end = std::chrono::high_resolution_clock::now();

    // calculate mt cpu running time
    double cpu_mt_delta_ms = std::chrono::duration<double, std::milli>(cpu_mt_end - cpu_mt_start).count();

    // Start the cpu timer -- use auto to simplify the typing 
    auto cpu_start = std::chrono::high_resolution_clock::now();
    // call the cpu add function
    vecadd_cpu(x.data(), y.data(), ref.data(), N);
    // End the cpu timer
    auto cpu_end = std::chrono::high_resolution_clock::now();

    // calculate cpu running time
    double cpu_delta_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // Allocate device memory for x, y, and z
    float *d_x, *d_y, *d_z;
    CUDA_CHECK(cudaMalloc(&d_x, bytes));
    CUDA_CHECK(cudaMalloc(&d_y, bytes));
    CUDA_CHECK(cudaMalloc(&d_z, bytes));


    // Application timing 
    auto app_start = std::chrono::high_resolution_clock::now();

    // Copy x and y from host to device memory
    CUDA_CHECK(cudaMemcpy(d_x, x.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y.data(), bytes, cudaMemcpyHostToDevice));

    // Thread Code
    // Already in gpu.cu

    // Call the Kernel with the specific launch configuration
    const int numThreadsPerBlock = 256; 
    const int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;

    // Warm Up Program before collecting time measurements (Had to search this up)
    vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(d_x, d_y, d_z, N);
    // catch launch errors
    CUDA_CHECK(cudaGetLastError());
    // Synchronize for correctness and accurate GPU timing 
    CUDA_CHECK(cudaDeviceSynchronize());
    // End of Warm Up


    // Kernel Only Timing 
    auto kern_start = std::chrono::high_resolution_clock::now();
    vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(d_x, d_y, d_z, N);

    // catch launch errors
    CUDA_CHECK(cudaGetLastError());

    // Synchronize for correctness and accurate GPU timing 
    CUDA_CHECK(cudaDeviceSynchronize());

    // end of kernel timing
    auto kern_end = std::chrono::high_resolution_clock::now();


    // Copy data back to host 
    CUDA_CHECK(cudaMemcpy(got.data(), d_z, bytes, cudaMemcpyDeviceToHost));

    auto app_end = std::chrono::high_resolution_clock::now();

    // get timing delta in ms for GPU Application and Kernel
    // Application also includes the memory copying
    double kernel_delta_ms = std::chrono::duration<double, std::milli>(kern_end - kern_start).count();
    double app_delta_ms = std::chrono::duration<double, std::milli>(app_end - app_start).count();

    // verify correctness of data 
    float tol = 1e-5; // set tolerance for comparison
    bool pass = verify(got.data(), ref.data(), N, tol);

    // print pass or fail based on the verification results 
    std::printf("N=%d\nThreads=%d\nBlocks=%d\nThreads/Block = %d\nTolerance = %f\nCPU_Result=%d\nGPU_RESULT=%d\nPASS/FAIL: %s\n", N, numBlocks * numThreadsPerBlock, numBlocks, numThreadsPerBlock, tol, ref.data(), got.data(), pass ? "PASS" : "FAIL");
    
    std::printf("Timing Results\nCPU Time = %f\nCPU_MT Time = %f\nGPU Kernel Time = %f\nGPU Application Time=%f\n", cpu_delta_ms, cpu_mt_delta_ms, kernel_delta_ms, app_delta_ms);
    // free the allocated memory 
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));

    // return 0 or 1 based on pass condition
    return pass ? 0 : 1;
}