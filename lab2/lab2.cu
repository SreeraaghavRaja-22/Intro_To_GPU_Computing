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

    // ------------- Initialization -------------- //
    // Create default input values if the flags aren't given
    const char* part = "mm";
    // default M, K, and N values
    int M = 31, K = 17, N = 50;
    // default block size is 16 x 16
    int bx = 16, by = 16; 

    // loop to parse through each part of the compile argument -- skip the program name argv[0]
    // argc is the number of indices
    for(int i = 1; i < argc; ++i){
        if(strcmp(argv[i], "--M") == 0 && i + 1 < argc){
            M = atoi(argv[i + 1]);      // the value is the next entry in compile argument
            ++i;                        // skip value to not read it as a flag
        }
        else if(strcmp(argv[i], "--K") == 0 && i + 1 < argc){
            K = atoi(argv[i + 1]);      // the value is the next entry in compile argument
            ++i;                        // skip value to not read it as a flag
        }
        else if(strcmp(argv[i], "--N") == 0 && i + 1 < argc){
            N = atoi(argv[i + 1]);      // the value is the next entry in compile argument
            ++i;                        // skip value to not read it as a flag
        }
        else if(strcmp(argv[i], "--bx") == 0 && i + 1 < argc){
            bx = atoi(argv[i + 1]);     // the value is the next entry in compile argument
            ++i;                        // skip value to not read it as a flag
        }
        else if(strcmp(argv[i], "--by") == 0 && i + 1 < argc){
            by = atoi(argv[i + 1]);     // the value is the next entry in compile argument
            ++i;                        // skip value to not read it as a flag
        }
        else if(strcmp(argv[i], "--part") == 0 && i + 1 < argc){
            part = argv[i + 1];         // the value is the next entry in compile argument
            ++i;                        // skip value to not read it as a flag
        }
        else {
            fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
            return 1;
        }

        //  If else branch to validate inputs are all positive and max number of threads is less than or equal to 1024
        if (M <= 0 || K <= 0 || N <= 0) {
                fprintf(stderr, "Error: M, K, and N must be positive (got M=%d, K=%d, N=%d)\n", M, K, N);
                return 1;
            }
        else if (bx <= 0 || by <= 0) {
            fprintf(stderr, "Error: block dimensions must be positive (got %dx%d)\n", bx, by);
            return 1;
        }
        else if (bx * by > 1024) {
            fprintf(stderr, "Error: block has %d threads; CUDA allows at most 1024\n", bx * by);
            return 1;
        }
    }

    // Allocate the size for each vector
    size_t bytesA = M * K * sizeof(float);
    size_t bytesB = K * N * sizeof(float);
    size_t bytesC = M * N * sizeof(float);

    // Check to ensure that these values do not provide any errors
    float* A_d = nullptr;
    float* B_d = nullptr;
    float* C_d = nullptr; 

    // Create four host buffers 
    std::vector<float> A_h(static_cast<size_t>(M) * K);
    std::vector<float> B_h(static_cast<size_t>(K) * N);
    std::vector<float> got(static_cast<size_t>(M) * N);
    std::vector<float> ref(static_cast<size_t>(M) * N);

    // Fill A and B with random values in [-1, 1] (this is to avoid all-one passes or all 0 passes for wrong Kernels)
    std::mt19937 gen(42); // fixed seed
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for(size_t i = 0; i < A_h.size(); ++i) A_h[i] = dist(gen);
    for(size_t i = 0; i < B_h.size(); ++i) B_h[i] = dist(gen);

    CUDA_CHECK(cudaMalloc(&A_d, bytesA));
    CUDA_CHECK(cudaMalloc(&B_d, bytesB));
    CUDA_CHECK(cudaMalloc(&C_d, bytesC));   
    
    // get raw pointers to first elements for A_h and B_h since they are vectors
    CUDA_CHECK(cudaMemcpy(A_d, A_h.data(), bytesA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(B_d, B_h.data(), bytesB, cudaMemcpyHostToDevice));

    // create block shape and grid size
    dim3 block(bx, by);
    dim3 grid(((N + block.x -1) / block.x), (M + block.y - 1) / block.y);

    // ------------------ Kernel Launch ----------------- //
    // launch kernel
    matmul<<<grid, block>>>(A_d, B_d, C_d, M, K, N);

    // Check Launch to ensure there are no hidden errors
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy C back to host
    CUDA_CHECK(cudaMemcpy(got.data(), C_d, bytesC, cudaMemcpyDeviceToHost));

    // Run the CPU reference
    matmul_cpu(A_h.data(), B_h.data(), ref.data(), M, K, N);

    // Compare the CPU and GPU results
    bool pass = verify(got.data(), ref.data(), M, N);

    // print out results
    std::printf("Part = %s, M = %d, K = %d, N = %d, block = %dx%d, grid = %ux%u, %s\n", part, M, K, N, bx, by, grid.x, grid.y, pass ? "PASS" : "FAIL");


    // --------- Timing Study -------------- //
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // warm up launch
    matmul<<<grid, block>>>(A_d, B_d, C_d, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    const int tests = 22;
    std::vector<float> times(tests);
    for (int t = 0; t < tests; ++t){
        CUDA_CHECK(cudaEventRecord(start));
        matmul<<<grid, block>>>(A_d, B_d, C_d, M, K, N);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventElapsedTime(&times[t], start, stop)); // get the time in ms
    }

    // sort the times
    std::sort(times.begin(), times.end());

    // find median time, more complicated for even number of tests
    float median = (tests % 2) ? times[tests / 2] : 0.5f * (times[tests / 2 - 1] + times[tests / 2]);

    // Print the median time
    std::printf("Median kernel Time = %.4f ms over %d tests (min %.4f, max %.4f)\n", median, tests, times.front(), times.back());

    // Destory the Cuda events now
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    
    // free the allocated memory
    CUDA_CHECK(cudaFree(A_d));
    CUDA_CHECK(cudaFree(B_d));
    CUDA_CHECK(cudaFree(C_d));

    return pass ? 0 : 1;
}