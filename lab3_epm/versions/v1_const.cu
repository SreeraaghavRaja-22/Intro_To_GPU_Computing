// v1: atoms in constant memory
//
// Hypothesis: in the baseline, every thread in a warp reads the SAME atom at
// the SAME time from global memory. Constant memory has a cache built for
// exactly this: one read is broadcast to all 32 threads in the warp. If I move
// the atoms into constant memory, global traffic for atoms should drop to ~0,
// and the kernel should get faster.
//
// Catch: constant memory is only 64 KB. One atom = 4 floats = 16 B, so at
// most 4096 atoms fit. The hidden test can use more atoms, so compute() walks
// the atoms in chunks of CHUNK: copy a chunk into constant memory, launch the
// kernel, and add that chunk's partial sums into the output.
//
// Only change from v0: where the atoms live. Block size and the
// one-thread-per-point mapping are the same as the baseline.

#include <cuda_runtime.h>
#include "cuda_check.h"
#include "coulomb.h"
#include "problem.h"

#define CHUNK 4000                 // 4000 * 16 B = 64000 B (limit is 65536)
__constant__ float4 c_atoms[CHUNK];

static EpmProblem s_p;
static float4* s_atoms = nullptr;  // all atoms, kept on the device
static float* s_out = nullptr;

// first == 1: this is the first chunk, so overwrite the output.
// first == 0: add this chunk's contribution to what is already there.
__global__ void potential_const(int n, float* out, int H, int W,
                                float h, float z0, int first) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W || j >= H) return;
  const float x = h * i, y = h * j;
  float sum = 0.0f;
  for (int a = 0; a < n; a++) {
    const float4 at = c_atoms[a];  // .x .y .z = position, .w = charge
    const float dx = x - at.x, dy = y - at.y, dz = z0 - at.z;
    sum += coulomb_term(at.w, dx*dx + dy*dy + dz*dz);
  }
  const size_t idx = (size_t)j * W + i;
  out[idx] = first ? sum : out[idx] + sum;
}

void epm_student_setup(const EpmProblem& p) {
  s_p = p;
  // The host layout is x,y,z,q per atom, which is exactly one float4.
  CHECK(cudaMalloc((void**)&s_atoms, (size_t)p.na * sizeof(float4)));
  CHECK(cudaMemcpy(s_atoms, p.atoms, (size_t)p.na * sizeof(float4), cudaMemcpyHostToDevice));
  CHECK(cudaMalloc((void**)&s_out, (size_t)p.H * p.W * sizeof(float)));
}

void epm_student_compute() {
  const dim3 block(16, 16);
  const dim3 grid((s_p.W + 15) / 16, (s_p.H + 15) / 16);
  for (int off = 0; off < s_p.na; off += CHUNK) {
    const int n = (s_p.na - off < CHUNK) ? (s_p.na - off) : CHUNK;
    // Device-to-device copy into constant memory. Same stream as the kernel,
    // so the kernel will not start until the copy is done.
    CHECK(cudaMemcpyToSymbolAsync(c_atoms, s_atoms + off, n * sizeof(float4), 0,
                                  cudaMemcpyDeviceToDevice));
    potential_const<<<grid, block>>>(n, s_out, s_p.H, s_p.W, s_p.h, s_p.z0, off == 0);
    CHECK(cudaGetLastError());
  }
}

void epm_student_fetch(float* potential) {
  CHECK(cudaMemcpy(potential, s_out, (size_t)s_p.H * s_p.W * sizeof(float), cudaMemcpyDeviceToHost));
}

void epm_student_teardown() {
  CHECK(cudaFree(s_atoms));
  CHECK(cudaFree(s_out));
  s_atoms = nullptr;
  s_out = nullptr;
}
