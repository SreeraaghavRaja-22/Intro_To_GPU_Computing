// Your implementation starts as a copy of the provided baseline.
// Edit this file and add .cu/.cuh files in src/ as needed.
// setup: allocate device memory and upload inputs (untimed).
//        Do not precompute contributions to the output.
// compute: calculate the entire map on every call (timed, device work only).
// fetch: copy the map back to the host (untimed).
// teardown: free your allocations (untimed).
// Use float32 arithmetic and the read-only coulomb_term helper.
// See include/problem.h and README.md for the full requirements.

#include <cuda_runtime.h>
#include "cuda_check.h"
#include "coulomb.h"
#include "problem.h"

static EpmProblem s_p;
static float *s_atoms = nullptr, *s_out = nullptr;

__global__ void potential_student(const float* atoms, int na, float* out,
                                  int H, int W, float h, float z0) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= W || j >= H) return;
  const float x = h * i, y = h * j;
  float sum = 0.0f;
  for (int a = 0; a < na; a++) {
    const float* atom = atoms + (size_t)4 * a;
    const float dx = x - atom[0], dy = y - atom[1], dz = z0 - atom[2];
    sum += coulomb_term(atom[3], dx*dx + dy*dy + dz*dz);
  }
  out[(size_t)j * W + i] = sum;
}
void epm_student_setup(const EpmProblem& p) {
  s_p = p;
  CHECK(cudaMalloc((void**)&s_atoms, (size_t)p.na * 4 * sizeof(float)));
  CHECK(cudaMemcpy(s_atoms, p.atoms, (size_t)p.na * 4 * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMalloc((void**)&s_out, (size_t)p.H * p.W * sizeof(float)));
}
void epm_student_compute() {
  const dim3 block(16, 16);
  const dim3 grid((s_p.W + 15) / 16, (s_p.H + 15) / 16);
  potential_student<<<grid, block>>>(s_atoms, s_p.na, s_out, s_p.H, s_p.W, s_p.h, s_p.z0);
  CHECK(cudaGetLastError());
}
void epm_student_fetch(float* potential) {
  CHECK(cudaMemcpy(potential, s_out, (size_t)s_p.H * s_p.W * sizeof(float), cudaMemcpyDeviceToHost));
}
void epm_student_teardown() {
  CHECK(cudaFree(s_atoms));
  CHECK(cudaFree(s_out));
  s_atoms = s_out = nullptr;
}
