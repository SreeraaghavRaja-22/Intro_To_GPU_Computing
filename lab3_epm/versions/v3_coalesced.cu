// v3: coalesced coarsening layout (built on v2)
//
// Hypothesis: in v2, thread t owns columns 4t..4t+3. When the warp does its
// k-th store, neighbouring threads write addresses 4 floats (16 B) apart, so
// one store instruction touches 4x more memory segments than it needs to.
// If instead thread t owns columns t, t+BX, t+2*BX, t+3*BX (a stride of the
// block width), then on every store neighbouring threads write neighbouring
// addresses, which is fully coalesced. Same amount of math as v2, cleaner
// memory pattern. Writes are a small part of the total work here (each
// output is written once per chunk, but each output does ~4000 atom
// iterations), so I expect a small gain, possibly within noise.
//
// Only change from v2: which columns each thread owns.

#include <cuda_runtime.h>
#include "cuda_check.h"
#include "coulomb.h"
#include "problem.h"

#define CHUNK 4000
#ifndef COARSEN
#define COARSEN 4
#endif
#ifndef BX
#define BX 16
#endif
#ifndef BY
#define BY 16
#endif

__constant__ float4 c_atoms[CHUNK];

static EpmProblem s_p;
static float4* s_atoms = nullptr;
static float* s_out = nullptr;

__global__ void potential_coalesced(int n, float* out, int H, int W,
                                    float h, float z0, int first) {
  // The block covers BX*COARSEN columns. Thread tx owns
  // i0, i0 + BX, i0 + 2*BX, ... inside that range.
  const int i0 = blockIdx.x * BX * COARSEN + threadIdx.x;
  const int j = blockIdx.y * BY + threadIdx.y;
  if (i0 >= W || j >= H) return;   // if i0 is out, every later column is too

  const float y = h * j;
  float x[COARSEN], sum[COARSEN];
  #pragma unroll
  for (int k = 0; k < COARSEN; k++) {
    x[k] = h * (i0 + k * BX);
    sum[k] = 0.0f;
  }

  for (int a = 0; a < n; a++) {
    const float4 at = c_atoms[a];
    const float dy = y - at.y, dz = z0 - at.z;
    const float dyz2 = dy*dy + dz*dz;
    #pragma unroll
    for (int k = 0; k < COARSEN; k++) {
      const float dx = x[k] - at.x;
      sum[k] += coulomb_term(at.w, dx*dx + dyz2);
    }
  }

  float* row = out + (size_t)j * W;
  #pragma unroll
  for (int k = 0; k < COARSEN; k++) {
    const int i = i0 + k * BX;
    if (i < W) row[i] = first ? sum[k] : row[i] + sum[k];
  }
}

void epm_student_setup(const EpmProblem& p) {
  s_p = p;
  CHECK(cudaMalloc((void**)&s_atoms, (size_t)p.na * sizeof(float4)));
  CHECK(cudaMemcpy(s_atoms, p.atoms, (size_t)p.na * sizeof(float4), cudaMemcpyHostToDevice));
  CHECK(cudaMalloc((void**)&s_out, (size_t)p.H * p.W * sizeof(float)));
}

void epm_student_compute() {
  const dim3 block(BX, BY);
  const dim3 grid((s_p.W + BX * COARSEN - 1) / (BX * COARSEN), (s_p.H + BY - 1) / BY);
  for (int off = 0; off < s_p.na; off += CHUNK) {
    const int n = (s_p.na - off < CHUNK) ? (s_p.na - off) : CHUNK;
    CHECK(cudaMemcpyToSymbolAsync(c_atoms, s_atoms + off, n * sizeof(float4), 0,
                                  cudaMemcpyDeviceToDevice));
    potential_coalesced<<<grid, block>>>(n, s_out, s_p.H, s_p.W, s_p.h, s_p.z0, off == 0);
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
