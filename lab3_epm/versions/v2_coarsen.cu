// v2: thread coarsening along a row (built on v1)
//
// Hypothesis: grid points in the same row share y (and every point shares
// z0), so dy*dy + dz*dz is the same for all of them. In v1 every thread
// recomputes it for every atom. If one thread handles COARSEN neighbouring
// points in a row, it computes dy*dy + dz*dz once per atom and reuses it
// COARSEN times. It also reads each atom from constant memory once for
// COARSEN points instead of once per point. Less arithmetic and fewer loads
// per output should mean a faster kernel.
//
// Layout in this version: thread t owns points 4t, 4t+1, 4t+2, 4t+3.
// (v3 changes this layout; see the note there about coalescing.)
//
// Cost: COARSEN sums and COARSEN x values live in registers, so register use
// per thread goes up, which can lower occupancy.

#include <cuda_runtime.h>
#include "cuda_check.h"
#include "coulomb.h"
#include "problem.h"

#define CHUNK 4000
#ifndef COARSEN
#define COARSEN 4
#endif
#define BX 16
#define BY 16

__constant__ float4 c_atoms[CHUNK];

static EpmProblem s_p;
static float4* s_atoms = nullptr;
static float* s_out = nullptr;

__global__ void potential_coarse(int n, float* out, int H, int W,
                                 float h, float z0, int first) {
  // First column this thread owns; it also owns the next COARSEN-1 columns.
  const int i0 = (blockIdx.x * blockDim.x + threadIdx.x) * COARSEN;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i0 >= W || j >= H) return;

  const float y = h * j;
  float x[COARSEN], sum[COARSEN];
  #pragma unroll
  for (int k = 0; k < COARSEN; k++) {
    x[k] = h * (i0 + k);   // may be past W on the last thread; not written
    sum[k] = 0.0f;
  }

  for (int a = 0; a < n; a++) {
    const float4 at = c_atoms[a];
    const float dy = y - at.y, dz = z0 - at.z;
    const float dyz2 = dy*dy + dz*dz;          // shared by all COARSEN points
    #pragma unroll
    for (int k = 0; k < COARSEN; k++) {
      const float dx = x[k] - at.x;
      sum[k] += coulomb_term(at.w, dx*dx + dyz2);
    }
  }

  float* row = out + (size_t)j * W;
  #pragma unroll
  for (int k = 0; k < COARSEN; k++) {
    const int i = i0 + k;
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
  // Each block now covers BX*COARSEN columns.
  const dim3 grid((s_p.W + BX * COARSEN - 1) / (BX * COARSEN), (s_p.H + BY - 1) / BY);
  for (int off = 0; off < s_p.na; off += CHUNK) {
    const int n = (s_p.na - off < CHUNK) ? (s_p.na - off) : CHUNK;
    CHECK(cudaMemcpyToSymbolAsync(c_atoms, s_atoms + off, n * sizeof(float4), 0,
                                  cudaMemcpyDeviceToDevice));
    potential_coarse<<<grid, block>>>(n, s_out, s_p.H, s_p.W, s_p.h, s_p.z0, off == 0);
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
