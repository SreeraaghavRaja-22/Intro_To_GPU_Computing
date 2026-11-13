// v4: shared-memory tiling of atoms instead of constant memory (built on v3)
//
// Hypothesis: v1-v3 need one kernel launch plus one copy into constant
// memory per 4000 atoms. An alternative from Chapter 5 is tiling: the block
// cooperatively loads TILE atoms from global memory into shared memory
// (one coalesced float4 load per thread), syncs, and then every thread reuses
// that tile. This needs a single launch for any atom count. Constant memory
// broadcasts a uniform read to the whole warp; shared memory also
// broadcasts when all threads read the same address, so per-atom read cost
// should be similar. I expect roughly the same speed as v3: maybe slightly
// faster from fewer launches, maybe slightly slower from the
// __syncthreads() barriers and the global loads.
//
// Only change from v3: where atoms are staged (shared instead of constant).

#include <cuda_runtime.h>
#include "cuda_check.h"
#include "coulomb.h"
#include "problem.h"

#ifndef COARSEN
#define COARSEN 4
#endif
#define BX 16
#define BY 16
#define TILE (BX * BY)   // one atom loaded per thread per tile

static EpmProblem s_p;
static float4* s_atoms = nullptr;
static float* s_out = nullptr;

__global__ void potential_shared(const float4* __restrict__ atoms, int na, float* out,
                                 int H, int W, float h, float z0) {
  __shared__ float4 s_at[TILE];

  const int tid = threadIdx.y * BX + threadIdx.x;
  const int i0 = blockIdx.x * BX * COARSEN + threadIdx.x;
  const int j = blockIdx.y * BY + threadIdx.y;
  // NO early return here: every thread must help load tiles and reach every
  // __syncthreads(), even if its own grid point is outside the map.

  const float y = h * j;
  float x[COARSEN], sum[COARSEN];
  #pragma unroll
  for (int k = 0; k < COARSEN; k++) {
    x[k] = h * (i0 + k * BX);
    sum[k] = 0.0f;
  }

  for (int base = 0; base < na; base += TILE) {
    // Same n for every thread in the block, so the loop below is uniform.
    const int n = (na - base < TILE) ? (na - base) : TILE;
    if (tid < n) s_at[tid] = atoms[base + tid];   // coalesced load
    __syncthreads();                              // tile is ready

    for (int a = 0; a < n; a++) {
      const float4 at = s_at[a];                  // broadcast read
      const float dy = y - at.y, dz = z0 - at.z;
      const float dyz2 = dy*dy + dz*dz;
      #pragma unroll
      for (int k = 0; k < COARSEN; k++) {
        const float dx = x[k] - at.x;
        sum[k] += coulomb_term(at.w, dx*dx + dyz2);
      }
    }
    __syncthreads();                              // done before overwrite
  }

  if (j >= H) return;
  float* row = out + (size_t)j * W;
  #pragma unroll
  for (int k = 0; k < COARSEN; k++) {
    const int i = i0 + k * BX;
    if (i < W) row[i] = sum[k];
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
  potential_shared<<<grid, block>>>(s_atoms, s_p.na, s_out, s_p.H, s_p.W, s_p.h, s_p.z0);
  CHECK(cudaGetLastError());
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
