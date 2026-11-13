// v5: 2D coarsening, columns AND rows (built on v3)
//
// Hypothesis: v3 reuses dy^2 + dz^2 across the COARSEN points in a row, but
// each point still computes its own dx^2. Points in the same column share dx
// too. If one thread owns CX columns x CY rows, then per atom it needs only
// CX values of dx^2 and CY values of dy^2 + dz^2, and each of the CX*CY points
// is one add: d2 = dx2[k] + dyz2[m]. That removes about one multiply per
// point and reads each atom once for CX*CY points instead of CX.
//
// Caveat: every point still needs its own rsqrtf inside coulomb_term, and
// that special-function work does not shrink. If rsqrtf is the real limit,
// this change may not help, and the extra registers (CX*CY sums) could even
// lower occupancy and make it slower. This is a good test of what the
// bottleneck actually is.
//
// Only change from v3: each thread also covers CY rows (strided by BY).

#include <cuda_runtime.h>
#include "cuda_check.h"
#include "coulomb.h"
#include "problem.h"

#define CHUNK 4000
#ifndef CX
#define CX 4
#endif
#ifndef CY
#define CY 2
#endif
#define BX 16
#define BY 16

__constant__ float4 c_atoms[CHUNK];

static EpmProblem s_p;
static float4* s_atoms = nullptr;
static float* s_out = nullptr;

__global__ void potential_2d(int n, float* out, int H, int W,
                             float h, float z0, int first) {
  const int i0 = blockIdx.x * BX * CX + threadIdx.x;   // columns i0 + k*BX
  const int j0 = blockIdx.y * BY * CY + threadIdx.y;   // rows    j0 + m*BY
  if (i0 >= W || j0 >= H) return;

  float x[CX], y[CY], sum[CY][CX];
  #pragma unroll
  for (int k = 0; k < CX; k++) x[k] = h * (i0 + k * BX);
  #pragma unroll
  for (int m = 0; m < CY; m++) {
    y[m] = h * (j0 + m * BY);
    #pragma unroll
    for (int k = 0; k < CX; k++) sum[m][k] = 0.0f;
  }

  for (int a = 0; a < n; a++) {
    const float4 at = c_atoms[a];
    const float dz = z0 - at.z;
    const float dz2 = dz * dz;
    float dx2[CX], dyz2[CY];
    #pragma unroll
    for (int k = 0; k < CX; k++) { const float dx = x[k] - at.x; dx2[k] = dx * dx; }
    #pragma unroll
    for (int m = 0; m < CY; m++) { const float dy = y[m] - at.y; dyz2[m] = dy * dy + dz2; }
    #pragma unroll
    for (int m = 0; m < CY; m++) {
      #pragma unroll
      for (int k = 0; k < CX; k++)
        sum[m][k] += coulomb_term(at.w, dx2[k] + dyz2[m]);
    }
  }

  #pragma unroll
  for (int m = 0; m < CY; m++) {
    const int j = j0 + m * BY;
    if (j >= H) break;                 // later rows are further out
    float* row = out + (size_t)j * W;
    #pragma unroll
    for (int k = 0; k < CX; k++) {
      const int i = i0 + k * BX;
      if (i < W) row[i] = first ? sum[m][k] : row[i] + sum[m][k];
    }
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
  const dim3 grid((s_p.W + BX * CX - 1) / (BX * CX), (s_p.H + BY * CY - 1) / (BY * CY));
  for (int off = 0; off < s_p.na; off += CHUNK) {
    const int n = (s_p.na - off < CHUNK) ? (s_p.na - off) : CHUNK;
    CHECK(cudaMemcpyToSymbolAsync(c_atoms, s_atoms + off, n * sizeof(float4), 0,
                                  cudaMemcpyDeviceToDevice));
    potential_2d<<<grid, block>>>(n, s_out, s_p.H, s_p.W, s_p.h, s_p.z0, off == 0);
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
