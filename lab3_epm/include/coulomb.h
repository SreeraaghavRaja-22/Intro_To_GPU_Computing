#pragma once
#include <cuda_runtime.h>

// Read-only helper: use this function for every atom's contribution to a grid point.
// Keep this calculation unchanged.
// Generated inputs ensure d2 > 0. The CPU reference evaluates q / sqrt(d2) in double precision.
__device__ __forceinline__ float coulomb_term(float q, float d2) {
  return q * rsqrtf(d2);
}
