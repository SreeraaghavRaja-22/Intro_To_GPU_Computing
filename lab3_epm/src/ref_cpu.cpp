// Read-only. Double-precision reference for the potential on a 2D slice.
#include <cmath>
#include "problem.h"

void epm_reference(const EpmProblem& p, double* out) {
#pragma omp parallel for
  for (int j = 0; j < p.H; j++) {
    for (int i = 0; i < p.W; i++) {
      const double x = (double)p.h * i, y = (double)p.h * j;
      double sum = 0.0;
      for (int a = 0; a < p.na; a++) {
        const float* atom = p.atoms + (size_t)4 * a;
        const double dx = x - atom[0], dy = y - atom[1];
        const double dz = (double)p.z0 - atom[2];
        sum += (double)atom[3] / std::sqrt(dx*dx + dy*dy + dz*dz);
      }
      out[(size_t)j * p.W + i] = sum;
    }
  }
}
