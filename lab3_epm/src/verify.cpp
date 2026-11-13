// Read-only. Reject non-finite values as well as excessive relative L2 error.
#include <cmath>
#include <limits>
#include "params.h"
#include "problem.h"

EpmVerdict epm_verify(const float* got, const double* ref, size_t count) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < count; i++) {
    if (!std::isfinite(got[i]) || !std::isfinite(ref[i]))
      return {std::numeric_limits<double>::infinity(), false};
    const double d = (double)got[i] - ref[i];
    num += d*d;
    den += ref[i]*ref[i];
  }
  const double err = den > 0.0 ? std::sqrt(num/den) : std::sqrt(num);
  return {err, std::isfinite(err) && err <= EPM_TOL_REL};
}
