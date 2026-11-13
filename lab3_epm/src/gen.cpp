// Read-only. Self-contained seeded atom generator; uses no molecular datasets.
#include "params.h"
#include "problem.h"

static unsigned xorshift(unsigned& s) {
  s ^= s << 13; s ^= s >> 17; s ^= s << 5;
  return s;
}
static float uni01(unsigned& s) {
  return (float)((xorshift(s) >> 8) * (1.0 / 16777216.0));
}
void epm_generate(int H, int W, int na, unsigned seed, EpmData& d) {
  unsigned s = seed * 2654435761u + 12345u;
  if (!s) s = 1;
  d.atoms.resize((size_t)na * 4);
  for (int a = 0; a < na; a++) {
    float* atom = d.atoms.data() + (size_t)4 * a;
    atom[0] = uni01(s) * EPM_GRID_SPACING * (W - 1);
    atom[1] = uni01(s) * EPM_GRID_SPACING * (H - 1);
    const float distance = EPM_MIN_Z_DISTANCE + 4.0f * uni01(s);
    const float sign = uni01(s) < 0.5f ? -1.0f : 1.0f;
    atom[2] = EPM_SLICE_Z + sign * distance;
    atom[3] = 2.0f * uni01(s) - 1.0f;
  }
}
EpmProblem epm_problem_from(int H, int W, int na, const EpmData& d) {
  return {H, W, na, EPM_GRID_SPACING, EPM_SLICE_Z, d.atoms.data()};
}
