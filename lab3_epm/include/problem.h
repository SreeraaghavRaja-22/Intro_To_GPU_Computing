#pragma once
#include <cstddef>
#include <vector>
// Read-only. All device inputs and outputs are float32.
// atoms[4*a + {0,1,2,3}] = {x, y, z, q} for atom a (AoS).
// Grid (i,j) is at (h*i, h*j, z0); output is potential[j*W+i].
// Every output sums coulomb_term(q, dx*dx + dy*dy + dz*dz) over ALL atoms.
struct EpmProblem {
  int H, W, na;
  float h, z0;
  const float* atoms;
};
struct EpmData { std::vector<float> atoms; };
void epm_generate(int H, int W, int na, unsigned seed, EpmData& d);
EpmProblem epm_problem_from(int H, int W, int na, const EpmData& d);
void epm_reference(const EpmProblem& p, double* out);
struct EpmVerdict {
  double rel_l2;
  bool pass;
};
EpmVerdict epm_verify(const float* got, const double* ref, size_t count);

void epm_baseline_setup(const EpmProblem& p);
void epm_baseline_compute();
void epm_baseline_fetch(float* potential);
void epm_baseline_teardown();

// What must be implemented. setup/fetch/teardown are untimed.
// compute must evaluate the complete map from the stored atoms on EVERY call.
// setup may re-layout inputs, but must not precompute output contributions.
void epm_student_setup(const EpmProblem& p);
void epm_student_compute();
void epm_student_fetch(float* potential);
void epm_student_teardown();
