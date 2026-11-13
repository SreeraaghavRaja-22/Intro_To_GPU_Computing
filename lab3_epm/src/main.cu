// Read-only driver. Usage: ./epm H W atoms seed [options]
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "params.h"
#include "problem.h"

struct Impl {
  const char* name;
  void (*setup)(const EpmProblem&);
  void (*compute)();
  void (*fetch)(float*);
  void (*teardown)();
};
static bool parse_uint(const char* s, unsigned long long max, unsigned long long& v) {
  if (!s[0] || s[0] < '0' || s[0] > '9') return false;
  char* end = nullptr;
  errno = 0;
  v = std::strtoull(s, &end, 10);
  return !errno && *end == '\0' && v <= max;
}
static bool positive_int(const char* s, int& v) {
  unsigned long long n = 0;
  if (!parse_uint(s, INT_MAX, n) || n == 0) return false;
  v = (int)n;
  return true;
}
static double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); i++) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i]))
      return std::numeric_limits<double>::infinity();
    const double d = (double)a[i] - b[i];
    num += d*d;
    den += (double)b[i]*b[i];
  }
  return den > 0.0 ? std::sqrt(num/den) : std::sqrt(num);
}
static double run_impl(const Impl& im, const EpmProblem& p, int iters,
                       std::vector<float>& first, bool& pass) {
  const size_t count = (size_t)p.H * p.W;
  first.assign(count, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> last(count, std::numeric_limits<float>::quiet_NaN());
  im.setup(p);
  im.compute();                    // warm-up; keep this result for verification
  im.fetch(first.data());
  cudaEvent_t e0, e1;
  CHECK(cudaEventCreate(&e0));
  CHECK(cudaEventCreate(&e1));
  CHECK(cudaEventRecord(e0));
  for (int i = 0; i < iters; i++) im.compute();
  CHECK(cudaEventRecord(e1));
  CHECK(cudaEventSynchronize(e1));
  float ms = 0.0f;
  CHECK(cudaEventElapsedTime(&ms, e0, e1));
  CHECK(cudaGetLastError());
  im.fetch(last.data());
  const double repeat = rel_l2(last, first);
  const double per_iter = (double)ms / iters;
  const bool ok = std::isfinite(repeat) && repeat <= 1e-6 &&
                  std::isfinite(per_iter) && per_iter > 0.0;
  pass = pass && ok;
  CHECK(cudaEventDestroy(e0));
  CHECK(cudaEventDestroy(e1));
  im.teardown();
  printf("[%s] iters=%d total_ms=%.3f per_iter_ms=%.4f repeat_rel_l2=%.3e\n",
         im.name, iters, (double)ms, per_iter, repeat);
  printf("VERIFY %s repeat_and_timing %s\n", im.name, ok ? "PASS" : "FAIL");
  return per_iter;
}
static void report(const char* who, const std::vector<float>& got,
                   const std::vector<double>& ref, bool& pass) {
  const EpmVerdict v = epm_verify(got.data(), ref.data(), got.size());
  printf("VERIFY %s_vs_ref rel_l2=%.3e tolerance=%.1e %s\n",
         who, v.rel_l2, EPM_TOL_REL, v.pass ? "PASS" : "FAIL");
  pass = pass && v.pass;
}
int main(int argc, char** argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s H W atoms seed [--iters N] [--skip-ref] [--skip-baseline] [--baseline-ms X]\n", argv[0]);
    return 1;
  }
  int H, W, na, iters = 1;
  unsigned long long seed_value = 0;
  if (!positive_int(argv[1], H) || !positive_int(argv[2], W) ||
      !positive_int(argv[3], na) || !parse_uint(argv[4], UINT_MAX, seed_value)) {
    fprintf(stderr, "need positive integer H, W, atoms, and an unsigned 32-bit seed\n");
    return 1;
  }
  // Avoid overflow and CUDA's grid.y limit with the starter's 16x16 blocks.
  if (H > 65535 * 16 || W > INT_MAX - 15 || (size_t)H * W > INT_MAX) {
    fprintf(stderr, "grid too large for this starter (H <= 1048560, H*W <= INT_MAX)\n");
    return 1;
  }
  const unsigned seed = (unsigned)seed_value;
  bool skip_ref = false, skip_base = false;
  double base_ms = -1.0;
  for (int a = 5; a < argc; a++) {
    if (!strcmp(argv[a], "--iters") && a + 1 < argc) {
      if (!positive_int(argv[++a], iters)) {
        fprintf(stderr, "--iters needs a positive integer\n"); return 1;
      }
    } else if (!strcmp(argv[a], "--skip-ref")) skip_ref = true;
    else if (!strcmp(argv[a], "--skip-baseline")) skip_base = true;
    else if (!strcmp(argv[a], "--baseline-ms") && a + 1 < argc) {
      char* end = nullptr;
      errno = 0;
      base_ms = std::strtod(argv[++a], &end);
      if (errno || end == argv[a] || *end || !std::isfinite(base_ms) || base_ms <= 0.0) {
        fprintf(stderr, "--baseline-ms needs a finite positive time\n"); return 1;
      }
    } else { fprintf(stderr, "unknown or incomplete argument: %s\n", argv[a]); return 1; }
  }
  printf("[epm] H=%d W=%d atoms=%d seed=%u h=%g z0=%g iters=%d\n",
         H, W, na, seed, EPM_GRID_SPACING, EPM_SLICE_Z, iters);
  EpmData data;
  epm_generate(H, W, na, seed, data);
  const EpmProblem p = epm_problem_from(H, W, na, data);
  std::vector<double> ref;
  if (!skip_ref) {
    ref.resize((size_t)H * W);
    epm_reference(p, ref.data());
    printf("[reference] done\n");
  }
  bool pass = true;
  std::vector<float> baseline, student;
  if (!skip_base) {
    const Impl im{"baseline", epm_baseline_setup, epm_baseline_compute,
                  epm_baseline_fetch, epm_baseline_teardown};
    base_ms = run_impl(im, p, iters, baseline, pass);
    if (!skip_ref) report("baseline", baseline, ref, pass);
  }
  const Impl im{"student", epm_student_setup, epm_student_compute,
                epm_student_fetch, epm_student_teardown};
  const double student_ms = run_impl(im, p, iters, student, pass);
  if (!skip_ref) report("student", student, ref, pass);
  else if (!skip_base) {
    const double err = rel_l2(student, baseline);
    const bool ok = std::isfinite(err) && err <= EPM_TOL_REL;
    printf("VERIFY student_vs_baseline rel_l2=%.3e (reference skipped) %s\n",
           err, ok ? "PASS" : "FAIL");
    pass = pass && ok;
  } else printf("VERIFY skipped (no reference and no baseline; timing only)\n");
  if (base_ms > 0.0 && student_ms > 0.0)
    printf("SPEEDUP baseline/student = %.2fx (per-iteration, baseline %.4f ms, student %.4f ms)\n",
           base_ms / student_ms, base_ms, student_ms);
  if (!pass) printf("RESULT FAIL\n");
  else if (skip_ref && skip_base) printf("RESULT TIMING_ONLY\n");
  else printf("RESULT PASS\n");
  return pass ? 0 : 1;
}
