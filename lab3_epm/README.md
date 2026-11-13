# Lab 03, Option A: Electrostatic Potential Map

Make a correct GPU program faster without changing its answer. This package
implements Option A from the Lab 3 potential-map and MLP slides. Read the
[top-level README](../README.md) first for the shared rules, timing, analysis,
and submission requirements.

## The problem

Each atom has a position `(x, y, z)` and a charge `q`. At every point of a 2D
grid slice, sum the contribution from every atom:

```
x = h * i; y = h * j
potential[j*W + i] = sum_a q[a] / distance((x, y, z0), atom[a])
```

The grid has `H` rows and `W` columns, with row-major output. Atom data is
stored as `x, y, z, q, x, y, z, q, ...` (four floats per atom). The provided
baseline assigns one thread to each grid point and reads atoms from global
memory. Positions and signed charges come from a seed; no dataset is needed.

`include/params.h` fixes spacing `h = 0.1f` and slice height `z0 = 0.0f`.
Generated atoms are at least `0.25` units away from the slice, so the distance
cannot be zero. There is no softening term, cutoff, or periodic boundary.
Grid dimensions and atom count are command-line inputs, including ragged sizes.

Use the provided `coulomb_term(q, d2)` from **`include/coulomb.h`** for each
atom contribution. Treat it as a black box: the scientific arithmetic is fixed.
All GPU storage and arithmetic must remain float32. The CPU reference computes
the same sum in double precision using `q / sqrt(d2)`.

## What you edit

Only **`src/student.cu`**, plus `.cu`/`.cuh` files you add in `src/`. It starts
as a copy of the baseline. The Makefile builds every `src/*.cu`, so keep backups
outside `src/`. Everything else is read-only and restored before grading.

| File | Role |
|------|------|
| `include/params.h` | fixed geometry and relative L2 tolerance |
| `include/coulomb.h` | fixed device arithmetic helper |
| `include/problem.h` | data layout and required functions |
| `include/cuda_check.h` | CUDA error checking |
| `src/gen.cpp` | seeded atom generator |
| `src/ref_cpu.cpp` | double-precision CPU reference |
| `src/verify.cpp` | numerical verification |
| `src/baseline.cu` | naive GPU baseline |
| `src/main.cu` | input generation, verification, and timing |
| **`src/student.cu`** | **your implementation** |

## Expectations

Implement `epm_student_setup`, `epm_student_compute`, `epm_student_fetch`, and
`epm_student_teardown` (see `include/problem.h`).

- `setup`: allocate device memory and upload inputs; may change the input layout.
- `compute`: compute the **entire map on every call**, using device work only.
- `fetch`: copy the `H * W` potential values back to the host.
- `teardown`: free your allocations.

Only `compute` is timed. Do not precompute contributions in `setup`, skip atoms,
or replace the provided arithmetic helper.

## Build and run

On a GPU compute node, inside an allocation:

```bash
module load cuda gcc
make
./epm 100 75 37 1           # H W atoms seed
./epm 257 129 1003 2 --iters 3
compute-sanitizer --tool memcheck --error-exitcode 1 ./epm 100 75 37 1
```

See `sizes.txt` for correctness and benchmark commands. Write your own Slurm
script and timing log.

| Option | Meaning |
|--------|---------|
| `--iters N` | number of timed complete map evaluations (default 1) |
| `--skip-ref` | skip the CPU reference for timing development |
| `--skip-baseline` | skip the baseline |
| `--baseline-ms X` | reuse a baseline per-iteration time measured with the same inputs and GPU |

Look for `VERIFY student_vs_ref ... PASS`, `repeat_rel_l2`, per-iteration
times, `SPEEDUP baseline/student`, and `RESULT PASS`. Without the CPU reference,
the driver compares the student output with the baseline when available. If
both comparisons are skipped, it prints `RESULT TIMING_ONLY`; this is not a
correctness check. The driver also checks that the final timed output agrees
with the warm-up output, catching implementations that accumulate across calls.

## Verification

The relative L2 error `||GPU - reference||_2 / ||reference||_2` must not exceed
`EPM_TOL_REL` in `include/params.h`; non-finite outputs fail. If the reference
norm is zero, the verifier uses absolute L2 error. Recheck against the CPU
reference after each change, test sizes that are not multiples of your block or
coarsening factor, and require zero Compute Sanitizer errors.

The slides suggest considering atom reuse, memory types, coalescing, and work
per thread when choosing what to try.
