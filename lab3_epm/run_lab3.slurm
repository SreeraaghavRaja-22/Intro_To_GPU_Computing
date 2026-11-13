#!/bin/bash
# Lab 3 Option A: build, verify, sanitize, and time every version in versions/.
# Submit from inside the epm/ folder:   sbatch run_lab3.slurm
# Run only some versions:               sbatch --export=ALL,VERSIONS="v0_baseline v3_coalesced" run_lab3.slurm
#
# Account/QOS match the course allocation used in Projects/project_1/gpu_alloc.sh.
#SBATCH --job-name=lab3_epm
#SBATCH --output=lab3_epm_%j.log
#SBATCH --account=eel4930
#SBATCH --qos=eel4930
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=02:00:00

set -u
module load cuda gcc
cd "${SLURM_SUBMIT_DIR:-.}"

VERSIONS="${VERSIONS:-v0_baseline v1_const v2_coarsen v3_coalesced v4_shared v5_coarsen2d}"
FINAL="${FINAL:-v3_coalesced}"   # version left in src/student.cu at the end
RUNS=5                           # timing repeats per version (median of these)
ITERS=20                         # compute() calls per timing run
BENCH="2045 2047 4000 7"         # benchmark size from sizes.txt

OUT="results/job_${SLURM_JOB_ID:-local}"
mkdir -p "$OUT"
cp src/student.cu "$OUT/student_before_job.cu"   # safety backup, outside src/

# Environment info for the report
{ echo "== date"; date; echo "== host"; hostname
  echo "== nvidia-smi"; nvidia-smi
  echo "== nvcc"; nvcc --version; } > "$OUT/environment.txt" 2>&1

# Correctness cases from sizes.txt (CPU reference on), including ragged sizes
CASES=(
  "1 1 1 0"
  "100 75 37 1"
  "257 129 1003 2"
  "33 17 4000 3"
  "64 64 64 4"
  "1 65 17 5"
  "65 1 17 6"
  "50 50 4001 8"     # extra: just over one 4000-atom constant-memory chunk
  "37 91 9000 9"     # extra: three chunks / many shared-memory tiles
)

median() { sort -g | awk '{a[NR]=$1} END {if (NR==0) print "NA"; else if (NR%2) print a[(NR+1)/2]; else print (a[NR/2]+a[NR/2+1])/2}'; }

printf "%-16s %-10s %-10s %-12s\n" version correct sanitizer median_ms > "$OUT/summary.txt"
BASE_MS=""

for v in $VERSIONS; do
  echo "=========== $v ==========="
  d="$OUT/$v"; mkdir -p "$d"
  cp "versions/$v.cu" src/student.cu
  make clean >/dev/null
  if ! make > "$d/build.txt" 2>&1; then
    echo "BUILD FAILED for $v (see $d/build.txt)"
    printf "%-16s %-10s %-10s %-12s\n" "$v" BUILD_FAIL - - >> "$OUT/summary.txt"
    continue
  fi

  # 1) correctness against the CPU reference
  pass=0; total=0
  for c in "${CASES[@]}"; do
    total=$((total+1))
    echo "### ./epm $c" >> "$d/correctness.txt"
    ./epm $c >> "$d/correctness.txt" 2>&1
    tail -n 1 "$d/correctness.txt" | grep -q "RESULT PASS" && pass=$((pass+1))
  done

  # 2) compute-sanitizer on a ragged size (two cases: small, and multi-chunk)
  san=CLEAN
  compute-sanitizer --tool memcheck --error-exitcode 1 ./epm 100 75 37 1 > "$d/sanitizer.txt" 2>&1 || san=ERRORS
  compute-sanitizer --tool memcheck --error-exitcode 1 ./epm 37 91 9000 9 >> "$d/sanitizer.txt" 2>&1 || san=ERRORS

  # 3) timing: RUNS repeats at the benchmark size, no CPU reference.
  #    The baseline is skipped to save time; its time is passed in so the
  #    driver still prints SPEEDUP.
  for r in $(seq 1 $RUNS); do
    if [ -n "$BASE_MS" ]; then
      ./epm $BENCH --iters $ITERS --skip-ref --skip-baseline --baseline-ms "$BASE_MS" >> "$d/timing.txt" 2>&1
    else
      ./epm $BENCH --iters $ITERS --skip-ref >> "$d/timing.txt" 2>&1
    fi
  done
  med=$(grep '^\[student\]' "$d/timing.txt" | sed 's/.*per_iter_ms=\([0-9.]*\).*/\1/' | median)
  # v0 is the starter code, so its median is the baseline for later versions
  [ "$v" = "v0_baseline" ] && BASE_MS="$med"

  printf "%-16s %-10s %-10s %-12s\n" "$v" "$pass/$total" "$san" "$med" >> "$OUT/summary.txt"
done

# Full benchmark size checked against the CPU reference, final version only
# (slow: the CPU reference does ~16.7 billion atom-point terms).
cp "versions/$FINAL.cu" src/student.cu
make clean >/dev/null && make >/dev/null 2>&1
./epm $BENCH --iters 1 > "$OUT/final_${FINAL}_bench_with_ref.txt" 2>&1

echo
echo "src/student.cu now holds $FINAL"
cat "$OUT/summary.txt"
