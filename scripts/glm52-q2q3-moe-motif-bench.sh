#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/glm52-q2q3-moe-motif-bench.sh [VARIANT...]

Run serial test-backend-ops perf samples for the GLM-5.2 routed Q2/Q3 MoE
decode motif and report medians.

Defaults:
  VARIANT list: top8 active6

Environment:
  LLAMA_DIR   llama.cpp checkout, default .deps/llama.cpp
  BENCH_BIN   test-backend-ops path, default $LLAMA_DIR/build/bin/test-backend-ops
  BACKEND     backend name, default MTL0
  OP          op name, default GLM_MOE_ROUTED_MOTIF_Q2Q3_GLM
  REPEATS     serial samples per variant, default 5
  WARMUPS     unreported warmup samples per variant, default 1
  LOG_DIR     optional directory for raw logs
  EXTRA_ENV   optional comma-separated KEY=VALUE list applied to every run

Variants:
  top8        normal top8 routed MoE motif
  active6     GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=6
  active4     GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=4
  active2     GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=2
  active6-slot1-dual
              active6 plus Q2 gate/up slot1-dual kernel
  active6-slot2-dual
              active6 plus Q2 gate/up slot2-dual kernel
  active6-slot4-dual
              active6 plus Q2 gate/up slot4-dual kernel
  active6-slot8
              active6 plus Q2 gate/up slot8 kernel
  active6-slot8-split
              active6 plus Q2 gate/up slot8-split kernel
  active6-rowtile
              active6 plus Q2 gate/up rowtile kernel
  active6-inblock
              active6 plus Q2 gate/up in-block repack kernel
  active6-dispatch-log
              active6 plus Metal MoE dispatch logging

The first variant is treated as the baseline for delta reporting.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
llama_dir="${LLAMA_DIR:-$repo_root/.deps/llama.cpp}"
bench_bin="${BENCH_BIN:-$llama_dir/build/bin/test-backend-ops}"
backend="${BACKEND:-MTL0}"
op="${OP:-GLM_MOE_ROUTED_MOTIF_Q2Q3_GLM}"
repeats="${REPEATS:-5}"
warmups="${WARMUPS:-1}"
log_dir="${LOG_DIR:-}"

if [[ ! -x "$bench_bin" ]]; then
  echo "missing executable: $bench_bin" >&2
  echo "build it with: cmake --build $llama_dir/build --target test-backend-ops -j 8" >&2
  exit 1
fi

case "$repeats" in
  ''|*[!0-9]*)
    echo "REPEATS must be a positive integer, got: $repeats" >&2
    exit 2
    ;;
esac
case "$warmups" in
  ''|*[!0-9]*)
    echo "WARMUPS must be a non-negative integer, got: $warmups" >&2
    exit 2
    ;;
esac
if (( repeats < 1 )); then
  echo "REPEATS must be >= 1" >&2
  exit 2
fi

if [[ -n "$log_dir" ]]; then
  mkdir -p "$log_dir"
fi

variants=("$@")
if (( ${#variants[@]} == 0 )); then
  variants=(top8 active6)
fi

variant_env() {
  case "$1" in
    top8) ;;
    active6) printf '%s\n' GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=6 ;;
    active4) printf '%s\n' GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=4 ;;
    active2) printf '%s\n' GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=2 ;;
    active6-slot1-dual)
      printf '%s\n' \
        GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=6 \
        GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SLOT1_DUAL=1
      ;;
    active6-slot2-dual)
      printf '%s\n' \
        GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=6 \
        GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SLOT2_DUAL=1
      ;;
    active6-slot4-dual)
      printf '%s\n' \
        GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=6 \
        GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SLOT4_DUAL=1
      ;;
    active6-slot8)
      printf '%s\n' \
        GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=6 \
        GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SLOT8=1
      ;;
    active6-slot8-split)
      printf '%s\n' \
        GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=6 \
        GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_SLOT8_SPLIT=1
      ;;
    active6-rowtile)
      printf '%s\n' \
        GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=6 \
        GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_SWIGLU_PAIR_SG_ROWTILE=1
      ;;
    active6-inblock)
      printf '%s\n' \
        GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=6 \
        GGML_METAL_EXPERIMENTAL_Q2_GATE_UP_INBLOCK_REPACK=1
      ;;
    active6-dispatch-log)
      printf '%s\n' \
        GGML_METAL_EXPERIMENTAL_GLM_MOE_MAX_ACTIVE_EXPERTS=6 \
        GGML_METAL_MOE_DISPATCH_LOG=1
      ;;
    *)
      echo "unknown variant: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
}

extract_us() {
  awk '/runs -/ {
    for (i = 1; i <= NF; ++i) {
      if ($i == "us/run") {
        print $(i - 1)
        exit
      }
    }
  }'
}

tmp_results="$(mktemp "/tmp/glm52-q2q3-motif.XXXXXX")"
trap 'rm -f "$tmp_results"' EXIT

printf 'backend=%s op=%s repeats=%s warmups=%s bench=%s\n' \
  "$backend" "$op" "$repeats" "$warmups" "$bench_bin"

for variant in "${variants[@]}"; do
  mapfile -t env_args < <(variant_env "$variant")
  if [[ -n "${EXTRA_ENV:-}" ]]; then
    IFS=',' read -r -a extra_env_args <<<"$EXTRA_ENV"
    env_args+=("${extra_env_args[@]}")
  fi
  printf '\n== %s ==\n' "$variant"

  total_runs=$((warmups + repeats))
  for ((i = 1; i <= total_runs; ++i)); do
    raw_log=""
    if [[ -n "$log_dir" ]]; then
      raw_log="$log_dir/${variant}-${i}.log"
    fi

    if [[ -n "$raw_log" ]]; then
      env "${env_args[@]}" "$bench_bin" perf -b "$backend" -o "$op" >"$raw_log" 2>&1
      output="$(cat "$raw_log")"
    else
      output="$(env "${env_args[@]}" "$bench_bin" perf -b "$backend" -o "$op" 2>&1)"
    fi

    us="$(printf '%s\n' "$output" | extract_us)"
    if [[ -z "$us" ]]; then
      echo "failed to parse us/run for variant=$variant sample=$i" >&2
      if [[ -n "$raw_log" ]]; then
        echo "raw log: $raw_log" >&2
      else
        printf '%s\n' "$output" >&2
      fi
      exit 1
    fi

    if (( i <= warmups )); then
      printf 'warmup[%d]=%s us\n' "$i" "$us"
    else
      sample=$((i - warmups))
      printf 'sample[%d]=%s us\n' "$sample" "$us"
      printf '%s,%s\n' "$variant" "$us" >>"$tmp_results"
    fi
  done
done

python3 - "$tmp_results" "${variants[@]}" <<'PY'
import csv
import statistics
import sys

path = sys.argv[1]
order = sys.argv[2:]
values = {name: [] for name in order}
with open(path, newline="") as f:
    for variant, us in csv.reader(f):
        values.setdefault(variant, []).append(float(us))

baseline = order[0]
baseline_median = statistics.median(values[baseline])

print("\nsummary")
print("variant,samples,median_us,best_us,worst_us,delta_vs_baseline_pct")
for variant in order:
    samples = values[variant]
    med = statistics.median(samples)
    best = min(samples)
    worst = max(samples)
    delta = ((med - baseline_median) / baseline_median) * 100.0
    print(f"{variant},{len(samples)},{med:.2f},{best:.2f},{worst:.2f},{delta:.2f}")
PY
