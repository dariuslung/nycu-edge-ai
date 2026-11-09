#!/usr/bin/env bash
# Lab 2 Part 2 entry point: bash 314561005_run.sh
# Installs the requirements, compresses Qwen3-4B for the 4 cases, evaluates each with run_eval.py
# (WikiText-2 test PPL, unmodified) and prints the compression ratio and PPL of every case.
cd "$(dirname "$0")" || exit 1

pip install -r requirements.txt >/dev/null 2>&1 || true

declare -A RATE=( [1]=0.75 [2]=0.81 [3]=0.50 [4]=0.85 )   # target compression ratios
for c in 1 2 3 4; do
    out=$(python run_eval.py --case "$c" --rate "${RATE[$c]}" 2>/dev/null)
    echo "Case $c"
    echo "Compression Ratio: $(echo "$out" | awk '/^comp_rate:/{print $2}')"
    echo "PPL: $(echo "$out" | awk '/^ppl:/{printf "%.2f", $2}')"
    echo
done
