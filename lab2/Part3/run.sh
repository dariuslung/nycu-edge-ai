#!/usr/bin/env bash
# Part 3: calibration-free per-tile MXFP4/MXFP6 W+A fake quantization of Qwen/Qwen3-4B.
cd "$(dirname "$0")" || exit 1
pip install -r requirements.txt >/dev/null 2>&1 || true
python quantize.py
