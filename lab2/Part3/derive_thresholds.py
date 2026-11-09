"""Derive the two fixed thresholds (weights / activations) WITHOUT any model or data.

Threshold = the relative MXFP4 quantization error (per 32-element tile, best of two shared exponents)
that is exceeded by a fraction `frac` of unit-Gaussian tiles.  Tiles above the threshold are stored in
MXFP6 (E2M3), the rest in MXFP4 (E2M1).  The fractions below were chosen so that the realised effective
bit rate (W+A)/2 on WikiText-2 stays below 5.3 bits (see README / report).
"""
import json
import os

from quantize import gaussian_threshold

FRAC_W, FRAC_A = 0.45, 0.40
out = {"thr_w": gaussian_threshold(FRAC_W), "thr_a": gaussian_threshold(FRAC_A), "frac_w": FRAC_W, "frac_a": FRAC_A,
       "source": "unit-Gaussian tiles, seed 0 (no model, no calibration data)"}
json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds.json"), "w"), indent=2)
print(out)
