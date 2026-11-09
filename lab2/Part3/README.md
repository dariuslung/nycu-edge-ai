# Part 3: Calibration-free per-tile MXFP4/MXFP6 quantization of Qwen3-4B

Student ID: 314561005

Weights **and** input activations of the 7 projections of every transformer block
(`q/k/v/o/gate/up/down_proj`, 252 Linear layers) are fake-quantized. Each 32-element tile (along the
input-feature dimension) has one shared power-of-two (E8M0) scale and independently picks **MXFP4 (E2M1)**
or **MXFP6 (E2M3)** from a data-free statistic of the tile itself. No calibration data and no
Qwen3-specific calibration is used anywhere.

## 1. Environment

Python 3.11, `torch==2.5.1+cu121` (the grading base environment), one GPU (peak ~12 GB, fits a T4).

```bash
pip install -r requirements.txt     # transformers==4.57.1, huggingface_hub<1.0, datasets<4, accelerate, numpy<2.3
```

`transformers >= 5` refuses to load models on torch 2.5.1, hence the pin. WikiText-2 and
`Qwen/Qwen3-4B` are downloaded from the Hugging Face hub on first use.

## 2. Reproduce the result

```bash
bash run.sh          # = pip install -r requirements.txt; python quantize.py
```

This loads the two fixed thresholds from `thresholds.json`, quantizes Qwen3-4B, evaluates the WikiText-2
test PPL with `evaluate.eval_ppl_wikitext2` (same function as Part 2, unmodified) and prints the realised
bit rates. Expected output:

```
Weight bits: 5.230
Activation bits: 5.342
Effective bit rate (W+A)/2: 5.286
PPL: 14.638
```

(`~4 min` on an RTX 4090.) `run.sh` needs no derivation step: it only *loads* the thresholds.

## 3. Where the thresholds come from (`derive_thresholds.py`)

```bash
python derive_thresholds.py          # rewrites thresholds.json, needs a GPU but no model and no data
```

* statistic per tile: relative error of MXFP4 quantization, `||x - Q4(x)||^2 / ||x||^2` (best of two shared exponents);
* threshold: the quantile of that statistic over unit-Gaussian tiles (seed 0, 2^21 samples) such that a fraction
  `frac_w = 0.45` (weights) / `frac_a = 0.40` (activations) of Gaussian tiles would be promoted to MXFP6;
* tiles of the real model whose relative MXFP4 error exceeds the threshold use MXFP6, all others MXFP4.

The fractions are the only knobs. They are smaller than the realised fractions (47.4 % of weight tiles, 53.1 % of
activation tiles are MXFP6) because real tensors, activations in particular, have heavier tails than a Gaussian.
Nothing is fitted on Qwen3 data or on any other model's data.

## 4. Effective bit rate

Per tile of 32 elements: MXFP4 = 4x32 + 8 (scale) + 1 (format tag) = 137 bits (4.281 b/elem);
MXFP6 = 6x32 + 8 + 1 = 201 bits (6.281 b/elem). AdaptQuant's 5-byte metadata is not counted (as the TA allowed);
our 1-bit format tag is counted anyway.

* weights: counted exactly over all 252 weight matrices;
* activations: counted over every activation tile that actually enters a quantized Linear during the WikiText-2
  test evaluation (realised, not estimated);
* effective bit rate = (weight bits + activation bits) / 2 = (5.230 + 5.342) / 2 = **5.286 <= 5.3**.

## 5. Files

| file | purpose |
|---|---|
| `mxquant.py` | minifloat grids, per-tile format selection, fake-quantized Linear, bit accounting |
| `quantize.py` | quantizes Qwen3-4B and evaluates PPL (the entry point) |
| `derive_thresholds.py` | data-free derivation of `thresholds.json` |
| `thresholds.json` | the fixed thresholds loaded by `quantize.py` |
| `evaluate.py` | provided PPL evaluation (WikiText-2 test, seqlen 2048, fp16) |
| `run.sh`, `requirements.txt` | one-command reproduction |
