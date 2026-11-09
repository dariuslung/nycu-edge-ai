"""Calibration-free W/A fake quantization of Qwen3-4B (7 projections per block) + WikiText-2 PPL."""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluate import eval_ppl_wikitext2, load_model  # noqa: E402
from mxquant import QLinear, TileQuantizer, TILE  # noqa: E402

PROJ_ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")
PROJ_MLP = ("gate_proj", "up_proj", "down_proj")


@torch.no_grad()
def gaussian_threshold(frac_hi, names=("fp4", "fp6"), n=1 << 21, seed=0, device="cuda"):
    """Data-free threshold: the relative-MXFP4-error quantile such that `frac_hi` of unit-Gaussian
    tiles would be promoted to the higher format."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(n // TILE, TILE, generator=g, device=device)
    q = TileQuantizer([0.0], names, device)
    t = x.reshape(-1, 1, TILE)
    amax = t.abs().amax(-1, keepdim=True)
    _, err = q.fmts[0].quant(t, amax)
    rel = (err / TILE / t.pow(2).mean(-1, keepdim=True)).flatten()
    return torch.quantile(rel, 1 - frac_hi).item()


def quantize_model(model, thr_w, thr_a, names=("fp4", "fp6")):
    dev = next(model.parameters()).device
    wq = TileQuantizer(thr_w, names, dev)
    aq = TileQuantizer(thr_a, names, dev)
    for layer in model.model.layers:
        for p in PROJ_ATTN:
            setattr(layer.self_attn, p, QLinear(getattr(layer.self_attn, p), wq, aq))
        for p in PROJ_MLP:
            setattr(layer.mlp, p, QLinear(getattr(layer.mlp, p), wq, aq))
    return wq, aq


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac_w", type=float, default=0.5, help="target fraction of FP6 tiles for weights (Gaussian-derived threshold)")
    ap.add_argument("--frac_a", type=float, default=0.5)
    ap.add_argument("--thr_w", type=float, default=None)
    ap.add_argument("--thr_a", type=float, default=None)
    ap.add_argument("--thresholds", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "thresholds.json"),
                    help="fixed thresholds produced by derive_thresholds.py (used unless --thr_w/--thr_a/--frac_* given)")
    args = ap.parse_args()
    t0 = time.time()
    explicit = args.thr_w is not None or args.thr_a is not None or "--frac_w" in sys.argv or "--frac_a" in sys.argv
    if not explicit:
        th = json.load(open(args.thresholds))
        args.thr_w, args.thr_a = th["thr_w"], th["thr_a"]
    thr_w = args.thr_w if args.thr_w is not None else gaussian_threshold(args.frac_w)
    thr_a = args.thr_a if args.thr_a is not None else gaussian_threshold(args.frac_a)
    model, tok = load_model()
    wq, aq = quantize_model(model, [thr_w], [thr_a])
    wb = wq.avg_bits()
    ppl = eval_ppl_wikitext2(model, tok)
    ab = aq.avg_bits()
    print(f"Weight bits: {wb:.3f}\nActivation bits: {ab:.3f}\nEffective bit rate (W+A)/2: {(wb+ab)/2:.3f}\nPPL: {ppl:.3f}", file=sys.stderr)
    print(f"thr_w={thr_w:.5f} thr_a={thr_a:.5f} | W bits={wb:.3f} (fp6 {wq.stats[1]/wq.stats[-1]:.1%}) | "
          f"A bits={ab:.3f} (fp6 {aq.stats[1]/aq.stats[-1]:.1%}) | avg={(wb+ab)/2:.3f} | PPL={ppl:.3f} | {time.time()-t0:.0f}s", flush=True)
