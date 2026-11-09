"""Calibration-free per-tile MX format selection (AdaptQuant-style) with fake quantization.

Tile = 32 consecutive elements along the reduction (input-feature) dimension, one shared
power-of-two (E8M0) scale per tile. Each tile independently picks MXFP4 (E2M1) or MXFP6 (E2M3)
from a data-free statistic of the tile itself (relative MXFP4 quantization error) compared to a
fixed threshold.  Nothing is fitted on calibration data.
"""
import math

import torch
import torch.nn as nn

TILE = 32
SCALE_BITS = 8      # E8M0 shared scale
FLAG_BITS = 1       # per-tile format tag (counted, although the TA said metadata is free)


def fp_grid(ebits, mbits):
    """Non-negative values of a sign-less ExMy minifloat (no inf/nan, subnormals, bias=2^(e-1)-1)."""
    bias = 2 ** (ebits - 1) - 1
    vals = {0.0}
    for e in range(2**ebits):
        for m in range(2**mbits):
            if e == 0:
                vals.add(m / 2**mbits * 2 ** (1 - bias))
            else:
                vals.add((1 + m / 2**mbits) * 2 ** (e - bias))
    return torch.tensor(sorted(vals))


FORMATS = {"fp4": (2, 1, 4), "fp6": (2, 3, 6), "fp8": (4, 3, 8)}


class Fmt:
    def __init__(self, name, device):
        e, m, self.bits = FORMATS[name]
        self.grid = fp_grid(e, m).to(device)
        self.mids = ((self.grid[1:] + self.grid[:-1]) / 2).contiguous()
        self.maxv = self.grid[-1].item()
        self.emax = math.floor(math.log2(self.maxv))

    def qdq_unit(self, x):
        """x already divided by the tile scale -> nearest representable value (saturating)."""
        a = x.abs().clamp(max=self.maxv)
        idx = torch.bucketize(a, self.mids)
        return self.grid[idx] * x.sign()

    def quant(self, t, amax):
        """t: [..., T, 32]. Try the MX default shared exponent and the next one, keep the best.
        Returns (dequantized, per-tile squared error)."""
        e0 = torch.floor(torch.log2(amax.clamp(min=1e-30))) - self.emax
        best_q = best_err = None
        for d in (0, 1):
            s = torch.exp2(e0 + d)
            q = self.qdq_unit(t / s) * s
            err = (q - t).pow(2).sum(-1, keepdim=True)
            if best_q is None:
                best_q, best_err = q, err
            else:
                m = err < best_err
                best_q = torch.where(m, q, best_q)
                best_err = torch.where(m, err, best_err)
        return best_q, best_err


class TileQuantizer:
    """Per-tile selection among `names` (ordered low->high precision)."""

    def __init__(self, thresholds, names=("fp4", "fp6"), device="cuda"):
        self.fmts = [Fmt(n, device) for n in names]
        self.thr = thresholds            # relative-error thresholds, len(names)-1
        self.stats = torch.zeros(len(names) + 1, dtype=torch.float64)  # counts per fmt, total tiles

    @torch.no_grad()
    def __call__(self, x, count=False):
        shp = x.shape
        t = x.float().reshape(*shp[:-1], shp[-1] // TILE, TILE)
        amax = t.abs().amax(-1, keepdim=True)
        energy = t.pow(2).mean(-1, keepdim=True).clamp(min=1e-30)
        out, err = self.fmts[0].quant(t, amax)
        rel = err / TILE / energy
        choice = torch.zeros(t.shape[:-1] + (1,), dtype=torch.long, device=x.device)
        for k in range(1, len(self.fmts)):
            use = rel > self.thr[k - 1]
            if use.any():
                q2, err2 = self.fmts[k].quant(t, amax)
                out = torch.where(use, q2, out)
                rel = torch.where(use, err2 / TILE / energy, rel)
                choice = torch.where(use, torch.full_like(choice, k), choice)
        if count:
            for k in range(len(self.fmts)):
                self.stats[k] += (choice == k).sum().item()
            self.stats[-1] += choice.numel()
        return out.reshape(shp).to(x.dtype)

    def avg_bits(self):
        n = self.stats[-1].item()
        if n == 0:
            return float("nan")
        bits = sum(self.stats[k].item() * (self.fmts[k].bits * TILE + SCALE_BITS + FLAG_BITS) for k in range(len(self.fmts)))
        return bits / (n * TILE)


class QLinear(nn.Module):
    """Linear with fake-quantized weights (done once) and dynamically fake-quantized input."""

    def __init__(self, lin, wq, aq):
        super().__init__()
        self.in_features, self.out_features = lin.in_features, lin.out_features
        self.weight = nn.Parameter(wq(lin.weight.data, count=True), requires_grad=False)
        self.bias = lin.bias
        self.aq = aq

    def forward(self, x):
        return nn.functional.linear(self.aq(x, count=True), self.weight, self.bias)
