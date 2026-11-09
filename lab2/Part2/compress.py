"""Compression toolkit for Qwen3-4B (Lab 2, Part 2).

Target layers: the 7 Linear projections of every transformer block
(q/k/v/o/gate/up/down). Original size = 16 bits per weight (fp16).

Everything is "fake" in the sense that the model keeps fp16 tensors holding the
dequantized values, but the *compressed size* is accounted from the real storage
format: integer codes + fp16 scale per group (+ integer zero point per group for
asymmetric quantization) + whatever structure SVD / pruning removes.
"""
import gc
import random

import torch
import torch.nn as nn

PROJS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GROUPS = (("q_proj", "k_proj", "v_proj"), ("o_proj",), ("gate_proj", "up_proj"), ("down_proj",))
FP_BITS = 16


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def block_linears(layer):
    d = {}
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        d[p] = getattr(layer.self_attn, p)
    for p in ("gate_proj", "up_proj", "down_proj"):
        d[p] = getattr(layer.mlp, p)
    return d


def set_block_linear(layer, proj, module):
    parent = layer.self_attn if proj in ("q_proj", "k_proj", "v_proj", "o_proj") else layer.mlp
    setattr(parent, proj, module)


def original_numel(model):
    return sum(m.weight.numel() for l in model.model.layers for m in block_linears(l).values())


def get_calib(tok, nsamples=128, seqlen=2048, seed=0):
    """Random windows of the WikiText-2 *train* split (never the test split)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    ids = tok("\n\n".join(ds["text"]), return_tensors="pt").input_ids
    rng = random.Random(seed)
    out = []
    for _ in range(nsamples):
        i = rng.randint(0, ids.shape[1] - seqlen - 1)
        out.append(ids[:, i : i + seqlen])
    return torch.cat(out, 0)


# ----------------------------------------------------------------------------
# quantization primitives
# ----------------------------------------------------------------------------
def _qdq(x, scale, zero, maxq):
    q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
    return scale * (q - zero)


def find_params(w, bits, sym, mse=True, grid=24, lo=0.55):
    """Per-row scale / zero for w of shape [rows, g]."""
    maxq = 2**bits - 1
    xmin = torch.minimum(w.min(1, keepdim=True)[0], torch.zeros(1, device=w.device))
    xmax = torch.maximum(w.max(1, keepdim=True)[0], torch.zeros(1, device=w.device))
    if sym:
        xmax = torch.maximum(xmin.abs(), xmax)
        xmin = -xmax
    tmp = (xmin == 0) & (xmax == 0)
    xmin = torch.where(tmp, -torch.ones_like(xmin), xmin)
    xmax = torch.where(tmp, torch.ones_like(xmax), xmax)

    def mk(a, b):
        scale = (b - a) / maxq
        zero = torch.full_like(scale, (maxq + 1) / 2) if sym else torch.round(-a / scale)
        return scale, zero

    scale, zero = mk(xmin, xmax)
    if mse:
        best = (_qdq(w, scale, zero, maxq) - w).abs().pow(2.4).sum(1, keepdim=True)
        for i in range(1, grid):
            p = 1 - i * (1 - lo) / grid
            s1, z1 = mk(p * xmin, p * xmax)
            err = (_qdq(w, s1, z1, maxq) - w).abs().pow(2.4).sum(1, keepdim=True)
            m = err < best
            best = torch.where(m, err, best)
            scale = torch.where(m, s1, scale)
            zero = torch.where(m, z1, zero)
    return scale, zero


def storage_bits(rows, cols, bits, gs, sym):
    ngroups = rows * (cols // gs)
    return rows * cols * bits + ngroups * (FP_BITS + (0 if sym else bits))


@torch.no_grad()
def gptq(W, H, bits, gs, sym=False, mse=True, blocksize=128, damp=0.01):
    """GPTQ with static groups. Returns the dequantized weight (float32)."""
    W = W.float().clone()
    rows, cols = W.shape
    maxq = 2**bits - 1
    H = H.clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=W.device)
    H = torch.linalg.cholesky(H)
    H = torch.cholesky_inverse(H)
    Hinv = torch.linalg.cholesky(H, upper=True)
    Q = torch.zeros_like(W)
    scale = zero = None
    for i1 in range(0, cols, blocksize):
        i2 = min(i1 + blocksize, cols)
        W1 = W[:, i1:i2].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            w = W1[:, i]
            d = Hinv1[i, i]
            if (i1 + i) % gs == 0:
                src = W1[:, i : i + gs] if gs <= blocksize else W[:, i1 + i : i1 + i + gs]
                scale, zero = find_params(src, bits, sym, mse)
            q = _qdq(w.unsqueeze(1), scale, zero, maxq).flatten()
            Q1[:, i] = q
            err = (w - q) / d
            W1[:, i:] -= err.unsqueeze(1) @ Hinv1[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q


@torch.no_grad()
def rtn(W, bits, gs, sym=False, mse=True):
    """Round-to-nearest group-wise quantization (no calibration)."""
    rows, cols = W.shape
    g = W.float().reshape(rows * (cols // gs), gs)
    scale, zero = find_params(g, bits, sym, mse)
    return _qdq(g, scale, zero, 2**bits - 1).reshape(rows, cols)



@torch.no_grad()
def svd_factors(W, H, rank, damp=0.01):
    """Activation-aware truncated SVD (SVD-LLM style whitening with the input Gram matrix H).

    Returns A [rank, in], B [out, rank] with W ~= B @ A."""
    W = W.float()
    cols = W.shape[1]
    H = H.clone()
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, device=W.device)
    L = torch.linalg.cholesky(H)
    U, S, Vh = torch.linalg.svd(W @ L, full_matrices=False)
    U, S, Vh = U[:, :rank], S[:rank], Vh[:rank]
    sq = S.sqrt()
    A = torch.linalg.solve_triangular(L, Vh, upper=False, left=False) * sq[:, None]
    B = U * sq[None, :]
    return A, B


def lowrank_storage_bits(out_f, in_f, rank, bits, gs):
    if bits is None:
        return rank * (out_f + in_f) * FP_BITS
    return storage_bits(rank, in_f, bits, gs, True) + storage_bits(out_f, rank, bits, gs, True)


class LowRankLinear(nn.Module):
    def __init__(self, A, B, dtype):
        super().__init__()
        self.in_features, self.out_features = A.shape[1], B.shape[0]
        self.a = nn.Linear(A.shape[1], A.shape[0], bias=False)
        self.b = nn.Linear(B.shape[1], B.shape[0], bias=False)
        self.a.weight.data = A.to(dtype)
        self.b.weight.data = B.to(dtype)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.b(self.a(x))


def _new_linear(w):
    m = nn.Linear(w.shape[1], w.shape[0], bias=False)
    m.weight.data = w.contiguous()
    m.weight.requires_grad_(False)
    return m


@torch.no_grad()
def prune_mlp(layer, keep, act_rms):
    """Structured pruning: keep the `keep` most important intermediate neurons.

    importance_j = rms(activation_j) * ||down_proj[:, j]||; gate/up rows and down columns of
    the dropped neurons are physically removed, so no mask / index is needed."""
    mlp = layer.mlp
    imp = act_rms * mlp.down_proj.weight.float().norm(dim=0)
    idx = torch.topk(imp, keep).indices.sort().values
    g, u, d = mlp.gate_proj.weight.data, mlp.up_proj.weight.data, mlp.down_proj.weight.data
    mlp.gate_proj = _new_linear(g[idx])
    mlp.up_proj = _new_linear(u[idx])
    mlp.down_proj = _new_linear(d[:, idx])


# ----------------------------------------------------------------------------
# sequential driver
# ----------------------------------------------------------------------------
class _Catcher(nn.Module):
    def __init__(self, module, store):
        super().__init__()
        self.module = module
        self.store = store
        self.attention_type = getattr(module, "attention_type", "full_attention")

    def forward(self, x, **kw):
        self.store["inps"][self.store["i"]] = x[0]
        self.store["i"] += 1
        self.store["kw"] = kw
        raise ValueError


def _layer_out(layer, x, kw):
    out = layer(x, **kw)
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def sequential_compress(model, calib_ids, spec_fn, prune_fn=None, log=print):
    """Quantize the model block by block.

    spec_fn(layer_idx, proj) -> None (leave fp16) or dict(bits, gs, sym, method)
    method: 'gptq' (default), 'rtn' or 'svd' (rank, optional bits/gs for the factors).
    prune_fn(layer_idx) -> number of MLP neurons to keep (None = keep all).
    Records storage cost in model._comp_bits (dict name -> bits).
    """
    dev = next(model.parameters()).device
    layers = model.model.layers
    ns, seqlen = calib_ids.shape
    hidden = model.config.hidden_size
    store = {"inps": torch.zeros(ns, seqlen, hidden, dtype=torch.float16, device=dev), "i": 0, "kw": None}
    layers[0] = _Catcher(layers[0], store)
    for j in range(ns):
        try:
            model(calib_ids[j : j + 1].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    kw = store["kw"]
    inps = store["inps"]
    outs = torch.zeros_like(inps)
    if not hasattr(model, "_comp_bits"):
        model._comp_bits = {}

    for li, layer in enumerate(layers):
        keep = prune_fn(li) if prune_fn else None
        if keep is not None and keep < layer.mlp.down_proj.in_features:
            acc = torch.zeros(layer.mlp.down_proj.in_features, device=dev)

            def prune_hook(mod, inp, out):
                acc.add_(inp[0].reshape(-1, inp[0].shape[-1]).float().pow(2).sum(0))

            h = layer.mlp.down_proj.register_forward_hook(prune_hook)
            for j in range(ns):
                _layer_out(layer, inps[j : j + 1], kw)
            h.remove()
            prune_mlp(layer, keep, (acc / (ns * seqlen)).sqrt())
        for group in GROUPS:
            lins = block_linears(layer)
            specs = {p: spec_fn(li, p) for p in group}
            todo = [p for p in group if specs[p] is not None and specs[p].get("method", "gptq") in ("gptq", "svd")]
            Hs = {}
            hooks = []
            if todo:
                for p in todo:
                    Hs[p] = torch.zeros(lins[p].in_features, lins[p].in_features, device=dev)

                def mkhook(p):
                    cnt = [0]

                    def hook(mod, inp, out):
                        x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                        n = x.shape[0]
                        Hs[p] *= cnt[0] / (cnt[0] + n)
                        cnt[0] += n
                        x = x * (2 / cnt[0]) ** 0.5
                        Hs[p] += x.t() @ x

                    return hook

                for p in todo:
                    hooks.append(lins[p].register_forward_hook(mkhook(p)))
                for j in range(ns):
                    _layer_out(layer, inps[j : j + 1], kw)
                for h in hooks:
                    h.remove()
            for p in group:
                s = specs[p]
                if s is None:
                    continue
                W = lins[p].weight.data
                if s.get("method") == "svd":
                    A, B = svd_factors(W, Hs[p], s["rank"])
                    if s.get("bits") is not None:
                        A = rtn(A, s["bits"], s["gs"], True, True)
                        B = rtn(B, s["bits"], s["gs"], True, True)
                    set_block_linear(layer, p, LowRankLinear(A, B, W.dtype))
                    model._comp_bits[(li, p)] = lowrank_storage_bits(W.shape[0], W.shape[1], s["rank"], s.get("bits"), s.get("gs"))
                    continue
                if s.get("method", "gptq") == "gptq":
                    Q = gptq(W, Hs[p], s["bits"], s["gs"], s.get("sym", False), s.get("mse", True))
                else:
                    Q = rtn(W, s["bits"], s["gs"], s.get("sym", False), s.get("mse", True))
                lins[p].weight.data = Q.to(W.dtype)
                model._comp_bits[(li, p)] = storage_bits(W.shape[0], W.shape[1], s["bits"], s["gs"], s.get("sym", False))
            del Hs
        for j in range(ns):
            outs[j] = _layer_out(layer, inps[j : j + 1], kw)[0]
        inps, outs = outs, inps
        if li % 6 == 0:
            log(f"  block {li}/{len(layers)} done")
        torch.cuda.empty_cache()
    del inps, outs, store
    gc.collect()
    torch.cuda.empty_cache()


# ----------------------------------------------------------------------------
# accounting
# ----------------------------------------------------------------------------
def compression_ratio(model, orig_numel):
    """1 - compressed/original over the target Linear layers.

    Layers that were not touched cost 16 bits/weight. Pruned/SVD layers are
    registered in model._comp_bits too (see their helpers)."""
    comp = 0
    for li, layer in enumerate(model.model.layers):
        for p, m in block_linears(layer).items():
            key = (li, p)
            if key in model._comp_bits:
                comp += model._comp_bits[key]
            else:
                comp += sum(t.numel() for t in m.parameters()) * FP_BITS
    return 1 - comp / (orig_numel * FP_BITS)


# ----------------------------------------------------------------------------
# the four cases
# ----------------------------------------------------------------------------
ATTN = ("q_proj", "k_proj", "v_proj", "o_proj")

CASES = {
    # Case 1: GPTQ 3-bit symmetric, one fp16 scale per 128 weights  -> 3.125 bit/weight
    1: dict(desc="GPTQ 3-bit, group 128, symmetric (all 7 projections)"),
    # Case 2: same, group 512 -> 3.031 bit/weight
    2: dict(desc="GPTQ 3-bit, group 512, symmetric (all 7 projections)"),
    # Case 3: activation-aware SVD (rank ~78% of the break-even budget) + 10-bit factors
    3: dict(desc="whitened SVD on all 7 projections, factors RTN 10-bit group 64", kf=0.78, fbits=10, fgs=64),
    # Case 4: structured MLP pruning (keep 6784/9728 neurons) + GPTQ (attention 3-bit, MLP 2-bit)
    4: dict(desc="MLP neuron pruning (keep 6784/9728) + GPTQ attn 3-bit / MLP 2-bit, group 128", keep=6784),
}


def apply_case(model, tok, case, wbits=None, log=print):
    """Compress `model` in place for the given case. Sets model._base_linear_numel."""
    torch.manual_seed(0)
    cfg = CASES[case]
    model._base_linear_numel = original_numel(model)
    log(f"[compress] case {case}: {cfg['desc']}")
    calib = get_calib(tok, 128, 2048, seed=0)
    prune_fn = None
    if case in (1, 2):
        gs = 128 if case == 1 else 512
        bits = wbits or 3
        spec = lambda li, p: dict(bits=bits, gs=gs, sym=True)
    elif case == 3:

        def spec(li, p):
            lin = block_linears(model.model.layers[li])[p]
            m, n = lin.weight.shape
            r = int(cfg["kf"] * m * n / (m + n)) // cfg["fgs"] * cfg["fgs"]
            return dict(method="svd", rank=r, bits=cfg["fbits"], gs=cfg["fgs"])

    else:
        prune_fn = lambda li: cfg["keep"]
        spec = lambda li, p: dict(bits=3 if p in ATTN else 2, gs=128, sym=True)
    sequential_compress(model, calib, spec, prune_fn=prune_fn, log=log)
    log(f"[compress] done, compression ratio = {compression_ratio(model, model._base_linear_numel):.4%}")
