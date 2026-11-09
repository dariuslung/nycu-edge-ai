import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Part2"))
import torch
from evaluate import load_model, eval_ppl_wikitext2
import compress as C
keep_blocks, attn_bits, mlp_bits, gs = map(int, sys.argv[1:5])
model, tok = load_model()
orig = C.original_numel(model)
t = time.time()
calib = C.get_calib(tok)
def spec(li, p):
    b = attn_bits if p in ("q_proj","k_proj","v_proj","o_proj") else mlp_bits
    return dict(bits=b, gs=gs, sym=True)
C.sequential_compress(model, calib, spec, prune_fn=lambda li: keep_blocks*128, log=lambda s: print(s, file=sys.stderr, flush=True))
removed = 1 - sum(sum(t.numel() for t in m.parameters()) for l in model.model.layers for m in C.block_linears(l).values() if False) if False else None
kept_mlp = model.model.layers[0].mlp.down_proj.in_features
mlp_params = 3*2560*9728
prune_rate = (mlp_params - 3*2560*kept_mlp)/ (orig/36)
r = C.compression_ratio(model, orig)
ppl = eval_ppl_wikitext2(model, tok)
print(f"keep={kept_mlp} prune_rate={prune_rate:.2%} attn{attn_bits}/mlp{mlp_bits} gs{gs}: ratio={r:.4%} ppl={ppl:.3f} time={time.time()-t:.0f}s", flush=True)
