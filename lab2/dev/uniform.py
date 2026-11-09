import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Part2"))
import torch
from evaluate import load_model, eval_ppl_wikitext2
import compress as C
bits, gs, sym, method = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "sym", sys.argv[4]
model, tok = load_model()
orig = C.original_numel(model)
t = time.time()
calib = C.get_calib(tok)
spec = lambda li, p: dict(bits=bits, gs=gs, sym=sym, method=method)
C.sequential_compress(model, calib, spec, log=lambda s: print(s, file=sys.stderr, flush=True))
r = C.compression_ratio(model, orig)
ppl = eval_ppl_wikitext2(model, tok)
print(f"bits={bits} gs={gs} sym={sym} {method}: ratio={r:.4%} ppl={ppl:.3f} time={time.time()-t:.0f}s peak={torch.cuda.max_memory_allocated()/2**30:.1f}GB", flush=True)
