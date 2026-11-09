import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Part2"))
import torch
from evaluate import load_model, eval_ppl_wikitext2
import compress as C
target = sys.argv[1]
model, tok = load_model()
calib = C.get_calib(tok)
C.sequential_compress(model, calib, lambda li, p: dict(bits=2, gs=128, sym=True) if p == target else None, log=lambda s: None)
print(f"only {target} at 2-bit g128: ppl={eval_ppl_wikitext2(model, tok):.3f}", flush=True)
