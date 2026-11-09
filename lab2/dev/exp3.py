import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Part2"))
import torch
from evaluate import load_model, eval_ppl_wikitext2
import compress as C
kf = float(sys.argv[1]); bits = None if sys.argv[2] == "fp16" else int(sys.argv[2]); gs = 64
model, tok = load_model()
orig = C.original_numel(model)
t = time.time()
calib = C.get_calib(tok)
svd_num = svd_den = 0
def spec(li, p):
    global svd_num, svd_den
    lin = C.block_linears(model.model.layers[li])[p]
    m, n = lin.weight.shape
    r = int(kf * m * n / (m + n)) // gs * gs
    if li == 0: svd_num += r * (m + n); svd_den += m * n
    return dict(method="svd", rank=r, bits=bits, gs=gs)
C.sequential_compress(model, calib, spec, log=lambda s: print(s, file=sys.stderr, flush=True))
ratio = C.compression_ratio(model, orig)
ppl = eval_ppl_wikitext2(model, tok)
print(f"kf={kf} factor_bits={bits}: svd_ratio(fp16 factors)={1-svd_num/svd_den:.2%} overall={ratio:.4%} ppl={ppl:.3f} time={time.time()-t:.0f}s", flush=True)
