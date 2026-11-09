from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

ss = getSampleStyleSheet()
B = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9.5, leading=13)
H1 = ss["Heading1"]; H2 = ss["Heading2"]

def tbl(rows, widths=None):
    t = Table(rows, colWidths=widths, hAlign="LEFT")
    t.setStyle(TableStyle([("FONTSIZE", (0,0), (-1,-1), 8.5), ("GRID", (0,0), (-1,-1), 0.4, colors.grey),
                           ("BACKGROUND", (0,0), (-1,0), colors.lightgrey), ("VALIGN", (0,0), (-1,-1), "TOP")]))
    return t

def to_md(path, title, items):
    import re
    clean = lambda t: re.sub(r"</?b>", "**", t).replace("&lt;", "<")
    out = [f"# {title}", "", "Student ID: 314561005", ""]
    for kind, v in items:
        if kind == "h":
            out += [f"## {v}", ""]
        elif kind == "p":
            out += [clean(v), ""]
        else:
            rows = v[0] if isinstance(v, tuple) else v
            out.append("| " + " | ".join(rows[0]) + " |")
            out.append("|" + "---|" * len(rows[0]))
            out += ["| " + " | ".join(r) + " |" for r in rows[1:]]
            out.append("")
    open(path, "w").write("\n".join(out))


def build(path, title, items):
    to_md(path.replace(".pdf", ".md"), title, items)
    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=1.8*cm, bottomMargin=1.8*cm,
                            title=title, author="314561005")
    out = [Paragraph(title, H1), Paragraph("Student ID: 314561005", B), Spacer(1, 6)]
    for kind, v in items:
        if kind == "h": out.append(Paragraph(v, H2))
        elif kind == "p": out.append(Paragraph(v, B)); out.append(Spacer(1, 4))
        elif kind == "t": out.append(tbl(*v) if isinstance(v, tuple) else tbl(v)); out.append(Spacer(1, 6))
    doc.build(out)

P = lambda s: ("p", s)
# ------------------------------------------------------------------ Part 2
p2 = [
 ("h", "Setup and size accounting"),
 P("Model: Qwen/Qwen3-4B, fp16, evaluated only through the provided run_eval.py (WikiText-2 test, seqlen 2048); the PPL code is unchanged. "
   "run_eval.py was extended only to call compress.apply_case() before the evaluation, and compression_rate() in evaluate.py (a stub in the handout) "
   "now returns the ratio computed by compress.py. Base model: PPL 13.64."),
 P("Target layers: the 7 Linear projections (q, k, v, o, gate, up, down) of all 36 blocks, 3.64 B weights, 16 bits each originally. "
   "Compression Ratio = 1 - compressed bits / original bits, where the compressed size is counted from the real storage format: "
   "integer codes + one fp16 scale per group (+ a zero point per group for asymmetric quantization) for quantized layers; "
   "r(m+n) weights (or their quantized size with scales) for SVD layers; removed rows/columns are simply absent for pruned layers. "
   "Layers that are not touched cost 16 bits per weight. The model itself keeps fp16 tensors holding the dequantized values (fake quantization), "
   "so the numerics are exactly those of the stored format."),
 P("Common tool: GPTQ (Hessian-based error compensation) with static groups, symmetric scales, a per-group MSE search over clipping ratios (24 candidates from 1.0 to 0.55), "
   "dampening 0.01, block size 128, sequential block-by-block quantization (q/k/v, then o, then gate/up, then down, so each group sees the already-quantized earlier layers). "
   "Calibration data: 128 random 2048-token windows of the WikiText-2 <b>train</b> split (seed 0); the test split is never touched. "
   "The TA confirmed that calibration data is allowed and is not fine-tuning. No weights are trained."),
 ("h", "Results"),
 ("t", ([["Case", "Method", "Compression Ratio", "PPL", "Requirement"],
         ["1", "GPTQ 3-bit, group 128", "80.47%", "14.55", ">75%, PPL<15"],
         ["2", "GPTQ 3-bit, group 512", "81.05%", "15.79", ">=81%, PPL<20"],
         ["3", "whitened SVD + 10-bit factors", "51.45% (SVD ratio 24.2%)", "23.11", "SVD>=20%, overall 50%, PPL<50"],
         ["4", "MLP neuron pruning + GPTQ 3/2-bit", "88.07% (pruning 22.4%)", "25.23", "prune>=20%, overall>=85%, PPL<50"]],
        [1.2*cm, 5.2*cm, 4.2*cm, 1.5*cm, 5*cm])),
 ("h", "Case 1: GPTQ 3-bit, group 128 (all 7 projections)"),
 P("<b>Strategy.</b> Every target layer is quantized to 3 bits with a symmetric fp16 scale per 128 weights: 3 + 16/128 = 3.125 bits per weight, i.e. 80.47% compression. "
   "Asymmetric 4-bit with groups of 128 costs 4.16 bits and gives only 74.02% (PPL 14.02), which misses the 75% requirement, so 3-bit was the natural choice."),
 P("<b>Result.</b> Compression 80.47%, PPL 14.55 (base 13.64). "
   "<b>Analysis.</b> GPTQ's error compensation plus the clipping search keeps 3-bit nearly lossless (+0.9 PPL). Group size 128 keeps the scale overhead at only 0.125 bit while still isolating outliers."),
 ("h", "Case 2: GPTQ 3-bit, group 512"),
 P("<b>Strategy.</b> The same method with one scale per 512 weights: 3 + 16/512 = 3.031 bits, 81.05% compression (just above the 81% requirement; plain group 128 gives 80.47%). "
   "<b>Result.</b> 81.05%, PPL 15.79. <b>Analysis.</b> Coarser groups lose resolution for outlier-heavy rows (+1.2 PPL versus group 128), but the budget is far from the limit of 20. "
   "A 2-bit probe on every layer (group 128) reaches 86.7% but PPL 25.2, so 3-bit with larger groups is the better trade-off at 81%."),
 ("h", "Case 3: activation-aware SVD + quantized factors"),
 P("<b>Strategy.</b> All 252 target layers are replaced by two thin Linear layers W ~ B A. The decomposition is SVD-LLM style: with the input Gram matrix H = X^T X = L L^T collected on the "
   "same 128 calibration windows, the SVD is taken on W L, truncated, and mapped back with L^-1, so the truncation minimizes the output error ||(W - W') X||, not the weight error. "
   "Rank per layer r = floor(0.78 m n/(m+n)) rounded down to a multiple of 64, which is a 24.2% reduction of the parameter count relative to the original W matrix (the TA's definition of the SVD compression ratio, fp16 factors). "
   "The two factors are then quantized to 10 bits (round-to-nearest, symmetric, fp16 scale per 64 values, 10.25 bits per weight), which brings the overall ratio to 51.45%."),
 P("<b>Result.</b> SVD ratio 24.2%, overall compression 51.45%, PPL 23.11. <b>Analysis.</b> The truncation is the dominant source of error: with fp16 factors and a 21.4% SVD ratio the PPL is already 20.44, "
   "while the 10-bit quantization of the factors adds almost nothing. Plain weight-SVD is not usable here; the whitening is what keeps the PPL below 50. The 10-bit width was picked so that the overall ratio just clears 50%."),
 ("h", "Case 4: structured MLP pruning + mixed 3/2-bit GPTQ"),
 P("<b>Strategy.</b> In each block the 9728 MLP intermediate neurons are ranked by rms(activation) x ||down_proj column|| (activations from the calibration set, measured with the already-compressed previous blocks); "
   "the 6784 best are kept and the other rows of gate/up and columns of down are physically removed (30.3% of the MLP neurons, 22.4% of all target weights). Structured pruning needs no mask or index, "
   "whereas an unstructured mask would cost about 1 bit per weight and defeat the purpose. The remaining weights use GPTQ: attention (q, k, v, o) at 3-bit and MLP at 2-bit, symmetric, group 128."),
 P("<b>Result.</b> Pruning rate 22.4%, compression 88.07%, PPL 25.23. A milder prune (7040 neurons, 20.45%) gave 87.81% and 25.74. "
   "<b>Analysis.</b> The attention projections are cheap to compress (see the sensitivity table) so they get the extra bit; pruning the least important neurons costs little because those neurons are removed together with their 2-bit quantization error. "
   "For reference, 2-bit on every layer without pruning gives 25.16 at 86.7%; the pruned model reaches a similar PPL at a higher ratio, although the two differ in several ways (attention is 3-bit here), so I did not isolate the effect of the pruning alone."),
 ("h", "Sensitivity of the projection types"),
 P("Only one projection type quantized to 2-bit (GPTQ, group 128) in all 36 blocks, everything else fp16 (base PPL 13.64):"),
 ("t", ([["Projection", "q", "k", "v", "o", "gate", "up", "down"],
         ["PPL", "13.71", "13.53", "14.31", "14.04", "14.82", "15.34", "15.18"]], [2.4*cm] + [1.9*cm]*7)),
 ("h", "Overall conclusion"),
 P("GPTQ-style error-compensated quantization with calibration data is by far the most efficient tool: 3-bit costs less than 1 PPL at 80% compression, and 2-bit gives 87% at PPL 25. "
   "SVD is much weaker per bit: even the activation-aware version needs a 24% rank reduction just to reach a PPL of 23 at about 50% overall. Structured pruning is a mild, cheap add-on when the metadata is free. "
   "The MLP projections (up, down, gate) are the most sensitive per weight, k/q the least, so a mixed allocation (more bits for the MLP) is preferable. "
   "Limitations: compression is simulated (the model keeps fp16 dequantized tensors and the size is accounted from the storage format; no packed kernels); the calibration set is small and from WikiText-2, "
   "so results on the WikiText-2 test split may be optimistic relative to other domains; the per-case settings were selected by running a handful of configurations and checking the WikiText-2 PPL, "
   "not by an exhaustive search. Peak GPU memory is about 11.6 GB (fp16, fits a 15 GB T4)."),
 ("h", "Other configurations tried (all with GPTQ, WikiText-2 PPL)"),
 ("t", ([["Config", "Compression", "PPL"],
         ["4-bit asym, g128", "74.02%", "14.02"], ["3-bit sym, g128", "80.47%", "14.55"], ["3-bit sym, g512", "81.05%", "15.79"],
         ["2-bit sym, g128", "86.72%", "25.16"], ["SVD 20%, fp16 factors", "21.43%", "20.44"], ["SVD + 10-bit factors", "51.45%", "23.11"],
         ["prune 20.45% + 3/2-bit", "87.81%", "25.74"], ["prune 22.4% + 3/2-bit", "88.07%", "25.23"]], [6*cm, 3.5*cm, 2.5*cm])),
]
build("Part2/314561005_report.pdf", "Lab 2 Part 2 report: compressing Qwen3-4B", p2)

# ------------------------------------------------------------------ Part 3
p3 = [
 ("h", "1. Method"),
 P("I implemented my own calibration-free PTQ in the spirit of AdaptQuant (per-tile format selection among MX formats), not a reproduction of the paper. "
   "Scope: weights and input activations of the 7 projections of every block (252 Linear layers), fake quantization in fp16 on the GPU. "
   "A tile is 32 consecutive elements along the reduction dimension with one shared power-of-two (E8M0) scale. Each tile chooses between MXFP4 (E2M1) and MXFP6 (E2M3). "
   "Per tile and per format two shared exponents (the standard one and one above it) are tried and the lower-MSE one is kept; this costs no extra bits. "
   "The format is chosen from the tile itself: if the relative MXFP4 error ||x - Q4(x)||^2/||x||^2 exceeds a threshold, the tile is stored in MXFP6. "
   "Weights are decided once; activations are decided on the fly for every token."),
 P("<b>Why it is calibration-free.</b> No dataset is used to fit anything. The two thresholds (weights, activations) are quantiles of the same statistic over unit-Gaussian tiles "
   "(seed 0, no model, no data; derive_thresholds.py), stored in thresholds.json; the official run only loads them. "
   "<b>Honest note:</b> the two fractions (45% for weights, 40% for activations) that define the quantiles were chosen by running a few settings and looking at the realised bit rate and PPL on WikiText-2, "
   "so they are hyper-parameters picked with feedback from the evaluation set (three settings tried, listed below), although they are not derived from any model's calibration data."),
 P("<b>Differences from AdaptQuant.</b> AdaptQuant derives its thresholds from a calibration pass on Llama-3.1-8B and also selects a per-tile offset; my thresholds come from a synthetic Gaussian, there is no offset, and only two formats are used."),
 ("h", "2. Result"),
 ("t", ([["Quantity", "Value"],
         ["Final PPL (WikiText-2 test)", "14.64  (base fp16: 13.64; AdaptQuant: 13.98)"],
         ["Weight bits (exact)", "5.230  (47.4% of tiles MXFP6)"],
         ["Activation bits (realised on the evaluation)", "5.342  (53.1% of tiles MXFP6)"],
         ["Effective bit rate (W+A)/2", "5.286 <= 5.3"]], [7*cm, 9*cm])),
 P("<b>Bit-rate calculation.</b> A tile has 32 elements. MXFP4 = 4 x 32 + 8 (E8M0 scale) + 1 (format tag) = 137 bits = 4.281 bit/element; MXFP6 = 6 x 32 + 8 + 1 = 201 bits = 6.281 bit/element. "
   "Weight bits = 4.281 + 2 x 0.474 = 5.230. Activation bits are counted over every activation tile that enters a quantized Linear during the WikiText-2 test evaluation: 4.281 + 2 x 0.531 = 5.342. "
   "(5.230 + 5.342)/2 = 5.286. AdaptQuant's 5-byte metadata is not counted (as allowed by the TA); my own 1-bit tag is counted."),
 ("h", "3. Observations and discussion"),
 ("t", ([["Setting", "W bits", "A bits", "Avg", "PPL"],
         ["all MXFP4", "4.281", "4.281", "4.281", "17.24"],
         ["all MXFP6", "6.281", "6.281", "6.281", "13.90"],
         ["Gaussian fractions w 0.50 / a 0.50", "5.329", "5.507", "5.418 (over budget)", "14.57"],
         ["w 0.45 / a 0.40 (final)", "5.230", "5.342", "5.286", "14.64"],
         ["w 0.35 / a 0.45", "5.024", "5.425", "5.225", "14.92"]], [6.2*cm, 2*cm, 2*cm, 3.5*cm, 2*cm])),
 P("<b>Selection works.</b> Uniform MXFP4 gives PPL 17.24 and uniform MXFP6 13.90 (about the base model). Interpolating linearly in bits between them would give about 15.6 at 5.29 bits; the per-tile choice gives 14.64, "
   "so spending the MXFP6 budget on the tiles that MXFP4 represents worst is clearly better than a uniform split. "
   "<b>Activations are heavier-tailed than Gaussian.</b> With the same Gaussian quantile for both tensors, 61% of activation tiles (versus 52% of weight tiles) were promoted, which pushed the bit rate to 5.42; "
   "that is why the activation fraction had to be lowered. Realised activation bits depend on the input data, so the bit rate is a measurement on the evaluation text, not a guarantee. "
   "<b>Difficulties.</b> Getting the budget right without data: the threshold cannot be set from the real activation distribution (that would be calibration), so I set it conservatively from a synthetic one. "
   "<b>Possible reasons for the remaining degradation (not isolated by experiments).</b> MXFP4 tiles containing outliers, where a power-of-two scale wastes range, and the 48-53% of tiles that stay in MXFP4. Embeddings, lm_head and the attention score computation stay in fp16 and are not quantized. "
   "<b>Limitations and improvements.</b> No real low-precision kernel (fake quantization only); a third level (MXFP8) for the worst tiles, a smarter data-free statistic (e.g. crest factor), or non-power-of-two scales would likely lower PPL further at the same budget."),
]
build("Part3/314561005_report.pdf", "Lab 2 Part 3 report: calibration-free MX per-tile quantization of Qwen3-4B", p3)
