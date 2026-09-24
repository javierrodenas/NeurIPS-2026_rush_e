"""Family palette for every figure (final pass, brief 6.0).
Families by color, datasets by MARKER SHAPE only; never reuse a family color
for a different meaning. Null/baseline lines: gray, dashed."""
FAMILY_COLORS = {   # discrete viridis (author's brief, 2026-09-24); the four are distinct in grayscale too
    "supervised": "#3B528B",     # supervised ViT
    "ssl": "#21918C",            # DINO / DINOv2
    "contrastive": "#5EC962",    # CLIP / SigLIP
    "causal_lm": "#440154",      # text models: GPT-2 / Pythia / OLMo
    "embedder": "#440154",       # text models: BGE / GTE / E5
    "null": "#7F7F7F",           # nulls, stars and controls stay gray
}
BAND = "#C9D7EA"                 # bands: solid light blue with a thin darker edge, drawn behind the data (author's brief, 2026-09-24)
BAND_ALPHA = 1.0
BAND_EDGE = "#7D93AE"
BAND_LW = 0.7
DATASET_MARKERS = {"imagenet": "o", "cifar100": "s", "cifar10": "D",
                   "dtd": "^", "fashionmnist": "v", "mnist": "x"}
def family(model):
    m = model.lower()
    if m.startswith("i21k"): return "supervised"
    if m.startswith(("dino",)): return "ssl"
    if m.startswith(("clip", "siglip")): return "contrastive"
    if m.startswith(("gpt", "pythia", "olmo", "qwen")): return "causal_lm"
    return "embedder"
def color(model): return FAMILY_COLORS[family(model)]
