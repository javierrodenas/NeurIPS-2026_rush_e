"""Family palette for every figure (final pass, brief 6.0).
Families by color, datasets by MARKER SHAPE only; never reuse a family color
for a different meaning. Null/baseline lines: gray, dashed."""
FAMILY_COLORS = {
    "supervised": "#2A6F97",     # supervised ViT
    "ssl": "#C8553D",            # DINO / DINOv2
    "contrastive": "#6C8B3C",    # CLIP / SigLIP
    "causal_lm": "#7A4E9A",      # GPT-2 / Pythia / OLMo
    "embedder": "#A67C52",       # BGE / GTE / E5
    "null": "#7F7F7F",           # null / baseline (dashed)
}
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
