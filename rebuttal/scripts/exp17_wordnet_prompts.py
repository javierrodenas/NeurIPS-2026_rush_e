#!/usr/bin/env python3
"""
ICLR EXP 17 — WordNet-definition and non-visual prompts (committed to RJje A6:
"add WordNet-definition and non-visual prompts in the revision").

Same protocol, estimator and models as EXP 4 (GPT-2 S/M causal last-token;
BGE/E5 embedders), over the 1000 ImageNet classes, with five templates:

  photo        "a photo of a {name}"                  (paper baseline, anchors to exp4)
  def_pair     "{name}: {gloss}"                      (dictionary-style WordNet definition)
  gloss_only   "{gloss}"                              (NO class name at all: the strongest
                                                       anti-lexical control)
  concept      "the concept of {name}"                (non-visual framing)
  discussion   "a discussion about {name}"            (non-visual framing)

Glosses: WordNet definition of each class synset (wnids from timm, class order).
Output: rebuttal/results/exp17_wordnet_prompts.csv + exp17.log
"""
import os, sys, time
os.environ.setdefault("HF_HOME", "/home/javi/Platonic/.hf_cache")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
sys.stdout.reconfigure(line_buffering=True)
import numpy as np, pandas as pd, torch
from pathlib import Path
from scipy.spatial.distance import pdist, squareform

ROOT = Path("/home/javi/Platonic"); OUT = ROOT/"rebuttal/results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

sys.path.insert(0, str(ROOT/"rebuttal/scripts"))
from exp4_prompt_variation import (imagenet_class_names, delta_max_from_D,
                                   encode_causal, encode_embedder)

MODELS = [
    ("gpt2",     "causal", "gpt2"),
    ("gpt2_m",   "causal", "gpt2-medium"),
    ("bge_base", "embed",  "BAAI/bge-base-en-v1.5"),
    ("e5_base",  "embed",  "intfloat/e5-base-v2"),
]

def glosses():
    from timm.data import ImageNetInfo
    from nltk.corpus import wordnet as wn
    wnids = ImageNetInfo().label_names()
    return [wn.synset_from_pos_and_offset('n', int(w[1:])).definition() for w in wnids]

def main():
    names = imagenet_class_names()
    gl = glosses()
    assert len(names) == len(gl) == 1000
    TEMPLATES = {
        "photo":      [f"a photo of a {n}" for n in names],
        "def_pair":   [f"{n}: {g}" for n, g in zip(names, gl)],
        "gloss_only": gl,
        "concept":    [f"the concept of {n}" for n in names],
        "discussion": [f"a discussion about {n}" for n in names],
    }
    csv_path = OUT/"exp17_wordnet_prompts.csv"
    rows = pd.read_csv(csv_path).to_dict("records") if csv_path.exists() else []
    done = {(r["model"], r["template"]) for r in rows}
    for short, kind, hf in MODELS:
        for tkey, prompts in TEMPLATES.items():
            if (short, tkey) in done:
                print(f"{short}/{tkey}: skip"); continue
            t0 = time.time()
            if kind == "causal":
                X = encode_causal(hf, prompts)
            else:
                pool = "cls" if short == "bge_base" else "mean"
                pfx = "query: " if short == "e5_base" else ""
                X = encode_embedder(hf, [pfx+p for p in prompts], pool)
            D = squareform(pdist(X, 'euclidean'))
            dm, ds = delta_max_from_D(D)
            rows.append(dict(model=short, template=tkey, delta=dm, delta_sd=ds,
                             time_s=time.time()-t0))
            print(f"{short:9s} {tkey:11s} delta {dm:.4f}±{ds:.4f}  ({rows[-1]['time_s']:.0f}s)")
            pd.DataFrame(rows).to_csv(csv_path, index=False)
    print("Done")

if __name__ == "__main__":
    main()
