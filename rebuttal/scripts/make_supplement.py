"""Build ICLR2027/supplementary_code.zip: the code that produces the results, nothing that makes the paper.

Contents (author's brief, 2026-09-25):
  scripts/   every experiment script whose output is reported in the paper, plus the helpers they import
  tool/      the instrument as one script, with its demo and its example label files
  README.md  one line per script: what it produces, where that result appears, what it needs and how to run it

Everything that assembles the paper (figures/, the table generators, phaseE_submission.py, sweep_freeze.py,
make_supp_readme.py) and the result CSVs stay out. Absolute paths are rewritten to data/... in the copies that go
into the zip; the repository is not touched, and the script refuses to write a zip that still names the machine or
the authors.

    python3 rebuttal/scripts/make_supplement.py
"""
import os, re, sys, glob, shutil, zipfile, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCRIPTS = os.path.join(ROOT, "rebuttal", "scripts")
OUT = os.path.join(ROOT, "ICLR2027", "supplementary_code.zip")
MAPPING = os.path.join(ROOT, "ICLR2027", "supplementary_code", "README.md")
BANNED = ["rodenas", "radeva", "aguilar", "javi", "ub.edu", "neurips", "/media/"]
LAYOUT = [   # the archive is scripts/ + tool/ at the root, so the paths the repository uses are rewritten
    ('str(HERE.parents[1] / "ICLR2027" / "tool")', 'str(HERE.parent / "tool")'),
    ('str(ROOT/"rebuttal/scripts")', 'str(HERE)'),
    ('str(ROOT / "rebuttal" / "scripts")', 'str(HERE)'),
    # the two diagnostic plots look for the paper's palette and style: in the archive they sit beside the scripts
    ('HERE.parents[1] / "ICLR2027" / "figures"', 'HERE'),
    ('Path(__file__).resolve().parents[2]/"ICLR2027"/"figures"', 'Path(__file__).resolve().parent'),
    ('for o in (FIG, HERE.parents[1] / "ICLR2027" / "iclr2027" / "figures"):', 'for o in (FIG,):'),
]
SUBS = [("/media/HDD_4TB_2/javi/Platonic", "data/platonic"), ("/media/HDD_4TB_1/javi/ILSVRC2012_img_train", "data/imagenet/train"),
        ("/media/HDD_4TB_2/javi/hf_cache", "data/hf_cache"), ("/media/HDD_4TB_2/javi/torch_cache", "data/torch_cache"),
        ("/media/HDD_4TB_2/javi/mini-imagenet-tools", "data/mini-imagenet"), (ROOT, "."), ("neurips.cc", "openreview.net")]
DATASETS = {"imagenet": "ImageNet", "cifar100": "CIFAR-100", "cifar10": "CIFAR-10", "dtd": "DTD", "fashionmnist": "FMNIST",
            "fashion_mnist": "FMNIST", "mnist": "MNIST", "dbpedia": "DBpedia", "hierarcaps": "HierarCaps", "cub": "CUB-200",
            "miniimagenet": "MiniImageNet", "mini_imagenet": "MiniImageNet"}
MODELS = {"i21k": "supervised ViTs", "dinov1": "DINO-B", "dinov2": "DINOv2", "clip": "CLIP", "siglip": "SigLIP", "gpt2": "GPT-2",
          "pythia": "Pythia", "olmo": "OLMo", "bge": "BGE", "gte": "GTE", "e5": "E5", "meru": "MERU", "deit": "DeiT",
          "augreg": "augreg ViT-B", "barlow": "Barlow Twins", "byol": "BYOL", "resnet34": "ResNet-34"}

# ---- what the paper reports, from the mapping the chain keeps
mapping = open(MAPPING, encoding="utf-8").read()
res2paper, fig2res = {}, {}
for line in mapping.split("\n"):
    if not line.startswith("|") or line.count("|") < 4: continue
    cells = [c.strip() for c in line.strip("|").split("|")]
    if len(cells) < 3 or set(cells[0]) <= set("-: "): continue
    where = re.findall(r"`([^`]+)`", cells[0])
    for f in re.findall(r"`([A-Za-z0-9_./-]+\.(?:csv|json|npz))`", cells[-1]):
        res2paper.setdefault(os.path.basename(f), set()).update(where)

# ---- the printed numbers: walk the built paper and number tables and figures as they appear
TEX = os.path.join(ROOT, "ICLR2027", "iclr2027")
paper = open(os.path.join(TEX, "main_iclr2027_final.tex"), encoding="utf-8").read()
def expand(t):
    out = []
    for piece in re.split(r"(\\input\{[^}]*\})", t):
        m = re.fullmatch(r"\\input\{([^}]*)\}", piece)
        if m:
            f = m.group(1); f = f if f.endswith(".tex") else f + ".tex"
            p = os.path.join(TEX, f)
            out.append(open(p, encoding="utf-8").read() if os.path.exists(p) else "")
        else: out.append(piece)
    return "".join(out)
flat = expand(paper)
num, nt, nf = {}, 0, 0
for m in re.finditer(r"\\begin\{(table|figure)\}|\\label\{(tab:[^}]*|fig:[^}]*)\}", flat):
    if m.group(1) == "table": nt += 1
    elif m.group(1) == "figure": nf += 1
    elif m.group(2):
        lab = m.group(2)
        num[lab] = (f"Table {nt}" if lab.startswith("tab:") else f"Figure {nf}")
# ---- the section each result file is cited from, out of the % provenance comments of the body
body = paper[:paper.index("\\appendix")]
sec_of = {}
sec = ""
for line in body.split("\n"):
    m = re.match(r"\\(?:sub)?section\{([^}]*)\}", line)
    if m: sec = m.group(1)
    for f in re.findall(r"%[^%]*?([A-Za-z0-9_./-]+\.(?:csv|json|npz))", line):
        sec_of.setdefault(os.path.basename(f), set()).add(sec)

def label(w):
    if w in num: return num[w]
    if w.startswith("figures/"): return num.get("fig:" + w.split("/")[-1].replace("fig_", "").replace("_final.pdf", ""), "a figure")
    return w.replace("tab:", "Table ").replace("fig:", "Figure ")

# ---- the experiment scripts whose output the paper reports
cands = sorted(glob.glob(os.path.join(SCRIPTS, "exp*.py"))) + sorted(glob.glob(os.path.join(SCRIPTS, "final_wordnet_levels.py")))
keep, produces = [], {}
for p in cands:
    base = os.path.basename(p); stem = base[:-3]
    t = open(p, encoding="utf-8", errors="ignore").read()
    named = {os.path.basename(f) for f in re.findall(r"[\"']([A-Za-z0-9_./-]+\.(?:csv|json|npz))[\"']", t)}
    byname = {f for f in named if f in res2paper}
    bystem = {f for f in res2paper if f.split(".")[0].split("_")[0] == stem.split("_")[0]}
    out = sorted(byname | bystem)
    if out: keep.append(p); produces[base] = out

# ---- the helpers they import (transitively), from this same directory
def local_imports(text):
    names = set()
    for m in re.finditer(r"^(?:from|import)\s+([A-Za-z0-9_, ]+)", text, flags=re.M):
        for n in m.group(1).split(","):
            n = n.strip().split(" as ")[0].strip()
            if os.path.exists(os.path.join(SCRIPTS, n + ".py")): names.add(n + ".py")
    return names
helpers, frontier = set(), [os.path.basename(p) for p in keep]
while frontier:
    b = frontier.pop()
    for h in local_imports(open(os.path.join(SCRIPTS, b), encoding="utf-8", errors="ignore").read()):
        if h not in helpers and h not in [os.path.basename(p) for p in keep]:
            helpers.add(h); frontier.append(h)
        elif h not in [os.path.basename(p) for p in keep] and h not in helpers:
            helpers.add(h)

stage = tempfile.mkdtemp(prefix="supp_")
os.makedirs(os.path.join(stage, "scripts"))
for p in keep: shutil.copy2(p, os.path.join(stage, "scripts", os.path.basename(p)))
for h in sorted(helpers):
    src = os.path.join(SCRIPTS, h)
    if os.path.exists(src): shutil.copy2(src, os.path.join(stage, "scripts", h))
for f in ("palette.py", "style.mplstyle"):   # the colours and the style the two diagnostic plots import
    shutil.copy2(os.path.join(ROOT, "ICLR2027", "figures", f), os.path.join(stage, "scripts", f))
shutil.copytree(os.path.join(ROOT, "ICLR2027", "tool"), os.path.join(stage, "tool"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

# ---- the README: one line per script
def needs(text):
    ds = sorted({v for k, v in DATASETS.items() if re.search(r"[\"'/_]" + k + r"[\"'/_.]", text, flags=re.I)})
    ms = sorted({v for k, v in MODELS.items() if re.search(k, text, flags=re.I)})
    return ", ".join(ms[:6]) or "--", ", ".join(ds[:6]) or "--"
rows = []
for p in keep:
    b = os.path.basename(p); t = open(p, encoding="utf-8", errors="ignore").read()
    where = sorted({label(w) for f in produces[b] for w in res2paper.get(f, ())} - {"a figure"})
    secs = sorted({x for f in produces[b] for x in sec_of.get(f, ())})
    def _k(w):
        m_ = re.match(r"(Table|Figure) (\d+)", w); return (0 if m_ and m_.group(1) == "Figure" else 1, int(m_.group(2)) if m_ else 0)
    where = sorted([w for w in where if w.startswith(("Table", "Figure"))], key=_k) + secs
    m, d = needs(t)
    rows.append((b, ", ".join(where) or "--", m, d, ", ".join(produces[b][:4])))
README = """# Supplementary material: the code behind every result

This archive holds the code that produces the results, not the code that assembles the paper. Each experiment
writes one CSV or JSON per run into `results/` (created on first run); the tables and figures of the paper are
written from those files by the generators, which are not part of this archive.

## How to run

Every script is standalone and is run from this directory:

    PLATONIC_ROOT=<cache root> PLATONIC_RESULTS=results python3 scripts/<script>.py

`PLATONIC_ROOT` points at the directory holding the cached features and centroids (`results/centroids/...`,
`results/text_cache/...`); `PLATONIC_RESULTS` is where the outputs are written. Scripts that extract features
also read the image datasets and download the model weights through `timm` or `transformers`; the rest read only
the caches. Seeds are fixed inside each script.

## The instrument on its own

`tool/` ships the calibrated reading as one script: `calibrated_delta.py` (the estimator, the spectrum-matched
null and the excess), `demo.py` (a worked example on the provided labels), `run_checks.py` (its self-test), and
the example label files for the WordNet-30 and CIFAR-100 groupings. It reproduces any cell of the census from a
centroid matrix, without the rest of the pipeline.

## What each script produces

| script | reported in | models | datasets | writes |
|---|---|---|---|---|
"""
for r in sorted(rows):
    README += f"| `scripts/{r[0]}` | {r[1]} | {r[2]} | {r[3]} | `{r[4]}` |\n"
README += f"\n{len(rows)} experiment scripts, {len(helpers)} shared helpers, plus `tool/`.\n"
README += ("\n`scripts/palette.py` and `scripts/style.mplstyle` are the colours and the matplotlib style of the paper's figures; "
           "`expR64_implanted_depth.py` and `expR64b_implanted_depth_v2.py` import them for the diagnostic plot they draw after writing "
           "their CSV, and `expR55_depth_power.py` and `expR55b_depth_power_leafframe.py` use the style when it is present.\n")
open(os.path.join(stage, "README.md"), "w", encoding="utf-8").write(README)

for root, _, fs in os.walk(stage):   # anonymize the copies, never the repository
    for f in fs:
        p = os.path.join(root, f)
        if f.endswith((".png", ".pdf", ".npy", ".npz", ".zip")): continue
        try: t = open(p, encoding="utf-8").read()
        except Exception: continue
        o = t
        for a, b in LAYOUT + SUBS: t = t.replace(a, b)
        t = re.sub(r"/media/[A-Za-z0-9_./-]*", "data", t)
        t = re.sub(r"[A-Za-z0-9_./-]*javi[A-Za-z0-9_./-]*", "data", t, flags=re.I)
        if t != o: open(p, "w", encoding="utf-8").write(t)

hits = {}
for root, _, fs in os.walk(stage):
    for f in fs:
        p = os.path.join(root, f)
        if f.endswith((".png", ".pdf", ".npy", ".npz", ".zip")): continue
        try: low = open(p, encoding="utf-8", errors="ignore").read().lower()
        except Exception: continue
        for b in BANNED:
            if b in low: hits.setdefault(b, []).append(os.path.relpath(p, stage))
assert not hits, f"identifying strings left: { {k: v[:3] for k, v in hits.items()} }"

# no kept script may import something the archive does not carry
have = {f for f in os.listdir(os.path.join(stage, "scripts"))}
missing = {}
for f in sorted(have):
    for n in local_imports(open(os.path.join(stage, "scripts", f), encoding="utf-8", errors="ignore").read()):
        if n not in have and not os.path.exists(os.path.join(stage, "tool", n)): missing.setdefault(f, []).append(n)
assert not missing, f"scripts importing files the archive does not ship: {missing}"

if os.path.exists(OUT): os.remove(OUT)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, fs in os.walk(stage):
        for f in sorted(fs): z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), stage))
n = sum(len(fs) for _, _, fs in os.walk(stage))
shutil.rmtree(stage, ignore_errors=True)
print(f"supplementary_code.zip: {n} files ({len(keep)} experiment scripts, {len(helpers)} helpers, tool/, README) "
      f"-> {os.path.getsize(OUT)//1024} KB; grep for {', '.join(BANNED)}: 0 hits; no missing import")
