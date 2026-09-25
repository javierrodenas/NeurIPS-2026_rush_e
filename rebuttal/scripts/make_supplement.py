"""Build ICLR2027/supplementary_code.zip: the code and the result files behind every number, anonymized.

Contents come from the README mapping (supplementary_code/README.md, itself generated from the built paper):
its result files, its generators, the experiment scripts that write those results, the figure palette and style,
tool/ and the README. Absolute paths are rewritten to data/... in the copies that go into the zip; the repository
is not touched. The script refuses to write a zip that still names the machine or the authors.

    python3 rebuttal/scripts/make_supplement.py
"""
import os, re, sys, glob, shutil, zipfile, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUPP = os.path.join(ROOT, "ICLR2027", "supplementary_code")
README = os.path.join(SUPP, "README.md")
OUT = os.path.join(ROOT, "ICLR2027", "supplementary_code.zip")
EXTERNAL = os.environ.get("PLATONIC_RESULTS_EXTRA", "/media/HDD_4TB_2/javi/Platonic/results")
BANNED = ["rodenas", "radeva", "aguilar", "javi", "ub.edu", "neurips", "/media/"]
SUBS = [("/media/HDD_4TB_2/javi/Platonic", "data/platonic"), ("/media/HDD_4TB_1/javi/ILSVRC2012_img_train", "data/imagenet/train"),
        ("/media/HDD_4TB_2/javi/hf_cache", "data/hf_cache"), ("/media/HDD_4TB_2/javi/torch_cache", "data/torch_cache"),
        ("/media/HDD_4TB_2/javi/mini-imagenet-tools", "data/mini-imagenet"), (ROOT, "."), ("neurips.cc", "openreview.net")]

readme = open(README, encoding="utf-8").read()
res = sorted(set(re.findall(r"`([A-Za-z0-9_./-]+\.(?:csv|json|npz))`", readme)))
gens = sorted(set(re.findall(r"`([A-Za-z0-9_./-]+\.py)`", readme)))
stage = tempfile.mkdtemp(prefix="supp_")
for d in ("results", "scripts", "iclr2027", "figures"): os.makedirs(os.path.join(stage, d))

missing = []
for f in res:
    for src in (os.path.join(ROOT, "rebuttal", "results", f), os.path.join(EXTERNAL, f)):
        if os.path.exists(src): shutil.copy2(src, os.path.join(stage, "results", os.path.basename(f))); break
    else: missing.append(f)
for g in gens:
    base = {"scripts": ("rebuttal", "scripts"), "iclr2027": ("ICLR2027", "iclr2027"), "figures": ("ICLR2027", "figures")}[g.split("/")[0]]
    src = os.path.join(ROOT, *base, os.path.basename(g))
    if os.path.exists(src): shutil.copy2(src, os.path.join(stage, g))
    else: missing.append(g)
stems = {os.path.basename(f).split("_")[0] for f in res}
exp = [p for p in sorted(glob.glob(os.path.join(ROOT, "rebuttal", "scripts", "*.py"))) if os.path.basename(p).split("_")[0] in stems]
for p in exp: shutil.copy2(p, os.path.join(stage, "scripts", os.path.basename(p)))
for f in ("palette.py", "style.mplstyle"): shutil.copy2(os.path.join(ROOT, "ICLR2027", "figures", f), os.path.join(stage, "figures", f))
shutil.copytree(os.path.join(ROOT, "ICLR2027", "tool"), os.path.join(stage, "tool"), ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
shutil.copy2(README, os.path.join(stage, "README.md"))

for root, _, fs in os.walk(stage):   # anonymize the copies, never the repository
    for f in fs:
        p = os.path.join(root, f)
        if f.endswith((".png", ".pdf", ".npy", ".npz", ".zip")): continue
        try: t = open(p, encoding="utf-8").read()
        except Exception: continue
        o = t
        for a, b in SUBS: t = t.replace(a, b)
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

if os.path.exists(OUT): os.remove(OUT)
with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, fs in os.walk(stage):
        for f in sorted(fs): z.write(os.path.join(root, f), os.path.relpath(os.path.join(root, f), stage))
n = sum(len(fs) for _, _, fs in os.walk(stage))
shutil.rmtree(stage, ignore_errors=True)
print(f"supplementary_code.zip: {n} files ({len(res)} results, {len(gens)} generators, {len(exp)} experiment scripts) "
      f"-> {os.path.getsize(OUT)//1024} KB; grep for {', '.join(BANNED)}: 0 hits" + (f"; MISSING {missing}" if missing else ""))
