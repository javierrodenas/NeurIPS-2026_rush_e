"""Build ICLR2027/submission/ and submission.zip: the paper with the shape Overleaf wants (author's brief, 2026-09-25).

The main files sit at the top level (.tex, .bib, .bbl, the style files) and the rest lives in two directories,
appendix_tables/ and figures/; the \\input paths of the .tex are rewritten to match. Everything is copied from the
build sources, so the folder is never edited by hand.

    python3 rebuttal/scripts/make_submission.py [path/to/tectonic]
"""
import os, re, sys, shutil, subprocess, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "ICLR2027", "iclr2027") + os.sep
DST = os.path.join(ROOT, "ICLR2027", "submission") + os.sep
MAIN = "main_iclr2027_final.tex"

tex = open(SRC + MAIN, encoding="utf-8").read()
inputs = [m.group(1) for m in re.finditer(r"\\input\{([^}]*)\}", tex)]
figs = [m.group(1) for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)]
figs += [m.group(1) for m in re.finditer(r"\\IfFileExists\{([^}]*)\}", tex)]
bibs = [m.group(1) + ".bib" for m in re.finditer(r"\\bibliography\{([^}]*)\}", tex)]
styles = [m.group(1) + ".bst" for m in re.finditer(r"\\bibliographystyle\{([^}]*)\}", tex)]
for m in re.finditer(r"\\usepackage(?:\[[^\]]*\])?\{([^}]*)\}", tex):
    styles += [p.strip() + ".sty" for p in m.group(1).split(",") if os.path.exists(SRC + p.strip() + ".sty")]
for s in list(styles):   # a style file may require another one that ships here
    if s.endswith(".sty") and os.path.exists(SRC + s):
        for m in re.finditer(r"\\(?:RequirePackage|usepackage)(?:\[[^\]]*\])?\{([^}]*)\}", open(SRC + s, encoding="utf-8").read()):
            styles += [p.strip() + ".sty" for p in m.group(1).split(",") if os.path.exists(SRC + p.strip() + ".sty")]

shutil.rmtree(DST, ignore_errors=True)
os.makedirs(DST + "appendix_tables"); os.makedirs(DST + "figures")
# the appendix tables lose the extra "final/" level, and the .tex follows
for f in inputs:
    f = f if f.endswith(".tex") else f + ".tex"
    flat = "appendix_tables/" + os.path.basename(f) if f.startswith("appendix_tables/") else os.path.basename(f)
    shutil.copy2(SRC + f, DST + flat)
    tex = tex.replace("\\input{" + f[:-4] + "}", "\\input{" + flat[:-4] + "}")
for f in set(figs):
    shutil.copy2(SRC + f, DST + "figures/" + os.path.basename(f))
for f in set(bibs + styles):
    shutil.copy2(SRC + f, DST + os.path.basename(f))
open(DST + MAIN, "w", encoding="utf-8").write(tex)
TECTONIC = sys.argv[1] if len(sys.argv) > 1 else ""
if TECTONIC:   # compile the folder in place: the PDF it ships is the one its own sources produce
    subprocess.run([TECTONIC, "-X", "compile", "--keep-intermediates", MAIN], cwd=DST, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for junk in ("main_iclr2027_final.aux", "main_iclr2027_final.blg", "main_iclr2027_final.out", "main_iclr2027_final.log"):
        if os.path.exists(DST + junk): os.remove(DST + junk)
    pages = subprocess.run(["pdfinfo", DST + "main_iclr2027_final.pdf"], capture_output=True, text=True).stdout
    print("submission compiled in place:", [l for l in pages.split("\n") if l.startswith("Pages")][0])
    root_pdf = os.path.join(ROOT, "ICLR2027", "main_iclr2027_final.pdf")
    if os.path.exists(root_pdf):   # the shipped PDF and the root one must read the same
        a = subprocess.run(["pdftotext", "-layout", DST + "main_iclr2027_final.pdf", "-"], capture_output=True, text=True).stdout
        b = subprocess.run(["pdftotext", "-layout", root_pdf, "-"], capture_output=True, text=True).stdout
        print("submission vs root PDF text:", "identical" if a == b else "DIFFERENT")
        assert a == b, "the folder does not reproduce the shipped PDF"
else:
    shutil.copy2(os.path.join(ROOT, "ICLR2027", "main_iclr2027_final.pdf"), DST + "main_iclr2027_final.pdf")

README = """ICLR 2027 submission - source

Upload this whole directory (or submission.zip) to Overleaf and compile
main_iclr2027_final.tex. Every path in it is relative to this directory:

    main_iclr2027_final.tex   the paper            appendix_tables/   the 12 appendix tables
    references.bib            the bibliography     figures/           the 9 figures
    main_iclr2027_final.bbl   already compiled, so no bibtex pass is needed
    iclr2027_conference.sty/.bst, natbib.sty, fancyhdr.sty, tab_khrulkov_final.tex

    tectonic -X compile main_iclr2027_final.tex      # or pdflatex three times

Expected: 29 pages, main text through page 9, appendix from page 16.
main_iclr2027_final.pdf is the reference build; delete it before uploading if you prefer.
"""
open(DST + "README.txt", "w", encoding="utf-8").write(README)

zpath = os.path.join(ROOT, "ICLR2027", "submission.zip")
if os.path.exists(zpath): os.remove(zpath)
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, fs in os.walk(DST):
        for f in sorted(fs):
            p = os.path.join(root, f)
            z.write(p, os.path.join("submission", os.path.relpath(p, DST)))
n = sum(len(fs) for _, _, fs in os.walk(DST))
print(f"submission/: {n} files ({len(inputs)} inputs, {len(set(figs))} figures) -> submission.zip {os.path.getsize(zpath)//1024} KB")
