#!/usr/bin/env python3
"""demo.py -- the instrument on three synthetic clouds (n = 60 points, d = 32, fixed seeds).

  random    : iid Gaussian cloud                                -> expected: not genuine
  star      : 6 Gaussian hubs, 10 points around each            -> expected: genuine, no depth beyond a matched star
  hierarchy : 6 superclusters x 5 subclusters x 2 points        -> expected: genuine, more tree-like than its matched star

Runs calibrated_delta.run with 50 spectrum-null replicates (the paper uses 200) and, for the two
clustered clouds, the matched-star depth test with the 6 (super)cluster labels.

    python demo.py
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import calibrated_delta as cd

N, D, REPS = 60, 32, 50

def random_cloud(seed=0):
    return np.random.RandomState(seed).randn(N, D).astype(np.float32), None

def star(seed=1, K=6):
    rng = np.random.RandomState(seed)
    hubs = rng.randn(K, D) * 4.0
    lab = np.repeat(np.arange(K), N // K)
    return (hubs[lab] + rng.randn(N, D)).astype(np.float32), lab

def hierarchy(seed=2, K=6, S=5):
    rng = np.random.RandomState(seed)
    hubs = rng.randn(K, D) * 4.0                       # level 1: 6 superclusters
    sub = hubs[np.repeat(np.arange(K), S)] + rng.randn(K * S, D) * 1.5   # level 2: 5 subclusters each
    lab_sub = np.repeat(np.arange(K * S), N // (K * S))
    lab_sup = lab_sub // S
    return (sub[lab_sub] + rng.randn(N, D) * 0.3).astype(np.float32), lab_sup

def report(name, C, lab):
    t0 = time.time(); out = cd.run(C, lab, n_rep=REPS); c = out["census"]
    verdict = "GENUINE" if c["genuine"] else "not genuine"
    line = (f"{name:10s} raw delta_norm {c['delta']:.3f} | excess {c['excess']:+.3f} "
            f"r={c['r_above']}/{REPS} p={c['p_left']:.3f} -> {verdict}")
    if "depth" in out:
        d = out["depth"]
        sign = "more tree-like than its matched star" if d["depth_excess"] < 0 else "no depth beyond a matched star"
        line += f" | depth vs matched star {d['depth_excess']:+.3f} (z {d['z_depth']:+.1f}) -> {sign}"
    print(line + f"  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    print(f"n={N}, d={D}, {REPS} spectrum-null replicates; genuine := p <= 0.05\n")
    report("random", *random_cloud())
    report("star", *star())
    report("hierarchy", *hierarchy())
    print("\nReading: a low raw delta is shared by all three; the excess separates the random cloud from the "
          "clustered ones; only the matched-star test separates the star (depth > 0) from the hierarchy (depth < 0).\n"
          "With n = 60 points and K = 6 hubs the depth test is underpowered (the hub null has few degrees of freedom), "
          "so read the sign of the depth here, not its z; the paper's calibration (Tables B25/B29) uses n = 100-1000.")
