#!/usr/bin/env python3
"""Numeric check for 1Eqj: four-point delta of H^2 (K=-1) and spheres.
H^2: sample points in a Poincare disk region of increasing geodesic radius R;
report UNNORMALIZED max four-point defect and NORMALIZED delta/diam.
Sphere S^{d-1}: normalized delta/diam (geodesic) for d=3, 100.
Tree: balanced binary tree graph metric."""
import numpy as np
from scipy.spatial.distance import pdist, squareform

def delta_from_D(D, n_quads=500_000, seeds=5):
    diam = D.max(); n = D.shape[0]; out = []
    for s in range(seeds):
        rng = np.random.RandomState(s)
        i,j,k,l = (rng.randint(0,n,n_quads) for _ in range(4))
        ok = (i!=j)&(i!=k)&(i!=l)&(j!=k)&(j!=l)&(k!=l)
        i,j,k,l = i[ok],j[ok],k[ok],l[ok]
        S = np.sort(np.stack([D[i,j]+D[k,l], D[i,k]+D[j,l], D[i,l]+D[j,k]],1),1)
        out.append(((S[:,2]-S[:,1])/2).max())
    return np.mean(out), diam

# H^2 via Poincare disk, uniform in hyperbolic area up to geodesic radius R
def h2_D(R, n=1000, seed=0):
    rng = np.random.RandomState(seed)
    # geodesic radius CDF: A(r) ~ cosh(r)-1 -> invert
    u = rng.rand(n); r = np.arccosh(1 + u*(np.cosh(R)-1))
    th = rng.rand(n)*2*np.pi
    er = np.tanh(r/2); pts = np.stack([er*np.cos(th), er*np.sin(th)],1)
    sq = (pts**2).sum(1)
    d2 = np.maximum(sq[:,None]+sq[None,:]-2*pts@pts.T, 0)
    den = np.maximum((1-sq[:,None])*(1-sq[None,:]), 1e-15)
    return np.arccosh(np.maximum(1+2*d2/den, 1.0))

for R in [2, 4, 8, 16]:
    d, diam = delta_from_D(h2_D(R))
    print(f"H2 region R={R:2d}: delta={d:.3f} (abs)  diam={diam:.1f}  delta_norm={d/diam:.4f}")

for dim in [3, 100]:
    rng = np.random.RandomState(0)
    X = rng.randn(1000, dim); X /= np.linalg.norm(X,axis=1,keepdims=True)
    G = np.clip(X@X.T, -1, 1); D = np.arccos(G)
    d, diam = delta_from_D(D)
    print(f"S^{dim-1} geodesic: delta={d:.3f}  diam={diam:.2f}  delta_norm={d/diam:.4f}")
    De = squareform(pdist(X)); d, diam = delta_from_D(De)
    print(f"S^{dim-1} chord   : delta={d:.3f}  diam={diam:.2f}  delta_norm={d/diam:.4f}")

# balanced binary tree, unit edges, depth 10 (1023 nodes)
import collections
depth=10; n=2**depth-1
# distance via LCA
def tdist(a,b):
    da=int(np.log2(a+1)); db=int(np.log2(b+1)); d=0; 
    while da>db: a=(a-1)//2; da-=1; d+=1
    while db>da: b=(b-1)//2; db-=1; d+=1
    while a!=b: a=(a-1)//2; b=(b-1)//2; d+=2
    return d
idx = np.arange(n); D = np.zeros((n,n))
for i in range(n):
    for j in range(i+1,n): D[i,j]=D[j,i]=tdist(i,j)
d, diam = delta_from_D(D)
print(f"binary tree depth {depth}: delta={d:.3f}  diam={diam:.0f}  delta_norm={d/diam:.4f}")
