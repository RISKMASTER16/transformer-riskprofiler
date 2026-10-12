"""Generate simulated households: python gen_data.py <name> <n> <seed>  ->  data/<name>.npz"""
import os, sys, time, numpy as np
from sim import *

def gen(n, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((n, KMAX, 6), np.float32); Z = np.zeros((n, 5), np.float32)
    T = np.zeros((n, 2), np.float32); P = np.zeros((n, KMAX), np.float32)
    Dall = np.zeros((n, KMAX, 4), np.float32); meta = np.zeros((n, 3), np.float32)
    t0 = time.time()
    for i in range(n):
        band = rng.choice(3, p=BAND_P); age = rng.integers(18, 66); female = rng.random() < 0.5
        u, v = rng.normal(0, SIG_U), rng.normal(0, SIG_V)
        hh = shifted(base_household(age, female, band), u, v)
        D = sample_decisions(rng, KMAX); pr = probs(hh, D)
        y = (rng.random(KMAX) < pr).astype(float)
        X[i] = features(D, y); Z[i] = demo_vec(age, female, band); T[i] = (u, v)
        P[i] = pr; Dall[i] = D; meta[i] = (band, age, female)
        if i % 5000 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)
    return dict(X=X, Z=Z, T=T, P=P, D=Dall, meta=meta)

if __name__ == "__main__":
    name, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    os.makedirs("data", exist_ok=True)
    np.savez_compressed(f"data/{name}.npz", **gen(n, seed))
