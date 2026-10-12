"""Exact Bayes-optimal posterior on a 17x17 grid for the first 300 test households -> data/bayes.npz"""
import time, numpy as np
from sim import *
d = np.load("data/test.npz"); NB = 300
G = np.linspace(-1.2, 1.2, 17); UU, VV = np.meshgrid(G, G, indexing="ij")
logprior = -0.5 * (UU**2 / SIG_U**2 + VV**2 / SIG_V**2)
KS = [4, 8, 16, 24]; out = {k: np.zeros((NB, 5)) for k in KS}   # mu_u, mu_v, sd_u, sd_v, corr
LLall = np.zeros((NB, 17, 17, KMAX), np.float32)            # per-choice log-likelihoods, kept for attribution analysis
t0 = time.time()
for i in range(NB):
    band, age, female = d["meta"][i]; base = base_household(int(age), bool(female), int(band))
    D = d["D"][i].astype(float); y = d["X"][i, :, 5]
    LL = np.zeros((17, 17, KMAX))
    for a in range(17):
        for b in range(17):
            pr = np.clip(probs(shifted(base, G[a], G[b]), D), 1e-9, 1 - 1e-9)
            LL[a, b] = y * np.log(pr) + (1 - y) * np.log(1 - pr)
    LLall[i] = LL
    cum = np.cumsum(LL, axis=2)
    for k in KS:
        lp = logprior + cum[:, :, k - 1]; w = np.exp(lp - lp.max()); w /= w.sum()
        mu = (w * UU).sum(), (w * VV).sum()
        su = np.sqrt((w * (UU - mu[0])**2).sum()); sv = np.sqrt((w * (VV - mu[1])**2).sum())
        cv = (w * (UU - mu[0]) * (VV - mu[1])).sum() / (su * sv)
        out[k][i] = (mu[0], mu[1], su, sv, cv)
    if i % 50 == 0: print(i, f"{time.time()-t0:.0f}s", flush=True)
np.savez("data/bayes.npz", **{f"K{k}": out[k] for k in KS})
np.save("data/bayes_ll.npy", LLall)
