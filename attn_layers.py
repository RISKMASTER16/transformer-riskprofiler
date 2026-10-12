"""Share of CLS attention on choice tokens vs the demographic token, by layer and band -> results/attn_layers.json"""
import pickle, json, numpy as np, jax, jax.numpy as jnp
from models import *
te = np.load("data/test.npz"); ck = pickle.load(open("checkpoints/tx_0.pkl", "rb")); P = jax.tree_util.tree_map(jnp.asarray, ck["P"])
X = ((te["X"] - ck["mu_x"]) / ck["sd_x"]).astype(np.float32); band = te["meta"][:, 0].astype(int)
out = {}
for k in [4, 24]:
    M = (np.arange(24)[None] < k).astype(np.float32).repeat(len(X), 0)
    _, att = tx_apply(P, X, te["Z"], M, return_attn=True)
    for L, A in enumerate(att):
        A = np.asarray(A).mean(1)
        for j, b in enumerate(["low", "mid", "high"]):
            m = band == j
            out[f"K{k}_L{L+1}_{b}"] = {"cls": float(A[m, 0].mean()), "demo": float(A[m, 1].mean()), "choices": float(A[m, 2:].sum(1).mean())}
json.dump(out, open("results/attn_layers.json", "w"), indent=1)
for key, v in out.items(): print(key, {a: round(b, 3) for a, b in v.items()})
