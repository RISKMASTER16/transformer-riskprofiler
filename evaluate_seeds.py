"""Seed robustness: RMSE, distance to Bayes and coverage for seeds 0-2 -> results/seeds.json"""
import json, pickle, numpy as np, jax, jax.numpy as jnp
from models import *
te = np.load("data/test.npz"); bay = np.load("data/bayes.npz"); NB = 300; KS = [4, 8, 16, 24]; T = te["T"]
res = {}
for name in ["tx", "ds"]:
    f = tx_apply if name == "tx" else ds_apply
    for seed in [0, 1, 2]:
        ck = pickle.load(open(f"checkpoints/{name}_{seed}.pkl", "rb")); P = jax.tree_util.tree_map(jnp.asarray, ck["P"])
        X = ((te["X"] - ck["mu_x"]) / ck["sd_x"]).astype(np.float32); r = {}
        for k in KS:
            M = (np.arange(24)[None] < k).astype(np.float32).repeat(len(X), 0)
            o = np.asarray(f(P, X, te["Z"], M)); mu = o[:, :2]; sd = np.exp(np.clip(o[:, 2:], -4, 2))
            e = mu[:NB] - T[:NB]; d = mu[:NB] - bay[f"K{k}"][:, :2]
            r[k] = {"rmse_u": float(np.sqrt((e[:, 0] ** 2).mean())), "rmse_v": float(np.sqrt((e[:, 1] ** 2).mean())),
                    "dist_u": float(np.sqrt((d[:, 0] ** 2).mean())), "dist_v": float(np.sqrt((d[:, 1] ** 2).mean())),
                    "cover90": float((np.abs(mu - T) < 1.645 * sd).mean())}
        res[f"{name}_{seed}"] = r
summary = {}
for name in ["tx", "ds"]:
    for k in KS:
        for m in ["rmse_u", "rmse_v", "dist_u", "dist_v", "cover90"]:
            v = [res[f"{name}_{s}"][k][m] for s in [0, 1, 2]]
            summary[f"{name}_K{k}_{m}"] = [float(np.mean(v)), float(np.std(v))]
res["summary"] = summary
json.dump(res, open("results/seeds.json", "w"), indent=1)
for key, (m, s) in summary.items(): print(key, f"{m:.3f} ± {s:.3f}")
