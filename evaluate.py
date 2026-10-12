"""Head-to-head evaluation against the prior and the Bayes-optimal posterior -> results/results_tx.json"""
import os, pickle, json, numpy as np, jax, jax.numpy as jnp
from models import *
from sim import BANDS
te = np.load("data/test.npz"); bay = np.load("data/bayes.npz"); NB = 300; KS = [4, 8, 16, 24]
T = te["T"]; band = te["meta"][:, 0].astype(int); KM = te["X"].shape[1]
res = {"prior_rmse": np.sqrt((T ** 2).mean(0)).tolist(), "n_test": int(len(T)), "n_bayes": NB}
preds = {}
for name in ["tx", "ds"]:
    ck = pickle.load(open(f"checkpoints/{name}_0.pkl", "rb")); P = jax.tree_util.tree_map(jnp.asarray, ck["P"])
    X = ((te["X"] - ck["mu_x"]) / ck["sd_x"]).astype(np.float32)
    f = tx_apply if name == "tx" else ds_apply
    preds[name] = {}
    for k in KS:
        M = (np.arange(KM)[None] < k).astype(np.float32).repeat(len(X), 0)
        preds[name][k] = np.asarray(f(P, X, te["Z"], M))
    if name == "tx":
        M = np.ones((len(X), KM), np.float32)
        _, att = tx_apply(P, X, te["Z"], M, return_attn=True)
        os.makedirs("results", exist_ok=True)
        np.save("data/attn.npy", np.asarray(att[-1]).mean(1)[:, 2:])   # last layer, CLS -> choice tokens

def rmse(e): return float(np.sqrt((e ** 2).mean()))
for name in ["tx", "ds", "bayes"]:
    res[name] = {}
    for k in KS:
        if name == "bayes": mu = bay[f"K{k}"][:, :2]; sd = bay[f"K{k}"][:, 2:4]; idx = np.arange(NB)
        else: mu = preds[name][k][:, :2]; sd = np.exp(np.clip(preds[name][k][:, 2:], -4, 2)); idx = np.arange(len(T))
        e = mu - T[idx]
        r = {"rmse_u": rmse(e[:, 0]), "rmse_v": rmse(e[:, 1]), "cover90": float((np.abs(e) < 1.645 * sd).mean())}
        for j, bn in enumerate(BANDS):
            m = band[idx] == j; r[f"u_{bn}"] = rmse(e[m, 0]); r[f"v_{bn}"] = rmse(e[m, 1]); r[f"n_{bn}"] = int(m.sum())
        if name != "bayes":
            eb = mu[:NB] - T[:NB]; r["sub_rmse_u"] = rmse(eb[:, 0]); r["sub_rmse_v"] = rmse(eb[:, 1])
            dd = mu[:NB] - bay[f"K{k}"][:, :2]; r["dist_to_bayes_u"] = rmse(dd[:, 0]); r["dist_to_bayes_v"] = rmse(dd[:, 1])
        res[name][k] = r
res["prior_sub"] = {"u": rmse(T[:NB, 0]), "v": rmse(T[:NB, 1])}
res["bayes_mean_post_corr"] = {k: float(bay[f"K{k}"][:, 4].mean()) for k in KS}
os.makedirs("results", exist_ok=True)
json.dump(res, open("results/results_tx.json", "w"), indent=1)
for name in ["tx", "ds", "bayes"]:
    for k in KS: print(name, k, {a: round(b, 3) for a, b in res[name][k].items() if not a.startswith("n_")})
