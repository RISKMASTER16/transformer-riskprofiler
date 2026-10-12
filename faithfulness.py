"""Explanation faithfulness against an exact Bayesian ground truth -> results/faithfulness.json

Ground truth: the information each choice adds to the exact grid posterior,
  I_j = KL( p(u,v | all 24 choices) || p(u,v | all choices except j) ).
Attributions scored against it: last-layer CLS attention, attention rollout,
model occlusion (KL between the network's posterior with and without choice j),
and integrated gradients on the posterior means.
"""
import json, pickle, numpy as np, jax, jax.numpy as jnp
from scipy.stats import spearmanr
from models import *
from sim import BANDS, SIG_U, SIG_V

NB, K, TOPK = 300, 24, 6
te = np.load("data/test.npz"); LL = np.load("data/bayes_ll.npy").astype(np.float64)
band = te["meta"][:NB, 0].astype(int)
G = np.linspace(-1.2, 1.2, 17); UU, VV = np.meshgrid(G, G, indexing="ij")
logprior = -0.5 * (UU ** 2 / SIG_U ** 2 + VV ** 2 / SIG_V ** 2)

def norm_post(lp):
    lp = lp - lp.max(axis=(-2, -1), keepdims=True); w = np.exp(lp)
    return w / w.sum(axis=(-2, -1), keepdims=True)

full_lp = logprior[None] + LL.sum(-1)                         # (NB,17,17)
p_full = norm_post(full_lp)
def kl_from_full(lp_minus):                                   # KL(p_full || p_minus), batched over NB
    q = norm_post(lp_minus)
    return (p_full * (np.log(p_full + 1e-300) - np.log(q + 1e-300))).sum(axis=(-2, -1))

# ---- ground truth: Bayes leave-one-out information
truth = np.stack([kl_from_full(full_lp - LL[..., j]) for j in range(K)], 1)   # (NB,K)

def score(attr):
    rho = np.array([spearmanr(attr[i], truth[i]).correlation for i in range(NB)])
    top_m = np.argsort(-attr, 1)[:, :TOPK]; top_t = np.argsort(-truth, 1)[:, :TOPK]
    prec = np.mean([len(set(a) & set(b)) / TOPK for a, b in zip(top_m, top_t)])
    return {"spearman": float(np.nanmean(rho)), "spearman_by_band": {BANDS[b]: float(np.nanmean(rho[band == b])) for b in range(3)},
            "top6_precision": float(prec)}

def gauss_kl(mu0, ls0, mu1, ls1):
    s0, s1 = np.exp(ls0), np.exp(ls1)
    return (ls1 - ls0 + (s0 ** 2 + (mu0 - mu1) ** 2) / (2 * s1 ** 2) - 0.5).sum(-1)

def attributions(name, seed):
    ck = pickle.load(open(f"checkpoints/{name}_{seed}.pkl", "rb")); P = jax.tree_util.tree_map(jnp.asarray, ck["P"])
    X = jnp.asarray(((te["X"][:NB] - ck["mu_x"]) / ck["sd_x"]).astype(np.float32)); Z = jnp.asarray(te["Z"][:NB])
    M = jnp.ones((NB, K), jnp.float32); f = tx_apply if name == "tx" else ds_apply
    out0 = np.asarray(f(P, X, Z, M)); mu0, ls0 = out0[:, :2], np.clip(out0[:, 2:], -4, 2)
    A = {}
    occ = np.zeros((NB, K))                                    # model leave-one-out
    for j in range(K):
        o = np.asarray(f(P, X, Z, M.at[:, j].set(0.0))); occ[:, j] = gauss_kl(mu0, ls0, o[:, :2], np.clip(o[:, 2:], -4, 2))
    A["occlusion"] = occ
    ig = np.zeros((NB, K)); steps = 32                          # integrated gradients, baseline = mean token
    for out_idx in (0, 1):
        g = jax.jit(jax.grad(lambda Xa: f(P, Xa, Z, M)[:, out_idx].sum()))
        tot = sum(np.asarray(g(X * ((s + 0.5) / steps))) for s in range(steps)) / steps
        ig += np.abs((np.asarray(X) * tot).sum(-1))
    A["integrated_gradients"] = ig
    if name == "tx":
        _, att = tx_apply(P, X, Z, M, return_attn=True, full_attn=True)
        att = [np.asarray(a).mean(1) for a in att]              # (NB, K+2, K+2), heads averaged
        for L, a in enumerate(att): A[f"attention_layer{L + 1}"] = a[:, 0, 2:]
        R = np.broadcast_to(np.eye(K + 2), att[0].shape).copy()  # attention rollout with residual
        for a in att:
            a = 0.5 * a + 0.5 * np.eye(K + 2); a = a / a.sum(-1, keepdims=True); R = a @ R
        A["attention_rollout"] = R[:, 0, 2:]
    return {m: score(v) for m, v in A.items()}

res = {"truth_total_info_by_band": {BANDS[b]: float(truth[band == b].sum(1).mean()) for b in range(3)},
       "truth_share_top6": float(np.mean(np.sort(truth, 1)[:, -TOPK:].sum(1) / truth.sum(1))),
       "chance_top6_precision": TOPK / K, "models": {}}
for name in ["tx", "ds"]:
    for seed in [0, 1, 2]:
        try: res["models"][f"{name}_{seed}"] = attributions(name, seed); print(name, seed, {m: (round(v['spearman'], 3), round(v['top6_precision'], 3)) for m, v in res["models"][f"{name}_{seed}"].items()}, flush=True)
        except FileNotFoundError: print("missing", name, seed)
json.dump(res, open("results/faithfulness.json", "w"), indent=1)
print("top-6 share of information", round(res["truth_share_top6"], 3))
