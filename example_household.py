"""Worked example: exact per-choice information vs attributions for one household.
The household is the low-income one with the median occlusion faithfulness (seed 0), not a hand-picked case.
-> figures/fig_example.pdf, results/example.json"""
import json, pickle, numpy as np, jax, jax.numpy as jnp, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from models import *
from sim import SIG_U, SIG_V
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "legend.frameon": False})
NB, K = 300, 24
te = np.load("data/test.npz"); LL = np.load("data/bayes_ll.npy").astype(np.float64); band = te["meta"][:NB, 0].astype(int)
G = np.linspace(-1.2, 1.2, 17); UU, VV = np.meshgrid(G, G, indexing="ij"); lp0 = -0.5 * (UU ** 2 / SIG_U ** 2 + VV ** 2 / SIG_V ** 2)
def post(lp): w = np.exp(lp - lp.max()); return w / w.sum()
ck = pickle.load(open("checkpoints/tx_0.pkl", "rb")); P = jax.tree_util.tree_map(jnp.asarray, ck["P"])
X = jnp.asarray(((te["X"][:NB] - ck["mu_x"]) / ck["sd_x"]).astype(np.float32)); Z = jnp.asarray(te["Z"][:NB]); M = jnp.ones((NB, K), jnp.float32)
o0 = np.asarray(tx_apply(P, X, Z, M)); occ = np.zeros((NB, K))
for j in range(K):
    o = np.asarray(tx_apply(P, X, Z, M.at[:, j].set(0.0)))
    s0, s1 = np.exp(np.clip(o0[:, 2:], -4, 2)), np.exp(np.clip(o[:, 2:], -4, 2))
    occ[:, j] = (np.log(s1 / s0) + (s0 ** 2 + (o0[:, :2] - o[:, :2]) ** 2) / (2 * s1 ** 2) - 0.5).sum(-1)
_, att = tx_apply(P, X, Z, M, return_attn=True); att3 = np.asarray(att[-1]).mean(1)[:, 2:]
truth = np.zeros((NB, K))
for i in range(NB):
    full = lp0 + LL[i].sum(-1); p = post(full)
    for j in range(K): q = post(full - LL[i, :, :, j]); truth[i, j] = (p * (np.log(p + 1e-300) - np.log(q + 1e-300))).sum()
rho = np.array([spearmanr(occ[i], truth[i]).correlation for i in range(NB)])
low = np.where(band == 0)[0]; i = low[np.argsort(rho[low])[len(low) // 2]]
order = np.argsort(-truth[i]); nrm = lambda a: a / a.sum()
D = te["D"][i]; y = te["X"][i, :, 5]
fig, ax = plt.subplots(figsize=(3.4, 1.75)); xs = np.arange(K)
ax.bar(xs, nrm(truth[i])[order], color="0.8", label="exact information $I_j$")
ax.plot(xs, nrm(occ[i])[order], "o", ms=3, color="k", label="occlusion")
ax.plot(xs, nrm(att3[i])[order], "x", ms=3.5, color="tab:red", label="attention (last layer)")
ax.set_xlabel("choices, sorted by exact information"); ax.set_ylabel("share of total"); ax.set_xticks([])
ax.legend(fontsize=6, loc="upper right"); fig.tight_layout(pad=0.3); fig.savefig("figures/fig_example.pdf")
out = {"household": int(i), "age": float(te["meta"][i, 1]), "female": bool(te["meta"][i, 2]),
       "u": float(te["T"][i, 0]), "v": float(te["T"][i, 1]),
       "rho_occlusion": float(rho[i]), "rho_attention3": float(spearmanr(att3[i], truth[i]).correlation),
       "top3": [{"stake": float(D[j, 0]), "multiple": float(D[j, 1]), "p": float(D[j, 2]), "social": float(D[j, 3]), "accepted": int(y[j]),
                 "info_share": float(nrm(truth[i])[j]), "occ_share": float(nrm(occ[i])[j]), "att_share": float(nrm(att3[i])[j])} for j in order[:3]],
       "att_top_choice": {"rank_by_truth": int(np.where(order == np.argmax(att3[i]))[0][0]) + 1, "stake": float(D[np.argmax(att3[i]), 0]),
                          "accepted": int(y[np.argmax(att3[i])]), "info_share": float(nrm(truth[i])[np.argmax(att3[i])])},
       "top6_info_share": float(np.sort(nrm(truth[i]))[-6:].sum())}
json.dump(out, open("results/example.json", "w"), indent=1); print(json.dumps(out, indent=1))
