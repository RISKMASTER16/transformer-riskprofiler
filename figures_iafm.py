"""Figures for the IAFM paper -> figures/fig_rmse.pdf, figures/fig_legibility_faith.pdf"""
import os, json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.linewidth": 0.6, "legend.frameon": False, "mathtext.fontset": "cm"})
os.makedirs("figures", exist_ok=True)
r = json.load(open("results/results_tx.json")); s = json.load(open("results/seeds.json"))["summary"]
fa = json.load(open("results/faithfulness.json")); at = json.load(open("results/attention.json")); KS = [4, 8, 16, 24]

# Figure 1: RMSE vs K, three seeds (mean, min-max) against the exact posterior
seeds = json.load(open("results/seeds.json"))
fig, ax = plt.subplots(1, 2, figsize=(3.4, 1.8), sharey=True)
for j, (key, lab) in enumerate([("u", "caution $u$"), ("v", "constraint $v$")]):
    ax[j].axhline(r["prior_sub"][key], color="0.6", lw=0.8, ls=":", label="prior only")
    for name, mk, col, ls, lab2 in [("ds", "s", "0.45", "--", "DeepSets"), ("tx", "o", "k", "-", "Set transformer")]:
        vals = np.array([[seeds[f"{name}_{sd}"][str(k)][f"rmse_{key}"] for k in KS] for sd in range(3)])
        m = vals.mean(0); lo, hi = m - vals.min(0), vals.max(0) - m
        ax[j].errorbar(KS, m, yerr=[lo, hi], fmt=mk + ls, color=col, ms=3, lw=0.9, capsize=1.5, mfc="white" if name == "ds" else col, label=lab2)
    ax[j].plot(KS, [r["bayes"][str(k)][f"rmse_{key}"] for k in KS], "^-", color="tab:red", ms=3, lw=0.9, label="Bayes-optimal")
    ax[j].set_title(lab, fontsize=8); ax[j].set_xlabel("choices observed $K$"); ax[j].set_xticks(KS)
ax[0].set_ylabel("RMSE"); ax[1].legend(fontsize=5.5, loc="lower left", handlelength=1.8)
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_rmse.pdf")

# Figure 2: (a) legibility by band; (b) faithfulness of attributions (Spearman with exact information)
fig, ax = plt.subplots(1, 2, figsize=(7.0, 1.9), gridspec_kw={"width_ratios": [1, 1.7]})
bands = ["low", "mid", "high"]; ib = at["info_by_band"]; x = np.arange(3)
ax[0].bar(x - 0.18, [ib[b][0] for b in bands], 0.36, color="k", label="caution")
ax[0].bar(x + 0.18, [ib[b][1] for b in bands], 0.36, color="0.65", label="constraint")
ax[0].axhline(at["prior_precision_each"], color="tab:red", lw=0.8, ls="--", label="prior")
ax[0].set_yscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(bands); ax[0].set_ylabel("Fisher information, 24 choices")
ax[0].set_title("(a) legibility", fontsize=8); ax[0].legend(fontsize=6.5, loc="upper right")
methods = [("attention_layer3", "Attention, last layer"), ("attention_rollout", "Rollout"), ("integrated_gradients", "Int. gradients"), ("occlusion", "Occlusion")]
y = np.arange(len(methods))[::-1]
for name, off, col, lab in [("tx", 0.18, "k", "Set transformer"), ("ds", -0.18, "0.65", "DeepSets")]:
    for yi, (m, _) in zip(y, methods):
        v = [fa["models"][f"{name}_{sd}"][m]["spearman"] for sd in range(3) if m in fa["models"].get(f"{name}_{sd}", {})]
        if v: ax[1].barh(yi + off, np.mean(v), 0.34, xerr=np.std(v), color=col, error_kw={"lw": 0.6, "capsize": 1.5}, label=lab if m == "occlusion" else None)
ax[1].set_yticks(y); ax[1].set_yticklabels([l for _, l in methods], fontsize=7.5); ax[1].set_xlim(0, 1)
ax[1].set_xlabel("Spearman with exact information"); ax[1].set_title("(b) explanation faithfulness", fontsize=8)
ax[1].legend(fontsize=6.5, loc="upper right")
fig.tight_layout(pad=0.3); fig.savefig("figures/fig_legibility_faith.pdf"); print("ok")
