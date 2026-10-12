"""Does attention follow Fisher information? -> results/attention.json"""
import json, numpy as np
from scipy.stats import spearmanr
from sim import *
te = np.load("data/test.npz"); A = np.load("data/attn.npy"); NB = 300; h = 0.05
IU = np.zeros((NB, KMAX)); IV = np.zeros((NB, KMAX))
for i in range(NB):
    b, age, fem = te["meta"][i]; base = base_household(int(age), bool(fem), int(b)); u, v = te["T"][i]
    D = te["D"][i].astype(float)
    p0 = probs(shifted(base, u, v), D)
    du = (probs(shifted(base, u + h, v), D) - probs(shifted(base, u - h, v), D)) / (2 * h)
    dv = (probs(shifted(base, u, v + h), D) - probs(shifted(base, u, v - h), D)) / (2 * h)
    w = np.clip(p0 * (1 - p0), 1e-6, None); IU[i] = du ** 2 / w; IV[i] = dv ** 2 / w
a = A[:NB]; out = {}
def within(x):
    return float(np.nanmean([spearmanr(a[i], x[i]).correlation for i in range(NB) if np.std(x[i]) > 0]))
stake = np.log(te["D"][:NB, :, 0]); social = (te["D"][:NB, :, 3] > 0); choice = te["X"][:NB, :, 5]
out["rho_info_u"] = within(IU); out["rho_info_v"] = within(IV); out["rho_info_total"] = within(IU + IV)
out["rho_stake"] = within(stake); out["rho_choice"] = within(choice)
an = a / a.sum(1, keepdims=True) * KMAX                      # 1.0 = uniform attention
out["attn_social_vs_not"] = [float(an[social].mean()), float(an[~social].mean())]
top = IU + IV >= np.quantile(IU + IV, 0.75, axis=1, keepdims=True)
out["mass_on_top_quartile_info"] = float((a * top).sum(1).mean() / a.sum(1).mean())
band = te["meta"][:NB, 0].astype(int)
out["info_by_band"] = {BANDS[j]: [float(IU[band == j].sum(1).mean()), float(IV[band == j].sum(1).mean())] for j in range(3)}
out["prior_precision_each"] = 1 / SIG_U ** 2
bins = np.quantile(stake, np.linspace(0, 1, 7)); cen, att, inf = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (stake >= lo) & (stake <= hi); cen.append(float(np.exp(stake[m].mean()))); att.append(float(an[m].mean())); inf.append(float((IU + IV)[m].mean()))
out["binned"] = {"stake": cen, "attn_norm": att, "info": inf}
json.dump(out, open("results/attention.json", "w"), indent=1); print(json.dumps(out, indent=1))
