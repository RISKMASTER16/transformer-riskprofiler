"""Train: python train.py <tx|ds> <steps> <seed>  ->  checkpoints/<tx|ds>_<seed>.pkl"""
import os, sys, time, pickle, numpy as np, jax, jax.numpy as jnp, optax
from models import *
which, steps, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
tr, va = np.load("data/train.npz"), np.load("data/val.npz")
mu_x = tr["X"].reshape(-1, 6).mean(0); sd_x = tr["X"].reshape(-1, 6).std(0) + 1e-6
norm = lambda X: ((X - mu_x) / sd_x).astype(np.float32)
Xtr, Ztr, Ttr = norm(tr["X"]), tr["Z"], tr["T"]
Xva, Zva, Tva = norm(va["X"]), va["Z"], va["T"]
init, apply = (init_tx, tx_apply) if which == "tx" else (init_ds, ds_apply)
P = init(jax.random.PRNGKey(seed))
sched = optax.warmup_cosine_decay_schedule(0.0, 1e-3, min(300, steps // 4), steps, 1e-5)
opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=1e-4)); st = opt.init(P)

@jax.jit
def step(P, st, X, Z, T, M):
    l, g = jax.value_and_grad(lambda P: nll(apply(P, X, Z, M), T))(P)
    up, st = opt.update(g, st, P); return optax.apply_updates(P, up), st, l
evalf = jax.jit(lambda P, X, Z, M: apply(P, X, Z, M))

rng = np.random.default_rng(seed); KM = Xtr.shape[1]; B = 256; t0 = time.time(); best = (1e9, None)
for s in range(steps):
    idx = rng.integers(0, len(Xtr), B); k = rng.integers(1, KM + 1, B)   # random 1..24 visible choices
    M = (np.arange(KM)[None] < k[:, None]).astype(np.float32)
    P, st, l = step(P, st, Xtr[idx], Ztr[idx], Ttr[idx], M)
    if s % 1000 == 0 or s == steps - 1:
        Mv = np.ones((len(Xva), KM), np.float32); vl = float(nll(evalf(P, Xva, Zva, Mv), Tva))
        if vl < best[0]: best = (vl, jax.tree_util.tree_map(np.asarray, P))
        print(f"{which} step {s} train {float(l):.4f} val(K=24) {vl:.4f} {time.time()-t0:.0f}s", flush=True)
os.makedirs("checkpoints", exist_ok=True)
pickle.dump({"P": best[1], "mu_x": mu_x, "sd_x": sd_x}, open(f"checkpoints/{which}_{seed}.pkl", "wb"))
