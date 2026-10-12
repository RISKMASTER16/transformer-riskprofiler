"""Set transformer (no positional encoding) and DeepSets baseline, in plain JAX."""
import jax, jax.numpy as jnp, numpy as np
D_MODEL, HEADS, LAYERS = 64, 4, 3

def dense(key, n_in, n_out):
    w = jax.random.normal(key, (n_in, n_out)) * np.sqrt(2.0 / n_in)
    return {"w": w, "b": jnp.zeros(n_out)}
def lin(p, x): return x @ p["w"] + p["b"]
def ln_init(d): return {"g": jnp.ones(d), "b": jnp.zeros(d)}
def ln(p, x):
    m = x.mean(-1, keepdims=True); v = ((x - m) ** 2).mean(-1, keepdims=True)
    return p["g"] * (x - m) / jnp.sqrt(v + 1e-5) + p["b"]

# ---- set transformer: [CLS, demographic token, choice tokens...], no positions
def init_tx(key, d=D_MODEL, L=LAYERS):
    ks = iter(jax.random.split(key, 64))
    P = {"tok1": dense(next(ks), 6, d), "tok2": dense(next(ks), d, d),
         "demo": dense(next(ks), 5, d), "cls": jax.random.normal(next(ks), (d,)) * 0.02,
         "layers": [], "lnf": ln_init(d), "out": dense(next(ks), d, 4)}
    for _ in range(L):
        P["layers"].append({"ln1": ln_init(d), "q": dense(next(ks), d, d), "k": dense(next(ks), d, d),
                            "v": dense(next(ks), d, d), "o": dense(next(ks), d, d), "ln2": ln_init(d),
                            "m1": dense(next(ks), d, 4 * d), "m2": dense(next(ks), 4 * d, d)})
    return P

def tx_apply(P, X, Z, mask, return_attn=False, full_attn=False):
    B, K, _ = X.shape; d = P["cls"].shape[0]; hd = d // HEADS
    tok = lin(P["tok2"], jax.nn.gelu(lin(P["tok1"], X)))
    h = jnp.concatenate([jnp.broadcast_to(P["cls"], (B, 1, d)), lin(P["demo"], Z)[:, None], tok], 1)
    m = jnp.concatenate([jnp.ones((B, 2)), mask], 1)
    bias = (1.0 - m)[:, None, None, :] * -1e9
    attns = []
    for lp in P["layers"]:
        x = ln(lp["ln1"], h)
        q = lin(lp["q"], x).reshape(B, -1, HEADS, hd).transpose(0, 2, 1, 3)
        k = lin(lp["k"], x).reshape(B, -1, HEADS, hd).transpose(0, 2, 1, 3)
        v = lin(lp["v"], x).reshape(B, -1, HEADS, hd).transpose(0, 2, 1, 3)
        a = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(hd) + bias, -1)
        attns.append(a if full_attn else a[:, :, 0, :])  # full matrices, or the CLS row of each head
        h = h + lin(lp["o"], (a @ v).transpose(0, 2, 1, 3).reshape(B, -1, d))
        h = h + lin(lp["m2"], jax.nn.gelu(lin(lp["m1"], ln(lp["ln2"], h))))
    out = lin(P["out"], ln(P["lnf"], h[:, 0]))       # mu_u, mu_v, log sd_u, log sd_v
    return (out, attns) if return_attn else out

# ---- DeepSets baseline: mean pooling, no attention
def init_ds(key, d=D_MODEL):
    ks = iter(jax.random.split(key, 16))
    return {"p1": dense(next(ks), 11, 2 * d), "p2": dense(next(ks), 2 * d, 2 * d),
            "r1": dense(next(ks), 2 * d + 5, 2 * d), "r2": dense(next(ks), 2 * d, d), "out": dense(next(ks), d, 4)}

def ds_apply(P, X, Z, mask):
    B, K, _ = X.shape
    xz = jnp.concatenate([X, jnp.broadcast_to(Z[:, None], (B, K, 5))], -1)
    phi = jax.nn.gelu(lin(P["p2"], jax.nn.gelu(lin(P["p1"], xz))))
    pooled = (phi * mask[..., None]).sum(1) / jnp.maximum(mask.sum(1, keepdims=True), 1)
    r = jax.nn.gelu(lin(P["r2"], jax.nn.gelu(lin(P["r1"], jnp.concatenate([pooled, Z], -1)))))
    return lin(P["out"], r)

def nll(out, T):
    """Gaussian negative log-likelihood of the true (u, v)."""
    mu, logsd = out[:, :2], jnp.clip(out[:, 2:], -4, 2)
    return (logsd + 0.5 * ((T - mu) / jnp.exp(logsd)) ** 2).sum(-1).mean()
