"""Simulator wrapper: households with latent caution (u) and constraint (v) shifts.

u scales the loss-aversion weight: lambda -> lambda * exp(u)
v scales every buffer:            c      -> c * exp(v)      (low v = thin buffers)
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dls"))
import numpy as np
from dataclasses import replace
import dls_core as C, dls_analysis as A
from dls_core import Household, Decision, p_take_risk, IDX

BANDS = ["low", "mid", "high"]; BAND_P = [0.34, 0.45, 0.21]
SIG_U = SIG_V = 0.4          # prior s.d. of log caution / log buffer multipliers
KMAX = 24                    # choices per household

def base_household(age, female, band):
    return A.make_household(float(age), "F" if female else "M", BANDS[band], A.PP, A.GLM)

def shifted(base, u, v):
    cap = list(np.asarray(base.capacity) * np.exp(v))
    pp = replace(base.pp, lam=base.pp.lam * np.exp(u))
    return Household(age=base.age, gender=base.gender, income_band=base.income_band,
                     capacity=cap, pp=pp)

def sample_decisions(rng, k):
    """Stake log-uniform [0.25, 6] MHI, gain multiple log-uniform [1.2, 10],
    success prob uniform [0.25, 0.85], 35% of gambles also damage the social network."""
    stake = np.exp(rng.uniform(np.log(0.25), np.log(6.0), k))
    mult = np.exp(rng.uniform(np.log(1.2), np.log(10.0), k))
    p = rng.uniform(0.25, 0.85, k)
    social = rng.random(k) < 0.35
    frac = np.where(social, rng.uniform(0.3, 0.9, k), 0.0)
    return np.stack([stake, mult, p, frac], 1)

def probs(hh, D):
    out = np.empty(len(D))
    net = hh.capacity[IDX["social_network"]]
    for i, (s, m, p, f) in enumerate(D):
        dmg = {"social_network": f * net} if f > 0 else {}
        out[i] = p_take_risk(hh, Decision(gain_mhi=m * s, loss_mhi=s, p_success=p, loss_damage=dmg))
    return out

def features(D, y):
    """Per-choice token: log stake, log multiple, p, social fraction, social flag, choice."""
    return np.stack([np.log(D[:, 0]), np.log(D[:, 1]), D[:, 2], D[:, 3],
                     (D[:, 3] > 0).astype(float), y], 1)

def demo_vec(age, female, band):
    v = np.zeros(5); v[band] = 1; v[3] = (age - 40) / 15; v[4] = female
    return v
