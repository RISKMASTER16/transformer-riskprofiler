"""
dls_study.py
============
Demographic layer, survey vignettes, and joint calibration of the DLS-style
model to the three headline findings of the Mumbai study.

The central analytic move here is the DECOMPOSITION. A demographic gap in
risk-taking can arise from two quite different sources:

  POSITIONAL  the group holds thinner buffers, so the identical gamble
              consumes a larger share of its resources
  PREFERENCE  the group weights an identical resource loss more heavily

Policy implications diverge sharply. A positional gap is closed by giving
people buffers; a preference gap is not. The model separates them because
the structural layer (resource surface) and behavioural layer (lam, sigma)
are independently parameterised.
"""

from __future__ import annotations
import numpy as np
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple
from scipy.optimize import brentq

import dls_core as C
from dls_core import (Household, Decision, PrefParams, BUFFER_CAPACITY_PRIOR,
                      IDX, p_take_risk, lambda_effective, pv_flow, wickets)

# ---------------------------------------------------------------------------
# 1. DEMOGRAPHIC STRUCTURE
# ---------------------------------------------------------------------------

INCOME_BANDS = ["low", "mid", "high"]
AGE_BANDS = {"18-34": (18, 34), "35-49": (35, 49), "50-65": (50, 65)}
GENDERS = ["F", "M"]

# Band definition. NOTE: the study reports a 'sub-INR 1.25 lakh' threshold
# without a period. Treated here as ANNUAL household income, which places it
# near the 30th percentile of Mumbai households. Because every quantity in
# the instrument is denominated in months-of-household-income, the absolute
# threshold affects the model ONLY through the buffer inventory it implies —
# so a redefinition to monthly would be absorbed by re-estimating Module 1
# rather than by rewriting the model.
BAND_CUTOFFS_INR_ANNUAL = {"low": (0, 125_000),
                           "mid": (125_000, 500_000),
                           "high": (500_000, np.inf)}

# Female buffer multipliers: the POSITIONAL channel.
# Lower own-name assets and formal credit, lower labour-force attachment and
# so lower employability, offset by a higher chance of a second earner in the
# household and marginally denser kin networks. These are priors to be
# replaced by Module 1; they are stated separately from preferences precisely
# so that the decomposition is auditable.
GENDER_BUFFER_MULT = {
    "F": np.array([0.75,   # liquid_savings
                   0.55,   # formal_credit
                   0.70,   # insurance
                   0.60,   # physical_assets
                   1.15,   # second_earner
                   1.05,   # social_network
                   0.75,   # employability
                   1.00]), # physical_health
    "M": np.ones(8),
}

# Age buffer multipliers: older households have accumulated more savings and
# assets but face declining health and employability. Without this the model
# would attribute the entire age gradient to recovery time.
def age_buffer_mult(age: float) -> np.ndarray:
    t = (age - 30.0) / 20.0
    return np.array([
        1.0 + 0.45 * t,    # savings accumulate
        1.0 + 0.30 * t,    # credit history improves
        1.0 + 0.35 * t,    # insurance take-up rises
        1.0 + 0.55 * t,    # assets accumulate
        1.0 - 0.10 * t,    # second earner slightly less likely late
        1.0 + 0.10 * t,    # network thickens modestly
        1.0 - 0.35 * t,    # employability declines
        1.0 - 0.40 * t,    # health declines
    ]).clip(0.05, None)


# Subjective discount rate on future income flows, by band.
# delta_low is CALIBRATED; the others are held at plausible priors.
DELTA = {"low": 0.35, "mid": 0.18, "high": 0.12}


def make_household(age: float, gender: str, band: str,
                   pp: PrefParams = None,
                   gender_lam_mult: Dict[str, float] = None) -> Household:
    """Build a household with positional (buffer) and preference layers."""
    base = np.array(BUFFER_CAPACITY_PRIOR[band], dtype=float)
    cap = base * GENDER_BUFFER_MULT[gender] * age_buffer_mult(age)
    pp = deepcopy(pp) if pp is not None else PrefParams()
    if gender_lam_mult:
        pp.lam = pp.lam * gender_lam_mult.get(gender, 1.0)
    return Household(age=age, gender=gender, income_band=band,
                     capacity=list(cap), pp=pp)


# ---------------------------------------------------------------------------
# 2. THE COHORT (n = 1000, quota design)
# ---------------------------------------------------------------------------

def build_cohort(n: int = 1000, seed: int = 20240501) -> List[dict]:
    """
    Quota sample: 3 income x 3 age x 2 gender = 18 cells.

    At n=1000 that is ~56 per cell, which is thin for cell-level estimation.
    The write-up therefore specifies partial pooling (hierarchical priors)
    rather than 18 independent cell estimates.
    """
    rng = np.random.default_rng(seed)
    # Approximate Mumbai household marginals; quotas balanced within them.
    band_share = {"low": 0.34, "mid": 0.45, "high": 0.21}
    ageb_share = {"18-34": 0.40, "35-49": 0.36, "50-65": 0.24}
    rows = []
    for band, bs in band_share.items():
        for ab, as_ in ageb_share.items():
            for g in GENDERS:
                k = int(round(n * bs * as_ * 0.5))
                lo, hi = AGE_BANDS[ab]
                for _ in range(k):
                    rows.append(dict(age=float(rng.integers(lo, hi + 1)),
                                     gender=g, band=band, age_band=ab))
    return rows


# ---------------------------------------------------------------------------
# 3. VIGNETTES  (the Module 4 discrete-choice tasks)
# ---------------------------------------------------------------------------

POWER_DAMAGE_SHARE = 0.85   # share of the network buffer destroyed


def vignette_power_vs_income(hh: Household, delta: float,
                             raise_frac: float = 0.50) -> Decision:
    """
    THE CRITICAL ITEM behind the study's 4x finding.

    'A firm offers you a role paying {raise_frac:.0%} more. Taking it means
     leaving your neighbourhood association and the standing you hold in it.'

    The pay rise is a FLOW, capitalised subjectively via pv_flow. The standing
    loss is a CERTAIN, targeted destruction of the social_network buffer —
    which for a low-income household is simultaneously its insurance policy,
    so it also raises subsistence-ruin risk.
    """
    net = hh.capacity[IDX["social_network"]]
    pv = pv_flow(annual_mhi=12.0 * raise_frac, age=hh.age, delta=delta)
    return Decision(gain_mhi=pv, p_success=1.0,
                    certain_damage={"social_network": POWER_DAMAGE_SHARE * net},
                    label="power_vs_income")


def vignette_standard_gamble(hh: Household, stake_mhi: float = 2.0,
                             multiple: float = 2.5) -> Decision:
    """Benchmark 50-50 financial gamble used for cross-cell comparison."""
    return Decision(gain_mhi=multiple * stake_mhi, loss_mhi=stake_mhi,
                    p_success=0.5, label="standard_gamble")


# ---------------------------------------------------------------------------
# 4. GROUP RISK-AVERSION SUMMARIES
# ---------------------------------------------------------------------------

def collapse(cohort) -> List[dict]:
    """
    Collapse the cohort to unique (age, gender, band) design points with
    weights. 1000 rows reduce to ~275 distinct points, so every calibration
    evaluation costs a quarter as much with no change to any weighted mean.
    """
    from collections import Counter
    ctr = Counter((r["age"], r["gender"], r["band"]) for r in cohort)
    return [dict(age=a, gender=g, band=b, age_band=None, wt=float(n))
            for (a, g, b), n in ctr.items()]


def cohort_lambda(cohort, pp: PrefParams, glm: Dict[str, float],
                  rho_max: float, stake: float = 2.0,
                  positional_only: bool = False) -> np.ndarray:
    """
    lambda_eff for every design point.

    positional_only=True suppresses the preference channel (all groups share
    the male lam), isolating the share of any gap attributable to buffers.
    """
    C.RHO_MAX = rho_max
    out = np.empty(len(cohort))
    use_glm = None if positional_only else glm
    for i, r in enumerate(cohort):
        hh = make_household(r["age"], r["gender"], r["band"], pp, use_glm)
        out[i] = lambda_effective(hh, stake_mhi=stake)
    return out


def ratio(vals: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray,
          wt: np.ndarray = None) -> float:
    """Weighted ratio of group means, robust to an occasional infinite prize."""
    if wt is None:
        wt = np.ones_like(vals)
    def m(mask):
        v, w = vals[mask], wt[mask]
        ok = np.isfinite(v)
        return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))
    return m(mask_a) / m(mask_b)


# ---------------------------------------------------------------------------
# 5. JOINT CALIBRATION
# ---------------------------------------------------------------------------

TARGETS = {
    "anchor_lambda": 2.20,   # mid-band male aged 30, 2-MHI stake (lab anchor)
    "age_ratio":     1.37,   # over-35 vs under-35
    "gender_ratio":  1.23,   # female vs male
    "power_odds":    4.00,   # low band: odds of refusing the power-for-pay swap
}


def calibrate(cohort, verbose: bool = True) -> dict:
    """
    Solve for four parameters against four reported findings.

        lam_base   <- anchor_lambda
        rho_max    <- age_ratio
        gender_mult<- gender_ratio
        delta_low  <- power_odds

    Each target is dominated by one parameter but they interact, so the
    coordinate solves are iterated to a fixed point. Reported convergence
    diagnostics let a reader see whether the system is genuinely identified.
    """
    pp = PrefParams()
    st = dict(lam_base=1.30, rho_max=0.60, gmult=1.15, delta_low=0.35)

    grid = collapse(cohort)
    wt = np.array([r["wt"] for r in grid])
    g = np.array([r["gender"] for r in grid])
    ages = np.array([r["age"] for r in grid])
    mF, mM = g == "F", g == "M"
    mO, mU = ages >= 35, ages < 35
    cohort = grid

    def anchor(lam, rho):
        C.RHO_MAX = rho
        p = replace(pp, lam=lam)
        hh = make_household(30.0, "M", "mid", p, None)
        return lambda_effective(hh, 2.0)

    def age_r(lam, rho, gm):
        # The gender multiplier must be ACTIVE here. Calibrating the age
        # target with it switched off pins the ratio under a configuration
        # that is not the one shipped: it produced 1.370 during the solve but
        # 1.404 in the finished model.
        p = replace(pp, lam=lam)
        v = cohort_lambda(cohort, p, {"F": gm, "M": 1.0}, rho)
        return ratio(v, mO, mU, wt)

    def gender_r(lam, rho, gm):
        p = replace(pp, lam=lam)
        v = cohort_lambda(cohort, p, {"F": gm, "M": 1.0}, rho)
        return ratio(v, mF, mM, wt)

    def power_odds(lam, rho, gm, dlow):
        C.RHO_MAX = rho
        p = replace(pp, lam=lam)
        acc, ws = [], []
        for r in cohort:
            if r["band"] != "low":
                continue
            hh = make_household(r["age"], r["gender"], "low", p,
                                {"F": gm, "M": 1.0})
            d = vignette_power_vs_income(hh, dlow)
            acc.append(p_take_risk(hh, d))
            ws.append(r["wt"])
        pa = float(np.average(acc, weights=ws))
        return (1.0 - pa) / max(pa, 1e-9)

    hist = []
    DAMP = 0.5   # lam and rho both move the anchor, so undamped coordinate
                 # descent oscillates; halving each step converges cleanly.
    for it in range(10):
        prev = dict(st)
        st["lam_base"] = brentq(
            lambda L: anchor(L, st["rho_max"]) - TARGETS["anchor_lambda"],
            0.05, 20.0, xtol=1e-5)
        st["rho_max"] = brentq(
            lambda r: age_r(st["lam_base"], r, st["gmult"]) - TARGETS["age_ratio"],
            0.01, 0.97, xtol=1e-5)
        st["gmult"] = brentq(
            lambda gm: gender_r(st["lam_base"], st["rho_max"], gm)
            - TARGETS["gender_ratio"], 0.30, 6.0, xtol=1e-5)
        st["delta_low"] = brentq(
            lambda d: power_odds(st["lam_base"], st["rho_max"], st["gmult"], d)
            - TARGETS["power_odds"], 0.02, 3.0, xtol=1e-5)
        if it > 0:
            for k in ("lam_base", "rho_max", "gmult", "delta_low"):
                st[k] = prev[k] + DAMP * (st[k] - prev[k])
        hist.append(dict(st))
        if verbose:
            print(f"  iter {it}: lam={st['lam_base']:.4f} rho={st['rho_max']:.4f} "
                  f"gmult={st['gmult']:.4f} delta_low={st['delta_low']:.4f}")
        if it > 0 and all(abs(hist[-1][k] - hist[-2][k]) < 1e-3 for k in st):
            break

    C.RHO_MAX = st["rho_max"]
    st["achieved"] = {
        "anchor_lambda": anchor(st["lam_base"], st["rho_max"]),
        "age_ratio": age_r(st["lam_base"], st["rho_max"], st["gmult"]),
        "gender_ratio": gender_r(st["lam_base"], st["rho_max"], st["gmult"]),
        "power_odds": power_odds(st["lam_base"], st["rho_max"], st["gmult"],
                                 st["delta_low"]),
    }
    st["iterations"] = len(hist)
    return st
