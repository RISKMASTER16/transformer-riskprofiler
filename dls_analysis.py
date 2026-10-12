"""
dls_analysis.py
===============
Maps the study's factor inventory onto the buffer ladder, derives the
monetary heuristics, decomposes the demographic gaps into positional and
preference components, and produces the prediction tables.

Run after calibration (reads calib.json).
"""

from __future__ import annotations
import json
import numpy as np
from dataclasses import replace
from typing import Dict, List, Tuple
from scipy.optimize import brentq

import dls_core as C
from dls_core import (Household, Decision, PrefParams, IDX, BUFFER_NAMES,
                      p_take_risk, lambda_effective, prospect_value,
                      resource_deltas, ruin_deltas, p_ruin, ruin_capacity,
                      wickets, resources, pv_flow)
import dls_study as S
from dls_study import make_household, build_cohort, collapse, GENDER_BUFFER_MULT

CAL = json.load(open("calib.json"))
C.RHO_MAX = CAL["rho_max"]
PP = replace(PrefParams(), lam=CAL["lam_base"])
GLM = {"F": CAL["gmult"], "M": 1.0}
DELTA = {"low": CAL["delta_low"], "mid": 0.18, "high": 0.12}


# ---------------------------------------------------------------------------
# 1. FACTOR REGISTRY
# ---------------------------------------------------------------------------
# Every factor in the study's inventory is expressed as a change to the buffer
# ladder. Magnitudes are FRACTIONS OF THE HOUSEHOLD'S OWN BUFFER, not absolute
# rupee amounts. This is the single most important modelling choice in the
# registry: 'loss of professional reputation' does not destroy a fixed sum, it
# destroys most of whatever reputation that person had. Expressing it as a
# fraction is what makes the same factor automatically worth different amounts
# to different demographics, without any demographic term in the registry.

FACTORS: Dict[str, dict] = {
    # ---------------- PERSONAL — GAINS ----------------
    "fulfilment_satisfaction":  dict(dom="personal", dir="gain",
                                     frac={"physical_health": 0.20}),
    "sense_of_achievement":     dict(dom="personal", dir="gain",
                                     frac={"employability": 0.15,
                                           "physical_health": 0.10}),
    "quality_of_life_up":       dict(dom="personal", dir="gain",
                                     frac={"physical_health": 0.30}),
    "psych_health_up":          dict(dom="personal", dir="gain",
                                     frac={"physical_health": 0.25}),
    "life_expectancy_up":       dict(dom="personal", dir="gain",
                                     frac={"physical_health": 0.40}),
    "education_skills":         dict(dom="personal", dir="gain",
                                     frac={"employability": 0.50}),
    "safety_of_family_up":      dict(dom="personal", dir="gain",
                                     frac={"physical_health": 0.30,
                                           "social_network": 0.15}),
    # ---------------- PERSONAL — LOSSES ----------------
    "emotional_stress":         dict(dom="personal", dir="loss",
                                     frac={"physical_health": 0.25}),
    "strained_relationships":   dict(dom="personal", dir="loss",
                                     frac={"social_network": 0.35}),
    "quality_of_life_down":     dict(dom="personal", dir="loss",
                                     frac={"physical_health": 0.30}),
    "confidence_selfesteem_dn": dict(dom="personal", dir="loss",
                                     frac={"employability": 0.25,
                                           "physical_health": 0.10}),
    "psych_health_down":        dict(dom="personal", dir="loss",
                                     frac={"physical_health": 0.35}),
    "hazardous_conditions":     dict(dom="personal", dir="loss",
                                     frac={"physical_health": 0.45}),
    "safety_of_family_down":    dict(dom="personal", dir="loss",
                                     frac={"physical_health": 0.35,
                                           "social_network": 0.20}),
    # ---------------- SOCIAL — GAINS ----------------
    "network_union_access":     dict(dom="social", dir="gain",
                                     frac={"social_network": 0.45}),
    "social_capital_up":        dict(dom="social", dir="gain",
                                     frac={"social_network": 0.40}),
    "leadership_roles":         dict(dom="social", dir="gain",
                                     frac={"social_network": 0.30,
                                           "employability": 0.25}),
    "recognition_respect":      dict(dom="social", dir="gain",
                                     frac={"social_network": 0.25}),
    "family_community_impact":  dict(dom="social", dir="gain",
                                     frac={"social_network": 0.30}),
    "geography_up":             dict(dom="social", dir="gain",
                                     frac={"physical_health": 0.20,
                                           "social_network": 0.10}),
    # ---------------- SOCIAL — LOSSES ----------------
    "reputation_loss":          dict(dom="social", dir="loss",
                                     frac={"social_network": 0.50,
                                           "employability": 0.30}),
    "influence_credibility_dn": dict(dom="social", dir="loss",
                                     frac={"social_network": 0.40}),
    "demotion":                 dict(dom="social", dir="loss",
                                     frac={"employability": 0.35,
                                           "social_network": 0.20}),
    "career_progression_block": dict(dom="social", dir="loss",
                                     frac={"employability": 0.40}),
    "family_dynamics_down":     dict(dom="social", dir="loss",
                                     frac={"social_network": 0.30,
                                           "second_earner": 0.20}),
    "geography_down":           dict(dom="social", dir="loss",
                                     frac={"physical_health": 0.20,
                                           "social_network": 0.15}),
    "societal_power_loss":      dict(dom="social", dir="loss",
                                     frac={"social_network": 0.85}),
    # ---------------- FINANCIAL — GAINS (flows: mhi_pa) ----------------
    "income_up_50pct":          dict(dom="financial", dir="gain", mhi_pa=6.0),
    "bonus_commission":         dict(dom="financial", dir="gain", mhi=1.5),
    "cost_savings":             dict(dom="financial", dir="gain", mhi=0.8),
    "promotion":                dict(dom="financial", dir="gain", mhi_pa=3.0,
                                     frac={"employability": 0.30,
                                           "social_network": 0.20}),
    "better_job_access":        dict(dom="financial", dir="gain",
                                     frac={"employability": 0.45}),
    "microloan_access":         dict(dom="financial", dir="gain",
                                     frac={"formal_credit": 1.00}),
    "financial_literacy":       dict(dom="financial", dir="gain",
                                     frac={"formal_credit": 0.35,
                                           "employability": 0.15}),
    # ---------------- FINANCIAL — LOSSES ----------------
    "savings_investment_loss":  dict(dom="financial", dir="loss",
                                     frac={"liquid_savings": 0.80}),
    "debt_accumulation":        dict(dom="financial", dir="loss", mhi=3.0),
    "bankruptcy":               dict(dom="financial", dir="loss", mhi=9.0,
                                     frac={"formal_credit": 1.00,
                                           "social_network": 0.40}),
    "income_down_unemployment": dict(dom="financial", dir="loss", mhi_pa=-6.0),
    "living_expenses_up":       dict(dom="financial", dir="loss", mhi_pa=-1.8),
    "no_safety_net":            dict(dom="financial", dir="loss",
                                     frac={"insurance": 1.00}),
    "predatory_lending":        dict(dom="financial", dir="loss", mhi=2.5,
                                     frac={"formal_credit": 0.60}),
}


def factor_bundle(hh: Household, name: str) -> Tuple[Dict[str, float], float]:
    """
    Resolve a registry entry into (buffer_delta_dict, cash_mhi) for THIS
    household. Fractional entries are multiplied by the household's own
    buffer depth; flow entries (mhi_pa) are capitalised with its band's
    subjective discount rate.
    """
    f = FACTORS[name]
    dmg = {}
    for b, fr in f.get("frac", {}).items():
        dmg[b] = fr * hh.capacity[IDX[b]]
    cash = float(f.get("mhi", 0.0))
    if "mhi_pa" in f:
        cash += pv_flow(abs(f["mhi_pa"]), hh.age, DELTA[hh.income_band]) \
            * (1.0 if f["mhi_pa"] > 0 else 1.0)
    return dmg, cash


# ---------------------------------------------------------------------------
# 2. MONETARY EQUIVALENTS (the heuristic the survey must recover)
# ---------------------------------------------------------------------------

def wta(hh: Household, name: str, hi: float = 4000.0) -> float:
    """
    Willingness-to-accept, in MHI: the certain cash sum that exactly offsets
    suffering this factor, i.e. the G solving V = 0 for a sure loss.

    This is the model's prediction of what Module 3 of the survey should
    elicit. Comparing the two is the model's principal falsification test.
    """
    dmg, cash = factor_bundle(hh, name)

    def f(G):
        return prospect_value(hh, Decision(gain_mhi=G, p_success=1.0,
                                           certain_loss_mhi=cash,
                                           certain_damage=dmg))
    if f(hi) < 0:
        return float("inf")
    lo, high = 0.0, hi
    for _ in range(44):
        mid = 0.5 * (lo + high)
        if f(mid) < 0:
            lo = mid
        else:
            high = mid
        if high - lo < 1e-7:
            break
    return 0.5 * (lo + high)


def wtp(hh: Household, name: str, hi: float = 4000.0) -> float:
    """
    Willingness-to-pay, in MHI, for a sure gain factor. Solves for the cash
    outflow that exactly cancels the benefit.

    WTA and WTP are NOT equal in this model, and should not be: the loss
    branch carries lam and the ruin term. The observed WTA/WTP gap is
    therefore a prediction, not a nuisance — see the disparity table.
    """
    dmg, cash = factor_bundle(hh, name)

    def f(L):
        return prospect_value(hh, Decision(gain_mhi=cash, p_success=1.0,
                                           gain_credit=dmg,
                                           certain_loss_mhi=L))
    if f(0.0) <= 0:
        return 0.0
    if f(hi) > 0:
        return float("inf")
    lo, high = 0.0, hi
    for _ in range(44):
        mid = 0.5 * (lo + high)
        if f(mid) > 0:
            lo = mid
        else:
            high = mid
        if high - lo < 1e-7:
            break
    return 0.5 * (lo + high)


# ---------------------------------------------------------------------------
# 3. DECOMPOSITION: positional vs preference
# ---------------------------------------------------------------------------

def decompose_gender(grid, wt) -> dict:
    """
    Split the female/male risk-aversion gap.

      TOTAL       both channels active
      POSITIONAL  female buffer multipliers on, preference multiplier off
      PREFERENCE  total / positional

    A positional gap is closed by asset and credit policy. A preference gap
    is not. The two carry opposite policy implications, which is why the
    headline '23% more risk-averse' is uninformative until it is split.
    """
    g = np.array([r["gender"] for r in grid])
    mF, mM = g == "F", g == "M"
    v_tot = S.cohort_lambda(grid, PP, GLM, CAL["rho_max"])
    v_pos = S.cohort_lambda(grid, PP, {"F": 1.0, "M": 1.0}, CAL["rho_max"])
    tot = S.ratio(v_tot, mF, mM, wt)
    pos = S.ratio(v_pos, mF, mM, wt)
    return dict(total=tot, positional=pos, preference=tot / pos,
                pos_share=np.log(pos) / np.log(tot))


def decompose_age(grid, wt) -> dict:
    """
    Split the over-35/under-35 gap into the buffer channel (age_buffer_mult)
    and the recovery channel (rho declining with horizon), by re-running with
    recovery frozen at its age-30 value.
    """
    ages = np.array([r["age"] for r in grid])
    mO, mU = ages >= 35, ages < 35
    v_tot = S.cohort_lambda(grid, PP, GLM, CAL["rho_max"])
    tot = S.ratio(v_tot, mO, mU, wt)

    # Freeze recovery: monkey-patch to a constant at the age-30 level.
    rho30 = CAL["rho_max"] * (1.0 - np.exp(-(75.0 - 30.0) / C.U_REC))
    orig = C.recovery
    C.recovery = lambda age, p=None: rho30
    try:
        v_pos = S.cohort_lambda(grid, PP, GLM, CAL["rho_max"])
        pos = S.ratio(v_pos, mO, mU, wt)
    finally:
        C.recovery = orig
    return dict(total=tot, buffer_channel=pos, recovery_channel=tot / pos,
                buffer_share=np.log(pos) / np.log(tot))


# ---------------------------------------------------------------------------
# 4. HEURISTIC MULTIPLIER TABLE
# ---------------------------------------------------------------------------

CELLS = [(b, a, g) for b in ["low", "mid", "high"]
         for a in [26, 42, 56] for g in ["F", "M"]]
AGE_LABEL = {26: "18-34", 42: "35-49", 56: "50-65"}


def heuristic_table(factors: List[str]) -> Tuple[np.ndarray, List[str]]:
    """WTA in MHI for each factor x demographic cell."""
    M = np.zeros((len(factors), len(CELLS)))
    for i, fn in enumerate(factors):
        for j, (b, a, g) in enumerate(CELLS):
            hh = make_household(a, g, b, PP, GLM)
            M[i, j] = wta(hh, fn)
    labels = [f"{b}|{AGE_LABEL[a]}|{g}" for b, a, g in CELLS]
    return M, labels


def fit_multipliers(M: np.ndarray) -> dict:
    """
    Fit  ln WTA_fc = a_f + b_income + c_age + d_gender  by OLS.

    Gives the pocket heuristic the study needs: a single base value per
    factor, scaled by three demographic multipliers. R^2 reports how much
    of the model's own structure this simple form captures — if it is low,
    the interactions matter and the lookup table must be used instead.
    """
    nf, nc = M.shape
    rows, y = [], []
    binc = {"low": 0, "mid": 1, "high": 2}
    bage = {26: 0, 42: 1, 56: 2}
    for i in range(nf):
        for j, (b, a, g) in enumerate(CELLS):
            if not np.isfinite(M[i, j]) or M[i, j] <= 0:
                continue
            r = np.zeros(nf + 2 + 2 + 1)
            r[i] = 1.0
            if binc[b] > 0:
                r[nf + binc[b] - 1] = 1.0
            if bage[a] > 0:
                r[nf + 2 + bage[a] - 1] = 1.0
            if g == "F":
                r[nf + 4] = 1.0
            rows.append(r)
            y.append(np.log(M[i, j]))
    X = np.array(rows)
    y = np.array(y)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1.0 - np.var(resid) / np.var(y)
    return dict(
        income_mult={"low": 1.0,
                     "mid": float(np.exp(beta[nf])),
                     "high": float(np.exp(beta[nf + 1]))},
        age_mult={"18-34": 1.0,
                  "35-49": float(np.exp(beta[nf + 2])),
                  "50-65": float(np.exp(beta[nf + 3]))},
        gender_mult={"M": 1.0, "F": float(np.exp(beta[nf + 4]))},
        base={f: float(np.exp(beta[i])) for i, f in enumerate(FACTOR_ORDER)},
        r2=float(r2))


FACTOR_ORDER = [f for f, v in FACTORS.items() if v["dir"] == "loss"]


def fit_multipliers_by_domain(verbose: bool = True) -> dict:
    """
    The pooled log-linear fit achieves only R^2 = 0.47, because the income
    multiplier has OPPOSITE SIGNS across domains: financial losses are worth
    far more (in months of income) to the poor, while social and personal
    losses are worth more to the rich. A single income multiplier averages
    these into a number that is wrong for every factor.

    Fitting one set of multipliers per domain repairs this. The result is the
    form the field team should actually carry.
    """
    out = {}
    for dom in ["personal", "social", "financial"]:
        facs = [f for f, v in FACTORS.items()
                if v["dir"] == "loss" and v["dom"] == dom]
        M = np.zeros((len(facs), len(CELLS)))
        for i, fn in enumerate(facs):
            for j, (b, a, g) in enumerate(CELLS):
                M[i, j] = wta(make_household(a, g, b, PP, GLM), fn)
        nf = len(facs)
        binc = {"low": 0, "mid": 1, "high": 2}
        bage = {26: 0, 42: 1, 56: 2}
        rows, y = [], []
        for i in range(nf):
            for j, (b, a, g) in enumerate(CELLS):
                if not np.isfinite(M[i, j]) or M[i, j] <= 0:
                    continue
                r = np.zeros(nf + 5)
                r[i] = 1.0
                if binc[b] > 0:
                    r[nf + binc[b] - 1] = 1.0
                if bage[a] > 0:
                    r[nf + 2 + bage[a] - 1] = 1.0
                if g == "F":
                    r[nf + 4] = 1.0
                rows.append(r); y.append(np.log(M[i, j]))
        X, y = np.array(rows), np.array(y)
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        r2 = 1.0 - np.var(y - X @ beta) / np.var(y)
        out[dom] = dict(
            income={"low": 1.0, "mid": float(np.exp(beta[nf])),
                    "high": float(np.exp(beta[nf + 1]))},
            age={"18-34": 1.0, "35-49": float(np.exp(beta[nf + 2])),
                 "50-65": float(np.exp(beta[nf + 3]))},
            gender={"M": 1.0, "F": float(np.exp(beta[nf + 4]))},
            base={f: float(np.exp(beta[i])) for i, f in enumerate(facs)},
            r2=float(r2), n_factors=nf)
        if verbose:
            print(f"[{dom}]  R^2={r2:.3f}")
            print("   income:", {k: round(v, 3) for k, v in out[dom]['income'].items()})
            print("   age   :", {k: round(v, 3) for k, v in out[dom]['age'].items()})
            print("   gender:", {k: round(v, 3) for k, v in out[dom]['gender'].items()})
    return out
