"""
dls_core.py
===========
A DLS-style (Duckworth-Lewis-Stern) resource model of household risk-taking.

    CRICKET                        ->   HOUSEHOLD DECISION
    -----------------------------------------------------------------------
    Overs remaining      u         ->   Economic horizon remaining (years)
    Wickets in hand      w         ->   Shock-absorbing buffer depth
    Resource index R(u,w)          ->   Capacity to absorb a downside
    Losing a wicket                ->   Exhausting a buffer (savings, network)
    Par score S1 * R2/R1           ->   Par prize: gain needed to justify stake

All money is in MHI = Months of Household Income, making the instrument
scale-free across income bands.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# ---------------------------------------------------------------------------
# 0. BUFFER LADDER — the eight "wickets"
# ---------------------------------------------------------------------------

BUFFER_NAMES = [
    "liquid_savings",    # 0  most liquid: burned first
    "formal_credit",     # 1
    "insurance",         # 2
    "physical_assets",   # 3  gold, vehicle, property
    "second_earner",     # 4
    "social_network",    # 5  informal insurance; dominant for low income
    "employability",     # 6
    "physical_health",   # 7  least liquid: cannot be spent, only destroyed
]
NB = len(BUFFER_NAMES)
IDX = {n: i for i, n in enumerate(BUFFER_NAMES)}

# Only the first five are *spendable* against a financial shock. Social
# network, employability and health are stock buffers: they cannot be sold
# to pay a bill, but they can be destroyed by non-financial losses.
SPENDABLE = [0, 1, 2, 3, 4]

# Buffer depths in MHI by income band. PRIORS, to be replaced by Module 1.
#
# The low band is deliberately THIN in spendable buffers (1.95 MHI total) and
# THICK in social_network. That is the empirical signature of low-income urban
# Indian households: negligible formal savings, no insurance, but dense
# informal risk-sharing. It is also what makes the loss of social standing
# catastrophic for them and trivial for the high band — the mechanism behind
# the study's 4x finding.
BUFFER_CAPACITY_PRIOR: Dict[str, List[float]] = {
    #          sav  cred  ins  asset  2nd   net  empl  health
    "low":  [ 0.30, 0.15, 0.00, 0.50, 1.00, 3.00, 0.80, 1.50],
    "mid":  [ 3.00, 2.50, 2.00, 3.50, 3.00, 3.00, 3.00, 3.00],
    "high": [ 9.00, 8.00, 6.00,12.00, 5.00, 2.00, 6.00, 4.00],
}

S_BUFFER = 1.5   # MHI per "wicket unit" — sets curvature of depth -> wickets


def wickets(capacity) -> float:
    """
        w = SUM_k  ln(1 + c_k / s)

    Concave in each buffer's depth (the second month of savings is worth
    less than the first) and unbounded above, so wealth never saturates.
    Additive across buffers, so DIVERSITY is rewarded: eight buffers of
    1 MHI each beat one buffer of 8 MHI. That is the household analogue
    of wickets in hand.
    """
    c = np.asarray(capacity, dtype=float)
    return float(np.sum(np.log1p(np.clip(c, 0, None) / S_BUFFER)))


# ---------------------------------------------------------------------------
# 1. THE RESOURCE SURFACE  R(u, w)
# ---------------------------------------------------------------------------

@dataclass
class ResourceParams:
    xi:    float = 0.75    # concavity of the wicket curve F(w) = w^xi
    b_min: float = 0.030   # horizon-decay rate when buffers are deep
    b_max: float = 0.085   # horizon-decay rate when buffers are bare
    w_ref: float = 4.0     # scale over which b(w) relaxes
    age_h: float = 75.0    # age at which the economic horizon closes
    A:     float = 1.0     # normalisation, set by calibrate_scale()


def horizon(age, p: ResourceParams):
    """Economic 'overs remaining', in years."""
    return np.maximum(0.0, p.age_h - np.asarray(age, dtype=float))


def F_wickets(w, p: ResourceParams):
    """Concave, increasing, unbounded buffer curve."""
    return np.power(np.maximum(w, 1e-9), p.xi)


def b_decay(w, p: ResourceParams):
    """
    Horizon-decay rate, DECREASING in buffer depth.

    A household with almost no buffers cannot wait out a shock, so extra
    years of horizon do it little good: its resource curve saturates fast
    (large b). This mirrors DLS, where a side with one wicket left cannot
    exploit the overs it still has.
    """
    return p.b_min + (p.b_max - p.b_min) * np.exp(-np.asarray(w, float) / p.w_ref)


def resources(age, w, p: ResourceParams = None):
    """
        R(u, w) = A * F(w) * [1 - exp(-b(w) * u)]

    The structural analogue of the DLS resource table. Monotonicity and
    concavity are asserted numerically by validate().
    """
    if p is None:
        p = RP_DEFAULT
    u = horizon(age, p)
    return p.A * F_wickets(w, p) * (1.0 - np.exp(-b_decay(w, p) * u))


def calibrate_scale(p: ResourceParams, ref_age: float = 30.0,
                    ref_band: str = "mid") -> ResourceParams:
    """Set A so the reference household (age 30, mid band) scores 100."""
    w_r = wickets(BUFFER_CAPACITY_PRIOR[ref_band])
    p1 = ResourceParams(**{**p.__dict__, "A": 1.0})
    raw = float(resources(ref_age, w_r, p1))
    return ResourceParams(**{**p.__dict__, "A": 100.0 / raw})


def validate(p: ResourceParams, verbose: bool = False) -> bool:
    """
    Assert the surface behaves like a DLS table: strictly increasing in
    wickets and in horizon, and concave in wickets. Fails loudly if a
    parameter change breaks the economics.
    """
    ok = True
    ws = np.linspace(0.3, 16.0, 500)
    ages = np.arange(18.0, 70.0, 0.5)   # operating range of the survey
    for a in ages:
        R = resources(a, ws, p)
        if np.any(np.diff(R) <= 0):
            ok = False
            if verbose:
                print(f"  NON-MONOTONE in w at age {a}")
            break
        if np.max(np.diff(R, 2)) > 1e-6 * float(np.max(R)):
            ok = False
            if verbose:
                print(f"  NON-CONCAVE in w at age {a}")
            break
    if ok:
        for w in np.linspace(0.5, 16.0, 80):
            R = resources(ages, w, p)
            if np.any(np.diff(R) >= 0):
                ok = False
                if verbose:
                    print(f"  NON-MONOTONE in age at w={w:.2f}")
                break
    return ok


# ---------------------------------------------------------------------------
# 2. SHOCKS: rupees and non-monetary events -> wicket losses
# ---------------------------------------------------------------------------

def absorb_financial(capacity: List[float], loss_mhi: float) -> List[float]:
    """
    Burn a financial loss through the SPENDABLE buffers, most liquid first.

    Residue beyond all spendable buffers is an overhang (distress debt),
    charged against employability, health and network — which is what
    unpayable debt actually consumes: distress labour, untreated illness,
    and exhausted goodwill. This is the model's 'all out' regime.
    """
    cap = list(np.asarray(capacity, dtype=float))
    rem = float(loss_mhi)
    for k in SPENDABLE:
        if rem <= 0:
            break
        take = min(cap[k], rem)
        cap[k] -= take
        rem -= take
    if rem > 0:
        for k in (IDX["employability"], IDX["physical_health"],
                  IDX["social_network"]):
            take = min(cap[k], rem)
            cap[k] -= take
            rem -= take
    return cap


def apply_damage(capacity: List[float], damage: Dict[str, float]) -> List[float]:
    """
    Apply a TARGETED, non-financial loss to named buffers.

    This is how the study's personal / social / professional factors enter.
    'Loss of professional reputation' is not a rupee outflow; it is the
    destruction of the social_network and employability buffers.
    """
    cap = list(np.asarray(capacity, dtype=float))
    for name, amt in damage.items():
        k = IDX[name]
        cap[k] = max(0.0, cap[k] - amt)
    return cap


# How a windfall is deployed across the buffer ladder. A large gain does not
# merely swell the savings account: households buy insurance, clear credit,
# fund education (employability) and treat deferred illness. Restricting the
# windfall to savings alone would make the upside saturate logarithmically
# and produce absurd par prizes; this allocation is both more realistic and
# what keeps gains and losses on a comparable footing.
WINDFALL_SPLIT = np.array([0.30,   # liquid_savings
                           0.10,   # formal_credit (repayment restores headroom)
                           0.12,   # insurance
                           0.20,   # physical_assets
                           0.03,   # second_earner
                           0.05,   # social_network
                           0.10,   # employability
                           0.10])  # physical_health
assert abs(WINDFALL_SPLIT.sum() - 1.0) < 1e-12


def apply_gain(capacity: List[float], gain_mhi: float,
               credit: Optional[Dict[str, float]] = None) -> List[float]:
    """
    Deploy a windfall across the buffer ladder using WINDFALL_SPLIT.
    `credit` adds targeted non-monetary gains (a promotion raises
    employability and network directly).

    Diminishing returns remain automatic: ln(1 + c/s) is concave, so the
    tenth month of savings adds far fewer wicket units than the first.
    """
    cap = np.asarray(capacity, dtype=float) + WINDFALL_SPLIT * float(gain_mhi)
    cap = list(cap)
    for name, amt in (credit or {}).items():
        cap[IDX[name]] += amt
    return cap


# ---------------------------------------------------------------------------
# 3. PREFERENCE LAYER (cumulative prospect theory)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3a. THE SUBSISTENCE FLOOR — the 'all out' absorbing state
# ---------------------------------------------------------------------------

S_RUIN = 1.0   # MHI; scale of the subsistence gate


NET_LIQUIDITY = 0.70   # share of the social network usable as emergency cash


def spendable(capacity) -> float:
    """Buffers that can actually be converted to cash this month."""
    c = np.asarray(capacity, dtype=float)
    return float(np.sum(c[SPENDABLE]))


def ruin_capacity(capacity) -> float:
    """
    Cash a household could raise in an emergency.

    Includes a discounted share of the SOCIAL NETWORK. For a low-income
    Mumbai household the network is not a sentiment, it is the insurance
    product they actually hold: interest-free borrowing from kin, shared
    meals, a place to sleep. Excluding it from the ruin calculation makes the
    destruction of social standing costless in the model, which is precisely
    the effect the study set out to measure.
    """
    c = np.asarray(capacity, dtype=float)
    return spendable(c) + NET_LIQUIDITY * c[IDX["social_network"]]


def p_ruin(capacity) -> float:
    """
    Probability the household cannot meet next month's obligations.

        P_ruin = exp( - spendable / S_RUIN )

    In cricket, losing the last wicket ends the innings: no quantity of
    remaining overs can help you. Nothing in a smooth resource surface
    reproduces that discontinuity, and without it the model gets the income
    gradient BACKWARDS — the poor appear risk-tolerant, because at low buffer
    depth the log curve is steep in the UP direction too.

    This term restores the absorbing state. It is negligible above ~4 MHI of
    spendable buffer (P < 0.02) and approaches 1 as buffers vanish, so it
    bites for exactly the households for whom subsistence is a live threat.
    """
    return float(np.exp(-ruin_capacity(capacity) / S_RUIN))


@dataclass
class PrefParams:
    """
    Behavioural layer. Note two deliberate departures from textbook CPT:

    alpha = 1.0 : the value function is LINEAR IN RESOURCES. All diminishing
        sensitivity already lives in the resource surface (log depth -> wickets,
        then w^xi). Re-applying a power here would be a third, unidentified
        concavity and it inflated par prizes by an order of magnitude in
        testing. Resources are the correctly curved argument; utility over
        them is linear by construction.

    lam : NOT comparable to the Tversky-Kahneman 2.25. It is an internal
        component, applied AFTER the recovery term has already removed the
        healable share of a loss (rho ~ 0.66 at calibration), and it operates
        on resource fractions rather than on money. Calibration returns
        lam ~ 3.8, which sounds high only if misread as total loss aversion.
        The quantity comparable to the literature is lambda_eff at the
        reference household, which is pinned to 2.2 by construction.
        Do not interpret lam on its own.
    """
    alpha: float = 1.00   # linear in resources — see docstring
    lam:   float = 3.79   # internal component — see docstring, not total
    sigma: float = 0.65   # Prelec probability-weighting curvature
    theta: float = 0.012  # status-quo / inertia threshold
    tau:   float = 0.10   # logit temperature (choice noise)
    kappa: float = 0.55   # weight on the change in subsistence-ruin risk


def prelec(p: float, sigma: float) -> float:
    """Prelec (1998) one-parameter probability weighting."""
    p = float(np.clip(p, 1e-9, 1 - 1e-9))
    return float(np.exp(-((-np.log(p)) ** sigma)))


def pv_flow(annual_mhi: float, age: float, delta: float,
            cap_years: float = 15.0, age_h: float = 65.0) -> float:
    """
    Subjective present value of a PERMANENT income change, in MHI.

    A '50% pay rise' is a flow, not a lump sum, and cannot be compared with a
    stock of buffers until it is capitalised. Households do not capitalise at
    the market rate: they apply a steep subjective discount `delta`, and they
    truncate the horizon at `cap_years` because promised future income is not
    credible. Both are far more severe at low incomes, where income is
    volatile and no commitment device exists — delta is therefore estimated
    per income band rather than imposed globally.
    """
    n = float(np.clip(min(cap_years, age_h - age), 0.0, None))
    if n <= 0:
        return 0.0
    return annual_mhi * (1.0 - (1.0 + delta) ** (-n)) / delta


# ---------------------------------------------------------------------------
# 4. HOUSEHOLD AND DECISION
# ---------------------------------------------------------------------------

@dataclass
class Household:
    age: float
    gender: str            # 'F' | 'M'
    income_band: str       # 'low' | 'mid' | 'high'
    capacity: List[float] = field(default_factory=list)
    rp: ResourceParams = None
    pp: PrefParams = field(default_factory=PrefParams)

    def __post_init__(self):
        if self.rp is None:
            self.rp = RP_DEFAULT
        if len(self.capacity) == 0:
            self.capacity = list(BUFFER_CAPACITY_PRIOR[self.income_band])

    @property
    def w(self) -> float:
        return wickets(self.capacity)

    @property
    def R(self) -> float:
        return float(resources(self.age, self.w, self.rp))


@dataclass
class Decision:
    """
    With probability p_success the household receives `gain_mhi` (plus any
    `gain_credit` to named buffers); otherwise it suffers `loss_mhi` (plus
    `loss_damage`). `certain_*` costs are incurred either way — this is how
    a decision can carry a sure social cost alongside an uncertain
    financial upside.
    """
    gain_mhi: float = 0.0
    loss_mhi: float = 0.0
    p_success: float = 0.5
    gain_credit: Dict[str, float] = field(default_factory=dict)
    loss_damage: Dict[str, float] = field(default_factory=dict)
    certain_loss_mhi: float = 0.0
    certain_damage: Dict[str, float] = field(default_factory=dict)
    label: str = ""


# ---------------------------------------------------------------------------
# 5. RESOURCE DELTAS AND THE CHOICE MODEL
# ---------------------------------------------------------------------------

RHO_MAX = 0.60    # max share of a loss that can be rebuilt over a full career
U_REC   = 30.0    # years; time-scale of buffer regeneration


def recovery(age: float, p: ResourceParams = None) -> float:
    """
    Fraction of a destroyed buffer a household can expect to REBUILD over
    its remaining horizon.

        rho(u) = RHO_MAX * (1 - exp(-u / U_REC))

    WHY THIS EXISTS. Age enters R multiplicatively through the horizon
    bracket, so in any ratio like dR_loss / R it cancels exactly — testing
    confirmed a flat age gradient (1.84 at 25 vs 1.82 at 55) without this
    term. Scale effects cannot generate age-varying risk attitudes; only a
    SHAPE effect can.

    The shape effect is recovery time. A 25-year-old who loses six months of
    savings is poorer for a while; a 58-year-old is poorer permanently. The
    young therefore evaluate the down branch at a partially healed position.

    Applied to LOSSES ONLY. This is an identifying assumption, not a
    derivation: it encodes 'time heals shocks, but a windfall is a stock you
    simply keep'. It should be tested against the observed age gradient in
    Module 4, and RHO_MAX / U_REC re-estimated from it.
    """
    if p is None:
        p = RP_DEFAULT
    u = float(horizon(age, p))
    return RHO_MAX * (1.0 - np.exp(-u / U_REC))


def _R_of(cap, hh: Household) -> float:
    return float(resources(hh.age, wickets(cap), hh.rp))


def resource_deltas(hh: Household, d: Decision) -> Tuple[float, float, float]:
    """
    Return (dR_gain, dR_loss, dR_certain), all non-negative, by re-evaluating
    the resource surface after the buffer ladder is burned or refilled.
    This is where a rupee becomes a resource.

    Convexity of loss is EMERGENT, not assumed: because F(w) is concave and
    w is concave in buffer depth, the same rupee loss strips far more
    resource from a thin household than from a thick one.
    """
    base = hh.capacity
    R0 = _R_of(base, hh)

    # Sure component first: it shifts the reference point.
    cap_c = apply_damage(absorb_financial(base, d.certain_loss_mhi),
                         d.certain_damage)
    Rc = _R_of(cap_c, hh)
    dR_certain = max(0.0, R0 - Rc)

    # Uncertain branches evaluated FROM the post-sure-cost position.
    cap_up = apply_gain(cap_c, d.gain_mhi, d.gain_credit)
    dR_gain = max(0.0, _R_of(cap_up, hh) - Rc)

    cap_dn = apply_damage(absorb_financial(cap_c, d.loss_mhi), d.loss_damage)
    # Partial healing over the remaining horizon: the young evaluate the down
    # branch at a position they expect to rebuild toward.
    rho = recovery(hh.age, hh.rp)
    cap_eff = [cc - (1.0 - rho) * (cc - cd) for cc, cd in zip(cap_c, cap_dn)]
    dR_loss = max(0.0, Rc - _R_of(cap_eff, hh))

    return dR_gain, dR_loss, dR_certain


def ruin_deltas(hh: "Household", d: "Decision") -> Tuple[float, float]:
    """
    Change in subsistence-ruin probability on the down branch and on the
    sure component, relative to the household's current position.
    """
    base = hh.capacity
    P0 = p_ruin(base)
    cap_c = apply_damage(absorb_financial(base, d.certain_loss_mhi),
                         d.certain_damage)
    Pc = p_ruin(cap_c)
    cap_dn = apply_damage(absorb_financial(cap_c, d.loss_mhi), d.loss_damage)
    # Ruin is a SHORT-RUN event: it is judged at the unhealed position,
    # because a household cannot pay next month's rent out of future recovery.
    Pd = p_ruin(cap_dn)
    return max(0.0, Pd - Pc), max(0.0, Pc - P0)


def prospect_value(hh: Household, d: Decision) -> float:
    """
        V = w+(p)*(dRg/R)^a - lam*w-(1-p)*(dRl/R)^a - lam*(dRc/R)^a

    Two separable sources of risk aversion:
      (i)  STRUCTURAL  — dRl/R is convex in the rupee loss, via F(w)
      (ii) BEHAVIOURAL — lam, alpha, sigma
    The decomposition in the write-up exploits exactly this separability.
    """
    R0 = max(hh.R, 1e-9)
    dRg, dRl, dRc = resource_deltas(hh, d)
    a = hh.pp.alpha

    g = (dRg / R0) ** a
    l = (dRl / R0) ** a
    c = (dRc / R0) ** a

    dP_loss, dP_certain = ruin_deltas(hh, d)

    return (prelec(d.p_success, hh.pp.sigma) * g
            - hh.pp.lam * prelec(1.0 - d.p_success, hh.pp.sigma)
              * (l + hh.pp.kappa * dP_loss)
            - hh.pp.lam * (c + hh.pp.kappa * dP_certain))


def p_take_risk(hh: Household, d: Decision) -> float:
    """Logit probability of accepting the risky option over the status quo."""
    z = (prospect_value(hh, d) - hh.pp.theta) / hh.pp.tau
    return float(1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0))))


# ---------------------------------------------------------------------------
# 6. PAR PRIZE — direct analogue of the DLS par score
# ---------------------------------------------------------------------------

def par_prize(hh: Household, loss_mhi: float = 2.0, p_success: float = 0.5,
              loss_damage: Optional[Dict[str, float]] = None,
              hi: float = 5000.0) -> float:
    """
    DLS asks: given your resources, what total must you reach to be level?
    We ask: given your resources, what GAIN makes you exactly indifferent
    to staking this downside?

    Bisection on V(gain) = 0. Note that theta is deliberately EXCLUDED here.
    theta is behavioural stickiness in observed choice; as a fixed hurdle in
    resource-fraction units it would mechanically inflate the par prize for
    wealthy households, whose gain and loss branches are both small relative
    to R. Keeping theta in the logit but out of the indifference condition
    makes lambda_eff a clean measure of PREFERENCE, uncontaminated by inertia.

    Returns inf when no finite gain suffices — the analogue of an
    unreachable revised target.
    """
    def f(g):
        return prospect_value(
            hh, Decision(gain_mhi=g, loss_mhi=loss_mhi, p_success=p_success,
                         loss_damage=loss_damage or {}))

    if f(hi) < 0:
        return float("inf")
    lo, high = 0.0, hi
    # 44 halvings take the bracket from 5000 to ~3e-10 MHI: far below any
    # meaningful precision, and 2x cheaper than the 90 used while debugging.
    for _ in range(44):
        mid = 0.5 * (lo + high)
        if f(mid) < 0:
            lo = mid
        else:
            high = mid
        if high - lo < 1e-7:
            break
    return 0.5 * (lo + high)


def lambda_effective(hh: Household, stake_mhi: float = 2.0,
                     p_success: float = 0.5) -> float:
    """
    THE STUDY'S OPERATIONAL DEFINITION OF RISK AVERSION.

        lambda_eff = par_prize(stake) / stake

    Rupees of upside demanded per rupee of downside on a 50-50 gamble.
    Scale-free, comparable across cells, directly reportable as
    'group X is n% more risk-averse than group Y'.
    """
    pp = par_prize(hh, stake_mhi, p_success)
    return pp / stake_mhi if np.isfinite(pp) else float("inf")


# Default, validated parameter set -------------------------------------------
RP_DEFAULT = calibrate_scale(ResourceParams())
