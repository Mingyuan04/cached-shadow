"""Three accounts of the 'shadow of the future', for round-1 choices.

All functions are vectorized over subjects: arrays in `d` have shape
(N subjects, M matches), and the loop runs over matches. With N = 1 the same
code fits a single subject.

Notation
  c1[i, m]  own round-1 choice      (1 = C: start with a conditionally cooperative strategy)
  o1[i, m]  partner's round-1 choice (1 = C ~ Grim/TFT, 0 = D ~ Always Defect)
  L[i, m]   realized length of match m
  ET[i]     instructed expected length 1 / (1 - delta)
  g[i], l[i] normalized temptation and sucker loss (mutual C = 1 per round, mutual D = 0)

Strategy-level payoffs, Grim vs Always Defect (Blonski et al. 2011; Dal Bo &
Frechette 2011; Mengel et al. 2022, Fig. 1):
  Grim vs Grim : H      Grim vs AD : -l      AD vs Grim : 1 + g      AD vs AD : 0
so the value of starting with C rather than D, given a belief p that the
partner is a conditional cooperator and a horizon of H rounds, is

      dU(p, H) = p * (H - 1 - g) - (1 - p) * l

The models differ only in WHAT is carried from one match to the next:
  planner           nothing about the future: H = ET, the instructed horizon
  value cache       payoff differences (Mengel, Orlandi & Weidenholzer 2022)
  prediction cache  H itself: how long relationships last (SR-like)

None of them depends on the subject's own past choices, so the choice
probabilities are a function of the environment (o1, L) only, and the same
function serves for simulation and for the likelihood.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def delta_U(p, H, g, l):
    """Expected payoff of Grim minus Always Defect (normalized units)."""
    return p * (H - 1.0 - g) - (1.0 - p) * l


def neg_log_lik(probs, y, mask=None, eps=1e-9):
    probs = np.clip(probs, eps, 1 - eps)
    ll = y * np.log(probs) + (1 - y) * np.log(1 - probs)
    return -np.sum(ll if mask is None else mask * ll)


# ---------------------------------------------------------------------------
# 1) Planner  -- WORKED EXAMPLE (read this first, then write the other two)
# ---------------------------------------------------------------------------
PLANNER = {"beta": (0.0, 10.0), "bias": (-10.0, 10.0), "kappa": (0.0, 1.0), "p0": (0.01, 0.99)}


def planner_probs(theta, d):
    """P(c1 = 1) for every subject and match.

    P(c1_m = 1) = sigmoid(beta * dU(p_m, ET) + bias)
    p_{m+1}     = p_m + kappa * (o1_m - p_m)          p_1 = p0
    The horizon is the instructed one, so realized lengths never matter.
    """
    beta, bias, kappa, p0 = theta
    N, M = d["o1"].shape
    out = np.empty((N, M))
    p = np.full(N, p0)
    for m in range(M):
        out[:, m] = sigmoid(beta * delta_U(p, d["ET"], d["g"], d["l"]) + bias)
        p = p + kappa * (d["o1"][:, m] - p)
    return out


# ---------------------------------------------------------------------------
# 2) Value cache  -- after Mengel, Orlandi & Weidenholzer (2022), eqs. (6)-(7)
# ---------------------------------------------------------------------------
VALUE_CACHE = {"a0": (-10.0, 10.0), "lam": (0.0, 5.0), "alpha": (0.0, 1.0)}


def value_cache_probs(theta, d):
    """P(c1 = 1) for every subject and match.

    P(c1_m = 1) = sigmoid(a0 + lam * D_m)                       D_1 = 0
    After match m, read the partner's strategy off its first move and add the
    payoff difference Grim - AD that you WOULD have earned against it:
        dpi(H) = H - 1 - g    if o1_m == 1
               = -l           if o1_m == 0      <- no H: length cannot matter
        D_{m+1} = D_m + alpha * dpi(L_m) + (1 - alpha) * dpi(ET)
    alpha = weight on the realized rather than the expected length
    (Mengel et al. estimate about 0.43, pooled). a0 = lam * (psi_GT - psi_AD)
    in their notation; this reparameterization fits more stably.
    Hint: np.where(d["o1"][:, m] == 1, ..., ...) handles both partner types.
    """
    raise NotImplementedError("TODO(you): about 8 lines; mirror planner_probs")


# ---------------------------------------------------------------------------
# 3) Prediction cache  -- SR-like: the horizon itself is learned
# ---------------------------------------------------------------------------
PREDICTION_CACHE = {"beta": (0.0, 10.0), "bias": (-10.0, 10.0), "kappa": (0.0, 1.0),
                    "p0": (0.01, 0.99), "eta": (0.0, 1.0)}


def prediction_cache_probs(theta, d):
    """P(c1 = 1) for every subject and match.

    P(c1_m = 1) = sigmoid(beta * dU(p_m, H_m) + bias)
    after EVERY match m, whatever happened in it:
        p_{m+1} = p_m + kappa * (o1_m - p_m)                     p_1 = p0
        H_{m+1} = H_m + eta * (L_m - H_m)                        H_1 = ET
    H is the successor-representation entry for "being in a relationship":
    the expected number of rounds still to come; this is its Monte Carlo update.
    eta = 0 must reproduce planner_probs exactly (tests/test_models.py checks).
    Variant worth adding later: a free H_1. Dal Bo & Frechette (2011, Table 3)
    find delta has no effect in the first match, which a free prior captures.
    Preregistered variant: a decreasing learning rate, eta_m = 1 / (n0 + m),
    makes H a running average of experienced lengths with prior weight n0 on
    ET. Mengel et al. (2022) find early matches matter at least as much as
    recent ones, which a constant eta cannot produce. Add it as a fourth model
    (parameter n0 instead of eta) once the three above pass their checks.
    """
    raise NotImplementedError("TODO(you): about 10 lines; start from planner_probs")


# ---------------------------------------------------------------------------
# 4) Extension (after week 3): model-free reinforcement of the chosen first move
# ---------------------------------------------------------------------------
def model_free_probs(theta, d):
    """Fudenberg & Karreskog Rehbinder (2024, AEJ: Micro), 'IRL-SG', eqs. (1)-(2):
        P(c1_m = 1) = sigmoid(a0 + e_m)                       e_1 = 0
        e_{m+1}     = e_m + lam * (2 * c1_m - 1) * pay_m
    pay_m = your total normalized payoff in match m (array `pay`).
    Unlike the three models above it depends on your OWN choices, so
    simulating it needs within-match play too (they use a fixed semi-grim rule).
    """
    raise NotImplementedError("extension")


MODELS = {
    "planner": (planner_probs, PLANNER),
    "value_cache": (value_cache_probs, VALUE_CACHE),
    "prediction_cache": (prediction_cache_probs, PREDICTION_CACHE),
}

# Ranges for drawing "true" parameters in recovery studies: narrower than the
# fitting bounds, so simulated agents are neither deterministic nor random.
RECOVERY_RANGES = {
    "planner": {"beta": (0.3, 3.0), "bias": (-1.5, 1.5), "kappa": (0.05, 0.5), "p0": (0.2, 0.8)},
    "value_cache": {"a0": (-1.5, 1.5), "lam": (0.02, 0.4), "alpha": (0.0, 1.0)},
    "prediction_cache": {"beta": (0.3, 3.0), "bias": (-1.5, 1.5), "kappa": (0.05, 0.5),
                         "p0": (0.2, 0.8), "eta": (0.02, 0.5)},
}
