"""Models of first-round choices in indefinitely repeated PD games.

Arrays in `d` have shape (subjects, matches); every function loops over
matches and is vectorized over subjects, so the same code fits one subject
or a whole population.

    c1[i, m]   own first-round choice (1 = cooperate)
    o1[i, m]   partner's first-round choice
    L[i, m]    realized length of match m (rounds)
    ET[i]      stated expected length, 1 / (1 - delta)
    g[i], l[i] normalized temptation gain and sucker loss

Treating a first-round C as Grim and a D as Always Defect, the value of
starting with C given belief p that the partner is a conditional cooperator
and a horizon of H rounds is

    dU(p, H) = p * (H - 1 - g) - (1 - p) * l

(Blonski et al., 2011; Mengel et al., 2022). The models differ in what is
carried from one match to the next: nothing about the future (planner),
payoff differences (value cache) or the horizon itself (prediction cache).
None of them depends on the subject's own past choices, so choice
probabilities are a function of o1 and L only, and the same function is used
for the likelihood and for simulation.
"""
from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def delta_U(p, H, g, l):
    """Expected payoff of Grim minus Always Defect, in normalized units."""
    return p * (H - 1.0 - g) - (1.0 - p) * l


def neg_log_lik(probs, y, mask=None, eps=1e-9):
    probs = np.clip(probs, eps, 1 - eps)
    ll = y * np.log(probs) + (1 - y) * np.log(1 - probs)
    return -np.sum(ll if mask is None else mask * ll)


# Planner: horizon from the instructions, only beliefs about partners are learned.
PLANNER = {"beta": (0.0, 10.0), "bias": (-10.0, 10.0), "kappa": (0.0, 1.0), "p0": (0.01, 0.99)}


def planner_probs(theta, d):
    """P(c1 = 1) = sigmoid(beta * dU(p, ET) + bias); p moves toward o1 at rate kappa."""
    beta, bias, kappa, p0 = theta
    N, M = d["o1"].shape
    out = np.empty((N, M))
    p = np.full(N, p0)
    for m in range(M):
        out[:, m] = sigmoid(beta * delta_U(p, d["ET"], d["g"], d["l"]) + bias)
        p = p + kappa * (d["o1"][:, m] - p)
    return out


# Value cache: Mengel, Orlandi & Weidenholzer (2022), eqs. 6-7, reparameterized.
VALUE_CACHE = {"a0": (-10.0, 10.0), "lam": (0.0, 5.0), "alpha": (0.0, 1.0)}


def value_cache_probs(theta, d):
    """P(c1 = 1) = sigmoid(a0 + lam * D). After each match D adds the payoff
    difference Grim - AD against the partner's strategy (read off its first
    move), with weight alpha on the realized length and 1 - alpha on ET."""
    a0, lam, alpha = theta
    N, M = d["o1"].shape
    out = np.empty((N, M))
    D = np.zeros(N)
    g, l = d["g"], d["l"]

    for m in range(M):
        out[:, m] = sigmoid(a0 + lam * D)
        partner_coop = d["o1"][:, m] == 1
        # ET is one number per subject, not per match
        delta_pi_L = np.where(partner_coop, d["L"][:, m] - 1 - g, -l)
        delta_pi_ET = np.where(partner_coop, d["ET"] - 1 - g, -l)
        D = D + alpha * delta_pi_L + (1 - alpha) * delta_pi_ET
    return out


# Prediction cache: the horizon itself is learned from every match.
PREDICTION_CACHE = {"beta": (0.0, 10.0), "bias": (-10.0, 10.0), "kappa": (0.0, 1.0),
                    "p0": (0.01, 0.99), "eta": (0.0, 1.0)}


def prediction_cache_probs(theta, d):
    """Planner with H in place of ET; H starts at ET and moves toward each
    match's realized length at rate eta. eta = 0 gives the planner back."""
    beta, bias, kappa, p0, eta = theta
    N, M = d["o1"].shape
    out = np.empty((N, M))
    p = np.full(N, p0)
    H = np.array(d["ET"], dtype=float)   # copy, so the data are never modified
    g, l = d["g"], d["l"]

    for m in range(M):
        dU = p * (H - 1 - g) - (1 - p) * l
        out[:, m] = sigmoid(beta * dU + bias)
        p = p + kappa * (d["o1"][:, m] - p)
        H = H + eta * (d["L"][:, m] - H)
    return out


MODELS = {
    "planner": (planner_probs, PLANNER),
    "value_cache": (value_cache_probs, VALUE_CACHE),
    "prediction_cache": (prediction_cache_probs, PREDICTION_CACHE),
}

# Ranges for drawing true parameters in recovery simulations. Narrower than the
# fitting bounds so that simulated choices are neither deterministic nor random.
RECOVERY_RANGES = {
    "planner": {"beta": (0.3, 3.0), "bias": (-1.5, 1.5), "kappa": (0.05, 0.5), "p0": (0.2, 0.8)},
    "value_cache": {"a0": (-1.5, 1.5), "lam": (0.02, 0.4), "alpha": (0.0, 1.0)},
    "prediction_cache": {"beta": (0.3, 3.0), "bias": (-1.5, 1.5), "kappa": (0.05, 0.5),
                         "p0": (0.2, 0.8), "eta": (0.02, 0.5)},
}
