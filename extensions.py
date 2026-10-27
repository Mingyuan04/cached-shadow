"""Matched comparison: the planner plus one cached quantity.

The value cache in models.py starts every treatment from the same propensity,
while the planner and the prediction cache start from dU(p, ET). The models
here all start from the planner, so they differ only in what is cached.

    value_cache_plus      + lam * S, where S adds L - ET after each match with a
                          partner who cooperated first (the realized-length part
                          of Mengel et al.'s update)
    prediction_cache_avg  H is a running average of match lengths,
                          H <- H + (L - H) / (n0 + m)   (preregistered variant)
    both                  cached horizon (eta) and cached value (lam) together

    python -m src.extensions --recover
    python -m src.extensions --model-recovery
    python -m src.extensions --fit --config data/df2011_config.json
"""
from __future__ import annotations

import argparse

import numpy as np

from .models import MODELS, RECOVERY_RANGES, delta_U, sigmoid


def value_cache_plus_probs(theta, d):
    beta, bias, kappa, p0, lam = theta
    N, M = d["o1"].shape
    out, p, S = np.empty((N, M)), np.full(N, p0), np.zeros(N)
    for m in range(M):
        out[:, m] = sigmoid(beta * delta_U(p, d["ET"], d["g"], d["l"]) + bias + lam * S)
        S = S + (d["o1"][:, m] == 1) * (d["L"][:, m] - d["ET"])
        p = p + kappa * (d["o1"][:, m] - p)
    return out


def prediction_cache_avg_probs(theta, d):
    beta, bias, kappa, p0, n0 = theta
    N, M = d["o1"].shape
    out, p, H = np.empty((N, M)), np.full(N, p0), np.array(d["ET"], dtype=float)
    for m in range(M):
        out[:, m] = sigmoid(beta * delta_U(p, H, d["g"], d["l"]) + bias)
        p = p + kappa * (d["o1"][:, m] - p)
        H = H + (d["L"][:, m] - H) / (n0 + m + 1)
    return out


def both_probs(theta, d):
    beta, bias, kappa, p0, eta, lam = theta
    N, M = d["o1"].shape
    out, p = np.empty((N, M)), np.full(N, p0)
    H, S = np.array(d["ET"], dtype=float), np.zeros(N)
    for m in range(M):
        out[:, m] = sigmoid(beta * delta_U(p, H, d["g"], d["l"]) + bias + lam * S)
        S = S + (d["o1"][:, m] == 1) * (d["L"][:, m] - d["ET"])
        p = p + kappa * (d["o1"][:, m] - p)
        H = H + eta * (d["L"][:, m] - H)
    return out


BASE = {"beta": (0.0, 10.0), "bias": (-10.0, 10.0), "kappa": (0.0, 1.0), "p0": (0.01, 0.99)}
BASE_R = {"beta": (0.3, 3.0), "bias": (-1.5, 1.5), "kappa": (0.05, 0.5), "p0": (0.2, 0.8)}
EXTENSIONS = {
    "value_cache_plus": (value_cache_plus_probs, {**BASE, "lam": (0.0, 2.0)}, {**BASE_R, "lam": (0.005, 0.08)}),
    "prediction_cache_avg": (prediction_cache_avg_probs, {**BASE, "n0": (0.5, 200.0)}, {**BASE_R, "n0": (1.0, 30.0)}),
    "both": (both_probs, {**BASE, "eta": (0.0, 1.0), "lam": (0.0, 2.0)},
             {**BASE_R, "eta": (0.02, 0.5), "lam": (0.005, 0.08)}),
}
for _name, (_fn, _bounds, _ranges) in EXTENSIONS.items():
    MODELS[_name] = (_fn, _bounds)
    RECOVERY_RANGES[_name] = _ranges

FAIR_SET = ["planner", "value_cache_plus", "prediction_cache", "both"]


def main():
    from .fitting import compare, cv_by_session, model_recovery, parameter_recovery, plot_recovery
    ap = argparse.ArgumentParser()
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--model-recovery", action="store_true")
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--stratest")
    ap.add_argument("--datasets", type=int, default=20)
    a = ap.parse_args()
    if a.recover:
        for name in EXTENSIONS:
            df, corr = parameter_recovery(name, a.datasets)
            print(name, {k: round(v, 2) for k, v in corr.items()})
            plot_recovery(df, corr, name, f"figures/recovery_{name}.png")
    if a.model_recovery:
        print(model_recovery(FAIR_SET, a.datasets))
    if a.fit:
        from .prep import load_from_config, load_stratest_df2011, match_table, to_arrays
        df = load_stratest_df2011(a.stratest) if a.stratest else load_from_config(a.config)
        d = to_arrays(match_table(df))
        names = list(dict.fromkeys(["planner", "value_cache", "prediction_cache", *EXTENSIONS]))
        print(compare(names, d).round(3).to_string(index=False))
        print(cv_by_session(names, d).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
