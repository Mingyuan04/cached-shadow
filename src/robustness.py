"""Checks added after the first draft was reviewed.

--subsets runs the two preregistered robustness subsets in Table 3: the
delta = 3/4 sessions only, and the first half of each session.

--session-end reruns the signature regression three more ways. A session
ended with the first match to finish after 50 minutes, so its last match
tends to be long and has no successor. Dropping it selects on the predictor,
which does not bias the slope by itself, but matches near the end of a
session may differ, and a long previous match also means more rounds played.
So: control for log matches remaining, control for log rounds already
played, and drop each session's last three matches.

--cv-warm refits everything with nested starting values: a model that
contains another (e.g. `both` contains the prediction cache at lam = 0) is
also started from that model's optimum, so it can never fit its training data
worse. Every fold is checked, and per-fold eta, lam and n0 are printed.

    python -m src.robustness --subsets --config data/df2011_config.json
    python -m src.robustness --session-end --config data/df2011_config.json
    python -m src.robustness --cv-warm --config data/df2011_config.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import extensions  # noqa: F401  (registers the matched-comparison models)
from .fitting import _rows
from .models import MODELS, neg_log_lik
from .prep import load_from_config, load_stratest_df2011, match_table, to_arrays
from .signature import REGRESSORS, _design, fe_ols, run


def preregistered_subsets(m: pd.DataFrame, n_boot: int = 999) -> pd.DataFrame:
    """Table 3, columns 2 and 3: delta = 3/4 sessions, first half of each session."""
    first_half = m["match_index"] <= m.groupby("subject")["match_index"].transform("max") / 2
    parts = []
    for label, sub in [("delta = 3/4 sessions", m[m["delta"] == 0.75]),
                       ("first half of each session", m[first_half])]:
        r = run(sub, n_boot=n_boot)
        r.insert(0, "sample", label)
        parts.append(r)
    return pd.concat(parts, ignore_index=True)


# Session end and time on task

def add_session_position(m: pd.DataFrame) -> pd.DataFrame:
    m = m.sort_values(["subject", "match"]).copy()
    g = m.groupby("subject")
    m["remaining"] = g["match_index"].transform("max") - m["match_index"]   # 0 = last match
    m["is_last"] = (m["remaining"] == 0).astype(int)
    m["rounds_before"] = g["L"].cumsum() - m["L"]
    m["log_remaining"] = np.log1p(m["remaining"])
    m["log_rounds_before"] = np.log1p(m["rounds_before"])
    return m


def last_match_check(m: pd.DataFrame) -> pd.DataFrame:
    """Realized minus expected length, last match of each session versus the rest."""
    per = m.drop_duplicates(["session", "match_index"]).copy()
    per["Lrel"] = per["L"] - per["ET"]
    out = per.groupby("is_last")["Lrel"].agg(["mean", "count"])
    out.index = out.index.map({0: "other matches", 1: "last match of the session"})
    return out


def _boot(x: pd.DataFrame, xs, n_boot: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    sessions = x["session"].unique()
    by = {s: x[x["session"] == s] for s in sessions}
    draws = []
    for _ in range(n_boot):
        parts = []
        for k, s in enumerate(rng.choice(sessions, size=len(sessions), replace=True)):
            p = by[s].copy()
            p["session"] = k
            p["subject"] = p["subject"].astype(str) + f"_{k}"
            parts.append(p)
        b, _ = fe_ols(pd.concat(parts), xs=xs)
        draws.append(b["Lrel_prev"])
    return np.percentile(draws, [2.5, 97.5])


def signature_variants(m: pd.DataFrame, n_boot: int = 999) -> pd.DataFrame:
    x = _design(add_session_position(m))
    specs = [
        ("main analysis", x, REGRESSORS),
        ("+ log matches remaining", x, REGRESSORS + ["log_remaining"]),
        ("+ log rounds played before", x, REGRESSORS + ["log_rounds_before"]),
        ("without the last 3 matches", x[x["remaining"] >= 3], REGRESSORS),
    ]
    rows = []
    for label, data, xs in specs:
        for subset, sel in [("cooperated first", data["o1_prev"] == 1),
                            ("never cooperated", data["o_any_prev"] == 0)]:
            sub = data[sel]
            b, se = fe_ols(sub, xs=xs)
            lo, hi = _boot(sub, xs, n_boot) if n_boot else (np.nan, np.nan)
            rows.append({"specification": label, "previous partner": subset, "n": len(sub),
                         "slope (pp)": 100 * b["Lrel_prev"], "SE (pp)": 100 * se["Lrel_prev"],
                         "2.5%": 100 * lo, "97.5%": 100 * hi})
    return pd.DataFrame(rows)


# Fits and cross-validation with nested starting values

ORDER = ["planner", "value_cache", "prediction_cache", "value_cache_plus",
         "prediction_cache_avg", "both"]
BASE = ["beta", "bias", "kappa", "p0"]
NESTS = {   # model -> [(model it contains, start built from that model's optimum)]
    "prediction_cache": [("planner", lambda p: [p[k] for k in BASE] + [0.0])],
    "value_cache_plus": [("planner", lambda p: [p[k] for k in BASE] + [0.0])],
    "prediction_cache_avg": [("planner", lambda p: [p[k] for k in BASE] + [200.0])],
    "both": [("prediction_cache", lambda p: [p[k] for k in BASE] + [p["eta"], 0.0]),
             ("value_cache_plus", lambda p: [p[k] for k in BASE] + [0.0, p["lam"]])],
}
CHECKS = [("prediction_cache", "planner"), ("value_cache_plus", "planner"),
          ("both", "prediction_cache"), ("both", "value_cache_plus")]


def fit_from(model: str, d: dict, starts, n_random: int, rng):
    fn, bounds = MODELS[model]
    lo = np.array([b[0] for b in bounds.values()])
    hi = np.array([b[1] for b in bounds.values()])
    f = lambda th: neg_log_lik(fn(th, d), d["c1"], d["mask"])
    cands = [np.clip(np.asarray(s, float), lo, hi) for s in starts]
    cands += [rng.uniform(lo, hi) for _ in range(n_random)]
    best = None
    for x0 in cands:
        res = minimize(f, x0, method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if best is None or res.fun < best.fun:
            best = res
    return dict(zip(bounds, best.x)), float(best.fun)


def fit_nested(d: dict, n_random: int, rng, models=ORDER) -> dict:
    out = {}
    for name in models:
        starts = [make(out[sub][0]) for sub, make in NESTS.get(name, []) if sub in out]
        out[name] = fit_from(name, d, starts, n_random, rng)
    return out


def cv_warm(d: dict, n_random: int = 8, seed: int = 0, models=ORDER):
    rng = np.random.default_rng(seed)
    rows = []
    for s in np.unique(d["session"]):
        train, test = _rows(d, d["session"] != s), _rows(d, d["session"] == s)
        for name, (params, nll_train) in fit_nested(train, n_random, rng, models).items():
            held = neg_log_lik(MODELS[name][0](list(params.values()), test), test["c1"], test["mask"])
            rows.append({"fold": s, "model": name, "train_nll": nll_train, "held_out_nll": held,
                         **{k: params[k] for k in ("eta", "lam", "n0") if k in params}})
    per_fold = pd.DataFrame(rows)
    per_fold = per_fold.reindex(columns=["fold", "model", "train_nll", "held_out_nll"]
                                + [c for c in ("eta", "lam", "n0") if c in per_fold])
    train = per_fold.pivot(index="fold", columns="model", values="train_nll")
    checks = {f"{a} fits training data at least as well as {b} in every fold":
              bool((train[a] <= train[b] + 1e-6).all())
              for a, b in CHECKS if a in train and b in train}
    summary = (per_fold.groupby("model")["held_out_nll"].sum()
               .reindex([m for m in models if m in set(per_fold["model"])]).to_frame())
    summary["per_choice"] = summary["held_out_nll"] / d["mask"].sum()
    if "planner" in summary.index:
        summary["vs_planner"] = summary["held_out_nll"] - summary.loc["planner", "held_out_nll"]
    return per_fold, summary, checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subsets", action="store_true")
    ap.add_argument("--session-end", action="store_true")
    ap.add_argument("--cv-warm", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--stratest")
    ap.add_argument("--n_boot", type=int, default=999)
    ap.add_argument("--n_random", type=int, default=8)
    a = ap.parse_args()
    df = load_stratest_df2011(a.stratest) if a.stratest else load_from_config(a.config)
    m = match_table(df)
    pd.set_option("display.width", 200)
    if a.subsets:
        print(preregistered_subsets(m, a.n_boot).round(4).to_string(index=False))
    if a.session_end:
        print(last_match_check(add_session_position(m)).round(3).to_string())
        print(signature_variants(m, a.n_boot).round(2).to_string(index=False))
    if a.cv_warm:
        d = to_arrays(m)
        n = d["mask"].sum()
        print("full-data fits (nested warm starts):")
        rng = np.random.default_rng(0)
        for name, (params, nll) in fit_nested(d, a.n_random, rng).items():
            k = len(params)
            print(f"  {name:22s} NLL {nll:10.3f}  BIC {2 * nll + k * np.log(n):10.3f}  ",
                  {p: round(float(v), 3) for p, v in params.items()})
        per_fold, summary, checks = cv_warm(d, a.n_random)
        print(per_fold.round(3).to_string(index=False))
        print(summary.round(4).to_string())
        for text, ok in checks.items():
            print(("OK   " if ok else "FAIL ") + text)


if __name__ == "__main__":
    main()
