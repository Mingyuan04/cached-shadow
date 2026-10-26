"""Fitting, cross-validation and recovery.

Parameters are estimated for the whole population at once. With 23-77
first-round choices per person the models are only weakly identified for a
single subject (planner recovery r = 0.1-0.45 in simulations of this design),
whereas pooled fits recover parameters almost perfectly; Mengel et al. (2022)
and Fudenberg & Karreskog Rehbinder (2024) fit at the population level too.

    python -m src.fitting --recover planner
    python -m src.fitting --model-recovery
    python -m src.fitting --cv --config data/df2011_config.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .models import MODELS, RECOVERY_RANGES, neg_log_lik
from .prep import normalized_params


def _bounds(model):
    b = MODELS[model][1]
    return np.array([v[0] for v in b.values()]), np.array([v[1] for v in b.values()])


def _rows(d: dict, idx) -> dict:
    """Subset of subjects (rows) from an arrays dict."""
    return {k: (v[idx] if isinstance(v, np.ndarray) and len(v) == len(d["mask"]) else v)
            for k, v in d.items()}


def fit(model: str, d: dict, n_starts: int = 8, rng=None):
    """Pooled maximum likelihood: one parameter vector for all subjects in d."""
    rng = rng or np.random.default_rng(0)
    probs_fn, bounds = MODELS[model]
    lo, hi = _bounds(model)
    f = lambda th: neg_log_lik(probs_fn(th, d), d["c1"], d["mask"])
    best = None
    for _ in range(n_starts):
        res = minimize(f, rng.uniform(lo, hi), method="L-BFGS-B", bounds=list(zip(lo, hi)))
        if best is None or res.fun < best.fun:
            best = res
    return dict(zip(bounds, best.x)), float(best.fun)


def bic(nll: float, k: int, n: float) -> float:
    return 2 * nll + k * np.log(n)


def compare(models, d: dict, n_starts: int = 8) -> pd.DataFrame:
    n = d["mask"].sum()
    rows = []
    for name in models:
        params, nll = fit(name, d, n_starts)
        rows.append({"model": name, "nll": nll, "k": len(params),
                     "bic": bic(nll, len(params), n), **params})
    return pd.DataFrame(rows).sort_values("bic")


def cv_by_session(models, d: dict, n_starts: int = 5) -> pd.DataFrame:
    """Leave-one-session-out: fit on the other sessions, score the held-out one.
    Lower held-out NLL = better out-of-sample prediction. Sessions are the
    natural unit because they share their sequence of match lengths."""
    sessions = np.unique(d["session"])
    rows = []
    for name in models:
        probs_fn = MODELS[name][0]
        total = 0.0
        for s in sessions:
            train, test = _rows(d, d["session"] != s), _rows(d, d["session"] == s)
            params, _ = fit(name, train, n_starts)
            total += neg_log_lik(probs_fn(list(params.values()), test), test["c1"], test["mask"])
        rows.append({"model": name, "held_out_nll": total,
                     "per_choice": total / d["mask"].sum()})
    return pd.DataFrame(rows).sort_values("held_out_nll")


def synthetic_population(rng, subjects_per_session=14, matches=45) -> dict:
    """A DF2011-like design: 6 treatments x 3 sessions; match lengths drawn once
    per session (shared by everyone in it); partners who start with C at a rate
    that drifts across the session, as in real sessions."""
    rows = []
    sess = 0
    for delta in (0.5, 0.75):
        for R in (32, 40, 48):
            g, l = normalized_params(R, 50, 12, 25)
            for _ in range(3):
                L = rng.geometric(1 - delta, size=matches).astype(float)
                q = np.linspace(*rng.uniform(0.2, 0.8, 2), matches)
                for _ in range(subjects_per_session):
                    rows.append((L, (rng.random(matches) < q).astype(float),
                                 1 / (1 - delta), float(g), float(l), sess))
                sess += 1
    o1 = np.array([r[1] for r in rows])
    return {"L": np.array([r[0] for r in rows]), "o1": o1,
            "o_any": o1.copy(),   # synthetic partners are pure Grim or pure AD
            "ET": np.array([r[2] for r in rows]), "g": np.array([r[3] for r in rows]),
            "l": np.array([r[4] for r in rows]), "session": np.array([r[5] for r in rows]),
            "mask": np.ones((len(rows), matches)), "pay": np.zeros((len(rows), matches))}


def draw_params(model: str, rng):
    return np.array([rng.uniform(*r) for r in RECOVERY_RANGES[model].values()])


def simulate(model: str, theta, env: dict, rng) -> dict:
    p = MODELS[model][0](theta, env)
    return dict(env, c1=(rng.random(p.shape) < p).astype(float))


def parameter_recovery(model: str, n_datasets: int = 20, seed: int = 1, env_fn=synthetic_population):
    rng = np.random.default_rng(seed)
    names = list(MODELS[model][1])
    rows = []
    for _ in range(n_datasets):
        true = draw_params(model, rng)
        d = simulate(model, true, env_fn(rng), rng)
        est, _ = fit(model, d, n_starts=5, rng=rng)
        rows.append({**{f"true_{n}": t for n, t in zip(names, true)},
                     **{f"fit_{n}": est[n] for n in names}})
    df = pd.DataFrame(rows)
    corr = {n: float(np.corrcoef(df[f"true_{n}"], df[f"fit_{n}"])[0, 1]) for n in names}
    return df, corr


def model_recovery(models, n_datasets: int = 10, seed: int = 2, env_fn=synthetic_population):
    """Rows: generating model. Columns: model with the lowest BIC.
    A clean diagonal means this design can tell the models apart."""
    rng = np.random.default_rng(seed)
    conf = pd.DataFrame(0, index=list(models), columns=list(models))
    for gen in models:
        for _ in range(n_datasets):
            d = simulate(gen, draw_params(gen, rng), env_fn(rng), rng)
            n = d["mask"].sum()
            scores = {m: bic(fit(m, d, 4, rng)[1], len(MODELS[m][1]), n) for m in models}
            conf.loc[gen, min(scores, key=scores.get)] += 1
    return conf


def plot_recovery(df: pd.DataFrame, corr: dict, model: str, path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(corr)
    fig, axes = plt.subplots(1, len(names), figsize=(2.8 * len(names), 2.8))
    for ax, n in zip(np.atleast_1d(axes), names):
        ax.scatter(df[f"true_{n}"], df[f"fit_{n}"], s=12, alpha=0.8)
        lo = min(df[f"true_{n}"].min(), df[f"fit_{n}"].min())
        hi = max(df[f"true_{n}"].max(), df[f"fit_{n}"].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        ax.set_title(f"{n}: r = {corr[n]:.2f}", fontsize=9)
        ax.set_xlabel("true", fontsize=8); ax.set_ylabel("recovered", fontsize=8)
    fig.suptitle(f"Population-level parameter recovery: {model}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recover", choices=list(MODELS))
    ap.add_argument("--model-recovery", action="store_true")
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--stratest")
    ap.add_argument("--config")
    ap.add_argument("--datasets", type=int, default=20)
    a = ap.parse_args()
    if a.recover:
        df, corr = parameter_recovery(a.recover, a.datasets)
        print({k: round(v, 2) for k, v in corr.items()})
        plot_recovery(df, corr, a.recover, f"figures/recovery_{a.recover}.png")
        print(f"figure -> figures/recovery_{a.recover}.png")
    if a.model_recovery:
        print(model_recovery(list(MODELS), a.datasets))
    if a.cv:
        from .prep import load_from_config, load_stratest_df2011, match_table, to_arrays
        df = load_stratest_df2011(a.stratest) if a.stratest else load_from_config(a.config)
        d = to_arrays(match_table(df))
        names = list(MODELS)
        print(compare(names, d).round(3).to_string(index=False))
        print(cv_by_session(names, d).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
