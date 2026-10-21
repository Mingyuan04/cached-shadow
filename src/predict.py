"""Week 3: what each model predicts for the week-1 signature, and whether the
fitted models reproduce what the data show (posterior predictive check).

  python -m src.predict --theory                          # after the TODOs
  python -m src.predict --ppc --config data/df2011_config.json
  python -m src.predict --ppc --stratest data/stratEst_DF2011_late_subset.rda   # code test

The signature is the pair of slopes from src/signature.py: effect of the
previous match's length on next-match round-1 cooperation, after a partner who
cooperated first and after one who defected first.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .fitting import fit, implemented_models, simulate, synthetic_population
from .signature import _design, fe_ols

# Representative parameters for the theory figure (mid-range cooperation).
THEORY_PARAMS = {
    "planner": [1.5, 0.0, 0.3, 0.5],                 # beta, bias, kappa, p0
    "value_cache": [0.0, 0.15, 1.0],                 # a0, lam, alpha
    "prediction_cache": [1.5, 0.0, 0.3, 0.5, 0.3],   # beta, bias, kappa, p0, eta
}
COLORS = {"after_C": "tab:blue", "after_D": "tab:orange", "strict": "tab:green"}
LABELS = {"after_C": "previous partner cooperated first",
          "after_D": "previous partner defected first",
          "strict": "previous partner never cooperated"}


def arrays_to_matches(d: dict, c1: np.ndarray) -> pd.DataFrame:
    """Rebuild the match-level table that signature.py expects from (N, M) arrays."""
    i, m = np.nonzero(d["mask"])
    df = pd.DataFrame({"subject": i, "session": np.asarray(d["session"])[i], "match": m + 1,
                       "L": d["L"][i, m], "c1": c1[i, m], "o1": d["o1"][i, m],
                       "o_any": d.get("o_any", d["o1"])[i, m],
                       "ET": np.asarray(d["ET"])[i]}).sort_values(["subject", "match"])
    g = df.groupby("subject")
    for col in ["L", "c1", "o1", "o_any"]:
        df[f"{col}_prev"] = g[col].shift(1)
    df["consecutive"] = df["L_prev"].notna()
    df["Lrel_prev"] = df["L_prev"] - df["ET"]
    df["match_index"] = df["match"]
    return df


def slopes(df: pd.DataFrame) -> dict:
    x = _design(df)
    out = {}
    for key, sel in [("after_C", x["o1_prev"] == 1), ("after_D", x["o1_prev"] == 0),
                     ("strict", x["o_any_prev"] == 0)]:
        b, _ = fe_ols(x[sel])
        out[key] = float(b["Lrel_prev"])
    return out


def theory(n_sims: int = 20, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for name in implemented_models():
        for _ in range(n_sims):
            d = simulate(name, THEORY_PARAMS[name], synthetic_population(rng), rng)
            rows.append({"model": name, **slopes(arrays_to_matches(d, d["c1"]))})
    return pd.DataFrame(rows)


def ppc(d: dict, n_sims: int = 50, seed: int = 0):
    """Fit each implemented model to the real data, simulate it on the real
    environment (same partners' first moves, same match lengths), and collect
    the signature slopes it produces."""
    rng = np.random.default_rng(seed)
    observed = slopes(arrays_to_matches(d, d["c1"]))
    rows, fitted = [], {}
    for name in implemented_models():
        params, _ = fit(name, d)
        fitted[name] = params
        for _ in range(n_sims):
            sim = simulate(name, list(params.values()), d, rng)
            rows.append({"model": name, **slopes(arrays_to_matches(sim, sim["c1"]))})
    return observed, pd.DataFrame(rows), fitted


def plot(sims: pd.DataFrame, path: str, observed: dict | None = None, title: str = ""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(dict.fromkeys(sims["model"]))
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    keys = ["after_C", "strict"]           # the two that separate the accounts
    for k, key in enumerate(keys):
        for j, name in enumerate(models):
            v = sims.loc[sims["model"] == name, key] * 100      # percentage points
            lo, mid, hi = np.percentile(v, [2.5, 50, 97.5])
            xpos = j + (k - 0.5) * 0.25
            ax.errorbar(xpos, mid, yerr=[[mid - lo], [hi - mid]], fmt="o", color=COLORS[key],
                        capsize=4, label=LABELS[key] if j == 0 else None)
        if observed is not None:
            ax.axhline(observed[key] * 100, color=COLORS[key], ls="--", lw=1,
                       label=f"data: {LABELS[key]}")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_xticks(range(len(models)), [m.replace("_", " ") for m in models])
    ax.set_ylabel("effect of one extra round in the previous\nmatch on next round-1 cooperation (pp)")
    ax.set_title(title, fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", action="store_true")
    ap.add_argument("--ppc", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--stratest")
    ap.add_argument("--sims", type=int, default=30)
    a = ap.parse_args()
    if a.theory:
        sims = theory(a.sims)
        print(sims.groupby("model").agg(["mean", "std"]).mul(100).round(2))
        plot(sims, "figures/theory_predictions.png",
             title="Predicted signature: median and 95% of simulated experiments")
        print("figure -> figures/theory_predictions.png")
    if a.ppc:
        from .prep import load_from_config, load_stratest_df2011, match_table, to_arrays
        df = load_stratest_df2011(a.stratest) if a.stratest else load_from_config(a.config)
        d = to_arrays(match_table(df))
        observed, sims, fitted = ppc(d, a.sims)
        print("observed slopes (pp):", {k: round(v * 100, 2) for k, v in observed.items()})
        print(sims.groupby("model").agg(["mean", "std"]).mul(100).round(2))
        plot(sims, "figures/posterior_predictive.png", observed,
             title="Fitted models vs data (dashed lines)")
        print("figure -> figures/posterior_predictive.png")


if __name__ == "__main__":
    main()
