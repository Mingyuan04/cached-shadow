"""Predicted and fitted signatures.

--theory simulates each model in experiments shaped like DF2011 and runs the
signature regression on the simulated choices (Figure 1). --ppc fits each
model to the data, simulates it on the observed environment (same partners'
first moves, same match lengths) and compares its slopes with the observed
ones (Figure 4).

    python -m src.predict --theory
    python -m src.predict --ppc --ext --config data/df2011_config.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .fitting import fit, simulate, synthetic_population
from .models import MODELS
from .signature import _design, fe_ols

# Representative parameters for the theory figure (mid-range cooperation).
THEORY_PARAMS = {
    "planner": [1.5, 0.0, 0.3, 0.5],                 # beta, bias, kappa, p0
    "value_cache": [0.0, 0.15, 1.0],                 # a0, lam, alpha
    "prediction_cache": [1.5, 0.0, 0.3, 0.5, 0.3],   # beta, bias, kappa, p0, eta
}


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
    for name, theta in THEORY_PARAMS.items():
        for _ in range(n_sims):
            d = simulate(name, theta, synthetic_population(rng), rng)
            rows.append({"model": name, **slopes(arrays_to_matches(d, d["c1"]))})
    return pd.DataFrame(rows)


def ppc(d: dict, n_sims: int = 50, seed: int = 0):
    """Fit each implemented model to the real data, simulate it on the real
    environment (same partners' first moves, same match lengths), and collect
    the signature slopes it produces."""
    rng = np.random.default_rng(seed)
    observed = slopes(arrays_to_matches(d, d["c1"]))
    rows, fitted = [], {}
    for name in MODELS:
        params, _ = fit(name, d)
        fitted[name] = params
        for _ in range(n_sims):
            sim = simulate(name, list(params.values()), d, rng)
            rows.append({"model": name, **slopes(arrays_to_matches(sim, sim["c1"]))})
    return observed, pd.DataFrame(rows), fitted


def summarize(sims: pd.DataFrame) -> pd.DataFrame:
    """Median and 95% interval of each simulated slope, in percentage points."""
    rows = []
    for name, g in sims.groupby("model", sort=False):
        for key in ["after_C", "after_D", "strict"]:
            lo, mid, hi = np.percentile(g[key] * 100, [2.5, 50, 97.5])
            rows.append({"model": name, "slope": key, "median": round(mid, 2),
                         "2.5%": round(lo, 2), "97.5%": round(hi, 2)})
    return pd.DataFrame(rows).set_index(["model", "slope"])


LABEL = {"planner": "Planner", "value_cache": "Value cache", "value_cache_plus": "Planner + cached value",
         "prediction_cache": "Prediction cache", "prediction_cache_avg": "Prediction cache,\ndecreasing rate",
         "both": "Both cached terms"}
PANELS = [("after_C", "Previous partner\ncooperated in round 1"),
          ("after_D", "Previous partner\ndefected in round 1"),
          ("strict", "Previous partner\nnever cooperated (decisive)")]


def plot(sims: pd.DataFrame, path: str, observed: dict | None = None, title: str = ""):
    """One panel per kind of previous partner, one row per model: median (dot)
    and 95% interval (line) of the simulated slope. With observed data, the
    dashed line is the data and intervals that miss it are drawn light."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["planner", "value_cache", "value_cache_plus", "prediction_cache", "prediction_cache_avg", "both"]
    present = list(dict.fromkeys(sims["model"]))
    models = [m for m in order if m in present] + [m for m in present if m not in order]
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 0.45 * len(models) + 1.9), sharey=True)
    ys = list(range(len(models)))[::-1]
    # one x-range for all panels, wide enough for every interval and data line
    by_model = [sims.loc[sims["model"] == m] for m in models]
    lows = [np.percentile(g[k] * 100, 2.5) for g in by_model for k, _ in PANELS] + [0.0]
    highs = [np.percentile(g[k] * 100, 97.5) for g in by_model for k, _ in PANELS]
    if observed is not None:
        lows += [observed[k] * 100 for k, _ in PANELS]
        highs += [observed[k] * 100 for k, _ in PANELS]
    span = max(highs) - min(lows)
    xlim = (min(lows) - 0.08 * span, max(highs) + 0.12 * span)
    for ax, (key, panel_title) in zip(axes, PANELS):
        ax.set_xlim(*xlim)
        ax.axvline(0, color="0.8", lw=0.8)
        obs = None if observed is None else observed[key] * 100
        for y, name in zip(ys, models):
            lo, mid, hi = np.percentile(sims.loc[sims["model"] == name, key] * 100, [2.5, 50, 97.5])
            col = "0.15" if obs is None or lo <= obs <= hi else "0.62"
            ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="butt")
            ax.plot(mid, y, "o", color=col, ms=5)
        if obs is not None:
            ax.axvline(obs, color="#c0392b", ls="--", lw=1.4)
            right = obs < xlim[0] + 0.75 * (xlim[1] - xlim[0])
            ax.text(obs + (0.02 if right else -0.02) * (xlim[1] - xlim[0]), len(models) - 0.45,
                    f"data {obs:.2f}", color="#c0392b", ha="left" if right else "right", va="bottom", fontsize=8)
        ax.set_title(panel_title, fontsize=9, pad=14)
        ax.set_xlabel("slope (percentage points per round)")
    axes[0].set_yticks(ys, [LABEL.get(m, m) for m in models])
    axes[0].set_ylim(-0.6, len(models) - 0.05)
    if title:
        fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theory", action="store_true")
    ap.add_argument("--ppc", action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--stratest")
    ap.add_argument("--sims", type=int, default=30)
    ap.add_argument("--ext", action="store_true", help="also include the models in src/extensions.py")
    a = ap.parse_args()
    if a.ext:
        from . import extensions  # noqa: F401  (registers the extra models)
    if a.theory:
        sims = theory(a.sims)
        print(summarize(sims).to_string())
        plot(sims, "figures/theory_predictions.png",
             title="Predicted signature: median and 95% of simulated experiments")
        print("figure -> figures/theory_predictions.png")
    if a.ppc:
        from .prep import load_from_config, load_stratest_df2011, match_table, to_arrays
        df = load_stratest_df2011(a.stratest) if a.stratest else load_from_config(a.config)
        d = to_arrays(match_table(df))
        observed, sims, fitted = ppc(d, a.sims)
        print("observed slopes (pp):", {k: round(v * 100, 2) for k, v in observed.items()})
        print(summarize(sims).to_string())
        for name, params in fitted.items():
            print(name, {k: round(float(v), 3) for k, v in params.items()})
        plot(sims, "figures/posterior_predictive.png", observed,
             title="Fitted models vs data (dashed lines)")
        print("figure -> figures/posterior_predictive.png")


if __name__ == "__main__":
    main()
