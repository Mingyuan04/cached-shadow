"""Model-agnostic signature: does a long previous match raise first-round
cooperation in the next one, even when the previous partner never cooperated?

    planner           no length effect after either kind of partner
    value cache       an effect only after partners who cooperated; a long
                      match with a defector pays no more than a short one
    prediction cache  an effect after both, because every match updates the
                      expected length of relationships

The test uses partners who defected in every round, not just in round 1:
in long matches, partners who start with D sometimes cooperate later, which
would give payoff-based learners a reason to cooperate more as well.

Linear probability model with subject fixed effects, SEs clustered by
session, and a session-level bootstrap because there are only 18 sessions.

    python -m src.signature --config data/df2011_config.json
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from .prep import load_from_config, load_stratest_df2011, match_table

REGRESSORS = ["Lrel_prev", "c1_prev", "log_match"]


def _design(m: pd.DataFrame) -> pd.DataFrame:
    x = m[m["consecutive"]].dropna(subset=["L_prev", "o1_prev", "c1_prev"]).copy()
    if "o_any_prev" not in x.columns:
        x["o_any_prev"] = x["o1_prev"]
    x["log_match"] = np.log(x["match_index"])
    x["partner_defected_prev"] = 1 - x["o1_prev"]
    return x


def fe_ols(x: pd.DataFrame, y: str = "c1", xs=REGRESSORS, cluster: str = "session"):
    """OLS with subject fixed effects (within transformation) and CR1
    cluster-robust standard errors. Returns (beta, se) as pandas Series."""
    cols = [y] + list(xs)
    d = x[cols] - x.groupby("subject")[cols].transform("mean")
    Y, X = d[y].to_numpy(), d[list(xs)].to_numpy()
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ Y
    u = Y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    groups = x[cluster].to_numpy()
    uniq = np.unique(groups)
    for gid in uniq:
        idx = groups == gid
        s = X[idx].T @ u[idx]
        meat += np.outer(s, s)
    G, N = len(uniq), len(Y)
    K = X.shape[1] + x["subject"].nunique()          # count absorbed fixed effects
    V = XtX_inv @ meat @ XtX_inv * G / (G - 1) * (N - 1) / max(N - K, 1)
    return pd.Series(beta, index=xs), pd.Series(np.sqrt(np.diag(V)), index=xs)


def cluster_bootstrap(x: pd.DataFrame, coef: str = "Lrel_prev", n_boot: int = 999, seed: int = 0):
    """Resample whole sessions with replacement; percentile 95% CI for one coefficient."""
    rng = np.random.default_rng(seed)
    sessions = x["session"].unique()
    by_sess = {s: x[x["session"] == s] for s in sessions}
    draws = []
    for _ in range(n_boot):
        pick = rng.choice(sessions, size=len(sessions), replace=True)
        parts = []
        for k, s in enumerate(pick):            # relabel so repeated sessions stay distinct
            p = by_sess[s].copy()
            p["session"] = k
            p["subject"] = p["subject"].astype(str) + f"_{k}"
            parts.append(p)
        b, _ = fe_ols(pd.concat(parts))
        draws.append(b[coef])
    return np.percentile(draws, [2.5, 97.5])


def run(m: pd.DataFrame, n_boot: int = 999) -> pd.DataFrame:
    x = _design(m)
    rows = []
    for label, sub in [("previous partner cooperated in round 1", x[x["o1_prev"] == 1]),
                       ("previous partner defected in round 1", x[x["o1_prev"] == 0]),
                       ("STRICT: previous partner never cooperated", x[x["o_any_prev"] == 0])]:
        b, se = fe_ols(sub)
        lo, hi = cluster_bootstrap(sub, n_boot=n_boot) if n_boot else (np.nan, np.nan)
        rows.append({"subset": label, "n": len(sub), "subjects": sub["subject"].nunique(),
                     "sessions": sub["session"].nunique(),
                     "slope per extra round": b["Lrel_prev"], "cluster SE": se["Lrel_prev"],
                     "boot 2.5%": lo, "boot 97.5%": hi})
    return pd.DataFrame(rows)


def plot(m: pd.DataFrame, path: str = "figures/signature.png") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = _design(m)
    # Raw means mix treatments (only delta = .75 can produce very short-relative
    # matches), so plot WITHIN-subject cooperation: subtract each subject's mean,
    # add back the grand mean.
    x["c1_within"] = x["c1"] - x.groupby("subject")["c1"].transform("mean") + x["c1"].mean()
    edges = [-np.inf, -1.5, -0.5, 0.5, 2.5, 5.5, np.inf]
    labels = ["≤ −2", "−1", "0", "+1–2", "+3–5", "≥ +6"]
    x["bin"] = pd.cut(x["Lrel_prev"], edges, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 4))
    subsets = [(x["o1_prev"] == 1, "previous partner cooperated first", "o", "-"),
               (x["o_any_prev"] == 0, "previous partner never cooperated", "s", "--")]
    for sel, name, mk, ls in subsets:
        s = x[sel].groupby("bin", observed=False)["c1_within"].agg(["mean", "sem"])
        ax.errorbar(range(len(labels)), s["mean"], yerr=s["sem"], marker=mk, ls=ls, capsize=3, label=name)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("previous match: realized minus expected length (rounds)")
    ax.set_ylabel("round-1 cooperation in next match\n(within-subject, grand mean added back)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stratest", help="path to the stratEst DF2011 .rda (code test only)")
    ap.add_argument("--config", help="JSON config for the full data (see data/df2011_config.example.json)")
    ap.add_argument("--n_boot", type=int, default=999)
    a = ap.parse_args()
    df = load_stratest_df2011(a.stratest) if a.stratest else load_from_config(a.config)
    m = match_table(df)
    pd.set_option("display.width", 160)
    print(run(m, n_boot=a.n_boot).round(4).to_string(index=False))
    plot(m)
    print("figure -> figures/signature.png")


if __name__ == "__main__":
    main()
