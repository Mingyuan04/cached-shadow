"""Data preparation: round-level choices -> one row per (subject, match).

Standard round-level schema used everywhere in this repo
--------------------------------------------------------
session, subject, treatment, delta, R, T, S, P, match, round, coop, partner_coop
  coop / partner_coop are 1 = cooperate, 0 = defect.

Match-level table (output of `match_table`)
-------------------------------------------
one row per subject x match, with
  L      realized length of the match (rounds)
  ET     expected length under the instructions, 1 / (1 - delta)
  c1     own first-round choice  (1 = start by cooperating ~ Grim/TFT)
  o1     partner's first-round choice
  o_any  1 if the partner cooperated in any round of the match
  g, l   normalized temptation / sucker parameters (Dal Bo & Frechette 2018)
and lagged versions (L_prev, Lrel_prev = L_prev - ET, o1_prev, o_any_prev, c1_prev).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["session", "subject", "treatment", "delta", "R", "T", "S", "P",
            "match", "round", "coop", "partner_coop"]


def normalized_params(R, T, S, P):
    """g = (T-R)/(R-P), l = (P-S)/(R-P). Mutual C pays 1, mutual D pays 0."""
    R, T, S, P = map(np.asarray, (R, T, S, P))
    return (T - R) / (R - P), (P - S) / (R - P)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_stratest_df2011(path: str) -> pd.DataFrame:
    """Late-supergame subset of Dal Bo & Frechette (2011) shipped with the R
    package stratEst. Use it ONLY to test code: it starts around supergame
    20-65, so the early learning phase is missing."""
    import pyreadr  # pip install pyreadr

    raw = pyreadr.read_r(path)["DF2011"]
    df = pd.DataFrame({
        "subject": raw["id"].astype(int),
        "treatment": raw["treatment"].astype(str),
        "match": raw["game"].astype(int),
        "round": raw["period"].astype(int),
        "coop": (raw["choice"].astype(str) == "c").astype(int),
        "partner_coop": (raw["other.choice"].astype(str) == "c").astype(int),
    })
    # D5R32 -> delta = .5, R = 32 ; D75R48 -> delta = .75, R = 48
    df["delta"] = np.where(df["treatment"].str.startswith("D75"), 0.75, 0.5)
    df["R"] = df["treatment"].str.extract(r"R(\d+)")[0].astype(float)
    df["T"], df["S"], df["P"] = 50.0, 12.0, 25.0          # AER 2011, Table 1
    df = infer_sessions(df)
    return df[REQUIRED]


GUESSES = {
    "session": ["session", "sess", "session_id", "sessionid", "ses"],
    "subject": ["subject", "subj", "subject_id", "id", "player", "participant", "sid"],
    "match": ["match", "supergame", "game", "repeated_game", "repgame", "sg", "rg", "match_number"],
    "round": ["round", "period", "stage", "round_in_match", "rnd"],
    "coop": ["coop", "cooperate", "own_coop", "choice", "action", "decision", "c"],
    "partner_coop": ["ocoop", "other_coop", "partner_coop", "other_choice", "opp_coop",
                     "other", "o_coop", "partner", "opponent"],
    "treatment": ["treatment", "treat", "tr", "condition"],
    "delta": ["delta", "prob", "continuation", "p_continue", "cont_prob"],
    "R": ["r", "reward", "payoff_cc", "coop_payoff"],
}


def guess_colmap(columns) -> dict:
    """Best guess {their_name: our_name}, exact (case-insensitive) matches first."""
    lower = {c: str(c).strip().lower() for c in columns}
    out, used = {}, set()
    for ours, cands in GUESSES.items():
        for cand in cands:
            hit = [c for c, lc in lower.items() if lc == cand and c not in used]
            if hit:
                out[hit[0]] = ours
                used.add(hit[0])
                break
    return out


def to_binary(s: pd.Series, name: str, values: dict | None = None) -> pd.Series:
    """Map a cooperation column to 1 = C, 0 = D."""
    if values:
        mapped = s.astype(str).str.strip().map({str(k): v for k, v in values.items()})
        if mapped.isna().any():
            bad = sorted(map(str, pd.unique(s[mapped.isna()])))[:10]
            raise ValueError(f"{name}: no entry in coop_values for {bad}")
        return mapped.astype(int)
    uniq = set(pd.unique(s.dropna()))
    if uniq <= {0, 1}:
        return s.astype(int)
    low = s.astype(str).str.strip().str.lower()
    if set(low.unique()) <= {"c", "d", "cooperate", "defect", "coop", "def", "cooperation", "defection"}:
        return low.str.startswith("c").astype(int)
    raise ValueError(f"{name}: cannot tell C from D in values {sorted(map(str, uniq))[:10]}; "
                     f"pass coop_values, e.g. {{1: 1, 2: 0}}")


def read_any(path: str, sheet=None) -> pd.DataFrame:
    ext = path.lower().rsplit(".", 1)[-1]
    if ext in ("csv", "txt", "tsv"):
        return pd.read_csv(path, sep=None, engine="python")
    if ext == "dta":
        return pd.read_stata(path, convert_categoricals=False)
    if ext in ("xls", "xlsx"):
        return pd.read_excel(path, sheet_name=sheet if sheet is not None else 0)
    if ext in ("rda", "rdata"):
        import pyreadr
        return next(iter(pyreadr.read_r(path).values()))
    raise ValueError(f"unsupported file type: {path}")


def load_table(path: str, colmap: dict | None = None, constants: dict | None = None,
               treatment_params: dict | None = None, coop_values: dict | None = None,
               sheet=None) -> pd.DataFrame:
    """Load round-level data from a replication package (csv, dta, xls/xlsx).

    colmap           {their_column: our_column}; guessed if omitted (check it!)
    constants        design values missing from the file, e.g. {"T": 50, "S": 12, "P": 25}
    treatment_params {treatment_code: {"delta": .5, "R": 32}, ...} if the file
                     codes treatments instead of storing delta and R
    coop_values      e.g. {1: 1, 2: 0} if cooperation is not coded 0/1 or C/D
    Rounds are renumbered 1, 2, ... within each match (some files number
    decisions across the whole session); subject ids are made unique across
    sessions; sessions are inferred from shared length sequences if absent.
    """
    df = read_any(path, sheet)
    df = df.rename(columns=colmap or guess_colmap(df.columns))
    for col in ("coop", "partner_coop"):
        if col in df.columns:
            df[col] = to_binary(df[col], col, coop_values)
    for k, v in (constants or {}).items():
        df[k] = v
    if treatment_params:
        codes = df["treatment"].astype(str).str.strip()
        for k in ("delta", "R"):
            df[k] = codes.map({str(t): p[k] for t, p in treatment_params.items()})
        if df["delta"].isna().any():
            bad = sorted(codes[df["delta"].isna()].unique())[:10]
            raise ValueError(f"treatment codes missing from treatment_params: {bad}")
    if "treatment" not in df.columns and {"delta", "R"} <= set(df.columns):
        df["treatment"] = "d" + df["delta"].astype(str) + "_R" + df["R"].astype(str)
    if "session" in df.columns and df.groupby("subject")["session"].nunique().max() > 1:
        df["subject"] = df["session"].astype(str) + "_" + df["subject"].astype(str)
    if "round" in df.columns:
        df["round"] = (df.groupby(["subject", "match"])["round"]
                         .rank(method="first").astype(int))
    if "session" not in df.columns:
        df = infer_sessions(df)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns after mapping: {missing}. "
                         f"Run: python -m src.prep --inspect {path}")
    return df[REQUIRED]


load_csv = load_table          # backwards-compatible name


def load_from_config(config: str | dict) -> pd.DataFrame:
    """config: a dict or a JSON file with keys path, colmap, constants,
    treatment_params, coop_values, sheet (see data/df2011_config.example.json)."""
    import json
    cfg = json.load(open(config)) if isinstance(config, str) else dict(config)
    path = cfg.pop("path")
    cfg = {k: v for k, v in cfg.items() if not k.startswith("_")}
    if cfg.get("coop_values"):                       # JSON keys are strings
        cfg["coop_values"] = {_num(k): v for k, v in cfg["coop_values"].items()}
    return load_table(path, **cfg)


def _num(k):
    try:
        return int(k)
    except (TypeError, ValueError):
        return k


def inspect(path: str) -> None:
    """Print what is in a downloaded file (or every data file in a .zip) and a
    guessed column mapping, so you can fill in data/df2011_config.json."""
    import os
    import tempfile
    import zipfile

    if path.lower().endswith(".zip"):
        with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as tmp:
            names = [n for n in z.namelist() if n.lower().rsplit(".", 1)[-1]
                     in ("csv", "txt", "tsv", "dta", "xls", "xlsx")]
            print(f"{len(names)} data files in archive:", *names, sep="\n  ")
            for n in names:
                inspect(z.extract(n, tmp))
        return
    ext = path.lower().rsplit(".", 1)[-1]
    sheets = pd.ExcelFile(path).sheet_names if ext in ("xls", "xlsx") else [None]
    for sh in sheets:
        df = read_any(path, sh)
        print("\n" + "=" * 70)
        print(os.path.basename(path), f"[sheet {sh}]" if sh is not None else "", df.shape)
        print(df.dtypes.to_string())
        print(df.head(5).to_string())
        cm = guess_colmap(df.columns)
        print("guessed colmap:", cm)
        for theirs, ours in cm.items():
            if ours in ("coop", "partner_coop", "treatment", "delta", "R"):
                print(f"  values of {theirs!r}:", sorted(map(str, pd.unique(df[theirs].dropna())))[:12])


def infer_sessions(df: pd.DataFrame) -> pd.DataFrame:
    """All subjects in a session share the same realized length for every match
    (Mengel, Orlandi & Weidenholzer 2022). Group subjects whose
    (match -> length) sequences are identical within a treatment."""
    lengths = df.groupby(["treatment", "subject", "match"])["round"].max()
    sig = lengths.groupby(level=[0, 1]).apply(lambda s: tuple(s.droplevel([0, 1]).items()))
    keys = {k: i for i, k in enumerate(sorted(set(zip(sig.index.get_level_values(0), sig.values))))}
    sess = {(t, s): keys[(t, v)] for (t, s), v in sig.items()}
    out = df.copy()
    out["session"] = [sess[(t, s)] for t, s in zip(out["treatment"], out["subject"])]
    return out


# ---------------------------------------------------------------------------
# Match-level features
# ---------------------------------------------------------------------------
def match_table(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["subject", "match", "round"]).copy()
    # per-round normalized payoff: CC = 1, CD = -l, DC = 1 + g, DD = 0
    g_r, l_r = normalized_params(df["R"], df["T"], df["S"], df["P"])
    c, o = df["coop"].to_numpy(), df["partner_coop"].to_numpy()
    df["pay"] = c * o * 1.0 + c * (1 - o) * (-l_r) + (1 - c) * o * (1 + g_r)
    keys = ["session", "subject", "treatment", "match"]
    m = (df.groupby(keys, sort=False)
           .agg(L=("round", "max"), c1=("coop", "first"), o1=("partner_coop", "first"),
                o_any=("partner_coop", "max"), pay=("pay", "sum"),
                delta=("delta", "first"), R=("R", "first"), T=("T", "first"),
                S=("S", "first"), P=("P", "first"))
           .reset_index())
    m["ET"] = 1.0 / (1.0 - m["delta"])
    m["g"], m["l"] = normalized_params(m["R"], m["T"], m["S"], m["P"])
    m = m.sort_values(["subject", "match"]).reset_index(drop=True)
    grp = m.groupby("subject")
    for col in ["L", "c1", "o1", "o_any"]:
        m[f"{col}_prev"] = grp[col].shift(1)
    m["consecutive"] = (m["match"] - grp["match"].shift(1)) == 1
    m["Lrel_prev"] = m["L_prev"] - m["ET"]          # realized minus expected
    m["match_index"] = grp.cumcount() + 1           # 1, 2, ... within subject
    return m


def to_arrays(m: pd.DataFrame) -> dict:
    """Pad the match table into (N subjects x M matches) arrays for the models.

    Returns c1, o1, o_any, L, pay as (N, M) arrays, `mask` (1 = real match, 0 = padding),
    and per-subject vectors ET, g, l, session, treatment, subject. Padding sits
    after each subject's last match, so it never affects the likelihood.
    """
    m = m.sort_values(["subject", "match"])
    groups = list(m.groupby("subject", sort=True))
    N, M = len(groups), max(len(g) for _, g in groups)
    out = {k: np.zeros((N, M)) for k in ["c1", "o1", "o_any", "L", "pay", "mask"]}
    per = {k: [] for k in ["ET", "g", "l", "session", "treatment", "subject"]}
    for i, (sid, g) in enumerate(groups):
        n = len(g)
        for k in ["c1", "o1", "o_any", "L", "pay"]:
            out[k][i, :n] = g[k].to_numpy(dtype=float)
        out["mask"][i, :n] = 1.0
        for k in ["ET", "g", "l", "session", "treatment"]:
            per[k].append(g[k].iloc[0])
        per["subject"].append(sid)
    out.update({k: np.asarray(v) for k, v in per.items()})
    return out


# ---------------------------------------------------------------------------
# Fetch the code-test subset (not redistributed in this repo)
# ---------------------------------------------------------------------------
STRATEST_URL = "https://codeload.github.com/cran/stratEst/tar.gz/refs/heads/master"


def fetch_stratest_subset(dest: str = "data/stratEst_DF2011_late_subset.rda") -> str:
    """Download DF2011.rda from the GitHub mirror of the CRAN package stratEst."""
    import io
    import tarfile
    import urllib.request

    with urllib.request.urlopen(STRATEST_URL) as r:
        tar = tarfile.open(fileobj=io.BytesIO(r.read()), mode="r:gz")
    member = next(m for m in tar.getmembers() if m.name.endswith("data/DF2011.rda"))
    with open(dest, "wb") as f:
        f.write(tar.extractfile(member).read())
    return dest


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2 and sys.argv[1] == "--inspect":
        inspect(sys.argv[2])
    else:
        print("saved", fetch_stratest_subset())
