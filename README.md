# Cached shadow of the future

**Is "how long will this relationship last" cached as a value or as a prediction?**
A computational account of learning in indefinitely repeated cooperation, using public data.

*Status: work in progress (Fall 2026). The whole pipeline runs end to end on a public code-test subset; the planner is implemented, the two caching models are next; results wait on the full data (see Data).*

---

## 1. Question

In indefinitely repeated Prisoner's Dilemma experiments, every match continues after each round with a known probability δ, so people are *told* how long relationships last on average. Yet a long previous match makes them more cooperative in the next one. Three accounts differ in what experience leaves behind:

| account | what is carried from match to match | analogue in RL |
|---|---|---|
| **planner** | nothing about the future; the horizon comes from the instructions | model-based |
| **value cache** | payoff differences between cooperating and defecting | cached values |
| **prediction cache** | the expected number of rounds a relationship lasts | successor representation (SR) |

## 2. What is already known, and the gap

- δ does not raise round-1 cooperation in the first match, but strongly does later; a longer previous match raises round-1 cooperation (Dal Bó & Fréchette, 2011, Tables 3 and 6). They suggest people update an estimate of expected length.
- Early match lengths have lasting effects. In a reinforcement model with counterfactuals that weighs realized against expected length, the weight on realized length is about 0.43 (Mengel, Orlandi & Weidenholzer, 2022, Table 3).
- A model that reinforces the chosen first move by the realized payoff predicts cooperation out of sample (Fudenberg & Karreskog Rehbinder, 2024).

Both fitted models cache **values**: a long match matters only through the payoff it produced, so a long match against a partner who defected from the start changes (almost) nothing. Dal Bó & Fréchette's verbal account, an updated estimate of length, is a cached **prediction**: every match teaches you how long relationships last, whatever happened in it. As far as I can find, the two have not been pitted against each other. *(Check again on Google Scholar before writing the SOP.)*

## 3. The decisive signature (week 1, model-free)

Effect of the previous match's realized length on round-1 cooperation in the next match:

| | after a partner who **cooperated** first | after a partner who **never cooperated** |
|---|---|---|
| planner | none | none |
| value cache | positive | **none or negative** (Grim vs AD earns −l at any length) |
| prediction cache | positive | **positive** (scaled by the belief that partners cooperate) |

The right-hand column must use partners who defected in *every* round, not just in round 1: in long matches, partners who start by defecting sometimes cooperate later (about a quarter of matches longer than 4 rounds in the DF2011 subset), which would give payoff-based learners a reason to cooperate more too.

`src/signature.py`: linear probability model with subject fixed effects, standard errors clustered by session, plus a session-level cluster bootstrap (DF2011 has only 18 sessions). It reports three subsets: partner cooperated first, partner defected first, and the strict one.

## 4. Models (round-1 choices; `src/models.py`)

Normalized payoffs: mutual C = 1, mutual D = 0, temptation 1 + g, sucker −l. Grim vs Grim earns H over a match of H rounds; Grim vs AD −l; AD vs Grim 1 + g; AD vs AD 0. The value of starting with C, given belief p that the partner is a conditional cooperator and a horizon H:

  ΔU(p, H) = p(H − 1 − g) − (1 − p) l

**Planner** (worked example in code)
 P(C_m) = σ(β ΔU(p_m, E[T]) + b),  p_{m+1} = p_m + κ(o_m − p_m),  E[T] = 1/(1 − δ)

**Value cache** (after Mengel et al., 2022) — *to implement*
 P(C_m) = σ(a₀ + λ D_m),  D_{m+1} = D_m + α Δπ(L_m) + (1 − α) Δπ(E[T])
 Δπ(H) = H − 1 − g if the partner started with C; −l if it started with D

**Prediction cache** (SR-like) — *to implement*
 P(C_m) = σ(β ΔU(p_m, H_m) + b),  H_{m+1} = H_m + η(L_m − H_m),  H₁ = E[T]
 H is the SR entry for "being in a relationship": expected rounds still to come. η = 0 recovers the planner.

**Variant (preregistered)**: the prediction cache with a decreasing learning rate, η_m = 1/(n₀ + m), so that H is a running average of experienced lengths. Early matches matter at least as much as recent ones (Mengel et al., 2022), which a constant η cannot produce.

**Extension**: model-free reinforcement of the chosen first move (Fudenberg & Karreskog Rehbinder, 2024), and a free prior H₁.

## 5. Estimation and validation (`src/fitting.py`)

- **Population-level maximum likelihood.** Per-subject fits are weakly identified (planner recovery r ≈ 0.1–0.45 with 40–70 matches per subject). Pooled across subjects, with treatments and session length sequences supplying the variation, planner recovery is r ≥ 0.99. Individual differences need a hierarchical model (extension).
- **Checks to pass before touching real data** (synthetic design matched to DF2011: 6 treatments × 3 sessions × 14 subjects × 45 matches, lengths shared within session):
  1. parameter recovery, all parameters r > 0.95
  2. model recovery: a clean diagonal in the confusion matrix
  3. `pytest`: every theoretical property in `tests/test_models.py` passes
- **Model comparison:** BIC and leave-one-session-out cross-validated likelihood.
- **Posterior predictive check** (`src/predict.py --ppc`): fit each model, simulate it on the real environment (same partners' first moves, same match lengths), and re-run the section 3 regression on the simulated choices. The winning model must reproduce both slopes.
- **Theory figure** (`src/predict.py --theory`): what each model predicts for the section 3 slopes in simulated DF2011-sized experiments. Expected pattern: planner ≈ 0 for both; value cache > 0 only after cooperators; prediction cache > 0 for both.

## 6. Data

- **Full data (needed):** Dal Bó & Fréchette (2011), *AER* replication package, openICPSR project 112401 (free account). All matches from match 1, 266 subjects, 18 sessions.
  1. `python -m src.prep --inspect <downloaded file or .zip>` prints every data file, its columns, a guessed column mapping and the codes used for cooperation and treatments.
  2. Copy `data/df2011_config.example.json` to `data/df2011_config.json` and fill in the real column names, treatment codes and cooperation codes.
  3. Every command then takes `--config data/df2011_config.json`. The loader renumbers rounds within each match, makes subject ids unique across sessions, and infers sessions from shared length sequences if the file has none (tested on Stata, Excel and CSV variants).
- **Code-test subset:** `python -m src.prep` downloads `DF2011.rda` from the GitHub mirror of the R package stratEst (not redistributed here). It holds only late matches (about the last 10 per subject), so the early learning phase is missing. Use it to check the code runs, never for results: the models assume they start at match 1.
- **For power later:** the meta-data behind Dal Bó & Fréchette (2018) and Fudenberg & Karreskog Rehbinder (2024), with about 100–160 sessions.

## 7. Run

```bash
pip install -r requirements.txt
python -m src.prep                                        # fetch the code-test subset
python -m src.prep --inspect path/to/downloaded_file      # see what the real data look like
python -m pytest -q                                       # theory checks
python -m src.signature --config data/df2011_config.json  # week 1: the decisive slopes
python -m src.fitting --recover planner                   # recovery demo (works now)
python -m src.fitting --model-recovery                    # after the TODOs
python -m src.predict --theory                            # after the TODOs
python -m src.fitting --cv --config data/df2011_config.json
python -m src.predict --ppc --config data/df2011_config.json
```
Replace `--config data/df2011_config.json` with `--stratest data/stratEst_DF2011_late_subset.rda` to test any step on the subset.

## 8. Plan

| by | step |
|---|---|
| Oct 20 | download full data; section 3 signature on all matches; read Mengel et al. §2.3 |
| Oct 27 | implement value and prediction caches; tests and both recovery checks pass |
| Nov 5 | fit real data; BIC + cross-validation; posterior predictive check |
| Nov 10 | results figure + paragraph here; one-page summary |
| Nov 25 | extensions (model-free, free prior, meta-data) |

## References

Blonski, Ockenfels & Spagnolo (2011). Equilibrium selection in the repeated prisoner's dilemma. *AEJ: Micro*, 3, 164–192.
Dal Bó & Fréchette (2011). The evolution of cooperation in infinitely repeated games. *AER*, 101, 411–429.
Dal Bó & Fréchette (2018). On the determinants of cooperation in infinitely repeated games: A survey. *JEL*, 56, 60–114.
Fudenberg & Karreskog Rehbinder (2024). Predicting cooperation with learning models. *AEJ: Micro*, 16, 1–32.
Gershman (2018). The successor representation: Its computational logic and neural substrates. *J. Neurosci.*, 38, 7193–7200.
Mengel, Orlandi & Weidenholzer (2022). Match length realization and cooperation in indefinitely repeated games. *JET*, 200, 105416.
Wilson & Collins (2019). Ten simple rules for the computational modeling of behavioral data. *eLife*, 8, e49547.
