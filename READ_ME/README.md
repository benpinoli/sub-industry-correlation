# NAICS Sub-Industry PC1 Momentum Strategy

## Research Journey — From Hypothesis to Signal

This document traces the full development of an idiosyncratic industry momentum signal, from initial economic intuition through iterative model development, covering every hypothesis tested, every dead end encountered, and the final working pipeline.

## TL;DR:
We thought we found predictive power but actually had lookahead bias for those numbers. Fixing the code reveals a much weaker, sharpe ~.2 result. 
More tuning with principle component window and lag parameters + other regularization COULD find alpha. This could be continued in the future. 

---

## 1. Economic Hypothesis

**Core thesis:** Within a narrowly defined sub-industry, one firm's idiosyncratic gain is often another firm's idiosyncratic loss. When Firm A wins a contract, Firm B loses it. Crucially, this competitive transfer is **not reflected in stock prices simultaneously** — the winner reacts within hours while the loser adjusts over subsequent days as analysts revise estimates and institutional investors rebalance. This non-simultaneous price discovery creates a predictable lead-lag structure in idiosyncratic returns.

**Why idiosyncratic returns?** Total returns are dominated by market-wide and sector-level factors. By stripping out the systematic component via a multi-factor risk model, we isolate the *specific return* — the portion attributable to firm-specific competitive dynamics. It is within these residuals that the zero-sum competitive signal lives.

**The approach:** Extract the dominant latent factor from peer idiosyncratic returns via PCA, then use lagged values of this factor to forecast each stock's next-period residual return.

---

## 2. The Naive Approach and Its Failure

The most direct test would regress stock *i*'s return on the lagged idiosyncratic returns of all (N-1) peers over K=21 trading days. With typical sub-industries containing 30+ stocks, this produces 500+ regressors — far exceeding the number of observations in any reasonable estimation window.

This fails due to:
- **Curse of dimensionality:** P ≫ T in most windows
- **Extreme multicollinearity:** Peer residuals are correlated; the design matrix is near-singular
- **Overfitting:** The model fits microstructure noise and produces catastrophic out-of-sample performance

**Solution:** Factor-Augmented Lags via PCA. Instead of using raw peer returns, extract M principal components (where M ≪ N-1), then build lagged features from these orthogonal factors. With M=1 and K=21 lags, this reduces 500+ parameters to just 22 (21 lag coefficients + intercept).

> 📄 Mathematical formulation documented in `idiosyncratic_momentum_pca.md`  
---

## 3. Iteration 1 — GICS Sub-Industry, Daily PCA, Daily Lags

**Files:**
- `signal_research.ipynb` — Interactive notebook for development
- `create_signal.py` — Standalone signal generation script (GICS version)
- `pca_eigenvalue_analysis.py` — GICS eigenvalue analysis (no regression)

**Design:**
- **Peer grouping:** GICS sub-industry (`gsubind`), ~170 groups
- **Identifier:** CUSIP (stable across ticker changes, unlike tickers)
- **PCA:** Equal-weighted sample covariance, rolling 252-day window, recomputed daily
- **Regression:** 21 daily lags of PC1, OLS with 252-day rolling window
- **Prediction target:** Next-day idiosyncratic return

**Results:**
- Variance explained by PC1: ~10-20% (indicating weak common factor structure within GICS groups)
- Directional accuracy: **~50%** (no better than coin flip)
- Rank IC: negligible

**Diagnosis:** GICS sub-industries are too broad. Firms classified together aren't necessarily direct competitors. The "common idiosyncratic signal" captured by PC1 is too noisy to generate predictive power at the daily frequency.

---

## 4. Iteration 2 — Switch to NAICS-6 Classifications

**Files:**
- `pca_eigenvalue_naics.py` / `pca1_eigenvalue_naics.py` — NAICS eigenvalue analysis
- `run_pca.sh` / `run_naics_pca1.sh` / `run_naics_pca3.sh` — SLURM job scripts
- `2022_NAICS_Structure.xlsx` — Census Bureau NAICS code-to-name mapping
- `NAICS_00-26.csv` — CUSIP-to-NAICS mapping from Compustat

**Rationale:** NAICS 6-digit codes are more granular than GICS sub-industries (~1,000 codes vs ~170), producing tighter peer groups of genuine competitors.

**Results:**
- 225+ eligible NAICS groups (vs 150 GICS)
- Several NAICS groups showed **70%+ variance explained** by PC1
- English-labeled graphs revealed which specific industries had the strongest factor structures (e.g., corrugated box manufacturing, amusement parks, travel agencies)

**Observation:** Small groups (2-5 peers) mechanically produce high variance explained. The most meaningful results came from groups with 10+ peers that still showed high PC1 dominance.

---

## 5. Iteration 3 — EWMA PCA for Eigenvector Stability

**Files:**
- `EWMA_pass1_extract_pc_scores.py` — Pass 1: Extract daily PC1 scores using EWMA covariance
- `EWMA_pass1.sh` — SLURM job for Pass 1

**Problem identified:** Computing PCA independently each day produces eigenvectors that can rotate substantially between adjacent days. When we lag PC scores across days, we're stacking projections onto different axes into a single regression — the coefficients are uninterpretable.

**Three approaches considered:**

| Approach | Description | Verdict |
|----------|-------------|---------|
| Single-pass | PCA + regression every day on full 273-day block | Correct but 15-20 hours compute, no separation of PCA from regression |
| Two-pass, equal-weighted | Precompute daily PC scores, run regression separately | **Rejected** — eigenvectors rotate too much between days |
| Two-pass, EWMA | Precompute PC scores using exponentially weighted covariance | **Selected** — eigenvectors evolve smoothly |

**EWMA covariance details:**
- Half-life: 63 trading days (~3 months), decay λ = 0.9891
- Adjacent days' covariance matrices differ by only ~1%, so eigenvectors are nearly identical
- Sign-flip correction: if `dot(v_today, v_yesterday) < 0`, negate today's eigenvector (handles arbitrary sign ambiguity in PCA)

**Output:** `pc_scores_ewma.parquet` — daily PC1 scores and target idiosyncratic returns for every stock, with stable eigenvectors suitable for lagging.

> 📄 EWMA methodology documented in `ewma_methodology_memo.docx`

---

## 6. Iteration 4 — Daily Lag Regression (Pass 2)

**Files:**
- `EWMA_pass2_lag_regression.py` — 21 daily lags, 252-day rolling OLS
- `pass2_weekly_lags.py` — 4 weekly-averaged lags (5 parameters)

**Attempts:**
1. **21 daily lags (22 parameters)** → 50% accuracy, no directional edge
2. **4 weekly-averaged lags (5 parameters)** → 50% accuracy, same result

**Conclusion:** Lagged peer PC1 scores do not predict **next-day** idiosyncratic returns. The competitive transfer effect, if it exists, doesn't operate at the daily frequency. The signal-to-noise ratio at daily horizons is too low.

---

## 7. Iteration 5 — Monthly Prediction Horizon

**Files:**
- `EWMA_pass2_monthly_overlapping.py` — Predict 21-day cumulative idio return, overlapping targets
- `EWMA_pass2_monthly_nonoverlapping.py` — Same, non-overlapping targets for training
- `run_monthly_signals.sh` — SLURM job for both variants

**Key insight:** Instead of predicting tomorrow's return, predict the **cumulative idiosyncratic return over the next 21 trading days**. Same 4 weekly-averaged PC1 lags as features, but the target is now monthly, not daily.

THESE HAVE LOOKAHEAD BIAS!

**Results (overlapping):**
| Metric | Value |
|--------|-------|
| Mean Rank IC | 0.179 |
| Median Rank IC | 0.172 |
| IC > 0 % | 100% |
| IC IR | 2.69 |
| Precision | 55.97% |
| Accuracy | 55.93% |

**Results (non-overlapping):**
| Metric | Value |
|--------|-------|
| Mean Rank IC | 0.077 |
| IC > 0 % | 90.9% |
| IC IR | 0.938 |
| Accuracy | 52.95% |

The monthly horizon works where daily did not. The competitive transfer effect is real — it just operates over weeks, not days. Noise averages out over the 21-day window, revealing the underlying signal.

---

## 8. Lookahead Bias Discovery

⚠️ **Critical finding by subsequent review:** All Pass 2 scripts before the `nolookahead/` folder contained lookahead bias in two forms:

1. **Training target contamination:** The OLS training window at time t included `y_monthly[t-1]` as a target, which equals `sum(y[t:t+21])` — incorporating returns from the current day forward. This leaks future information into the regression coefficients.

2. **Survivorship conditioning:** The line `if np.isnan(y_monthly[t]): continue` meant signals were only emitted on days where the 21-day forward return was fully observed. This conditions signal existence on future data availability.

**Biased files (do not use for final results):**
- `data/signal_monthly_overlapping.parquet`
- `data/signal_monthly_nonoverlapping.parquet`
- Any `signal_final.parquet` derived from these

**Clean files (produced by `nolookahead/` scripts):**
- `data/signal_monthly_overlapping_nolook.parquet`
- `data/signal_monthly_nonoverlapping_nolook.parquet`

> The strong IC numbers reported in Iteration 7 above are from the biased signals. Clean signal performance is reported from the `nolookahead/` pipeline.

---

## 9. Hyperparameter Summary

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| EWMA half-life (h) | 63 trading days | ~3 months; eigenvector stability while adapting to structural changes |
| EWMA decay (λ) | 0.9891 | exp(−ln(2) / 63) |
| PCA window (W) | 252 trading days | 1 year, standard in equity risk models |
| Number of PCs (M) | 1 | Top principal component only |
| OLS lag length (K) | 4 weekly averages | Covers ~20 trading days, 5 total regression parameters |
| OLS estimation window | 252 trading days | Rolling 1-year, matches PCA window |
| Prediction target | 21-day cumulative idio return | Monthly horizon where signal has power |
| Minimum peers | 2 | 3 stocks total per NAICS group |
| Industry grouping | NAICS 6-digit | More granular than GICS, tighter peer groups |
| Identifier | CUSIP | Stable across ticker changes and M&A |
| Rebalancing frequency | Monthly (every 21st trading day) | Matches prediction horizon |
| MVO gamma | Tuned for 1.5-2.5x leverage | Risk aversion parameter in mean-variance optimizer |

---

## 10. Signal Preparation for Backtesting

**File:** `data/VS_claude_pred_beta_adder.py`

The raw signal parquet from Pass 2 contains only `date, cusip, signal`. The MVO backtester requires `date, barrid, signal, alpha, return, predicted_beta, specific_risk`. This script:

1. Joins signal onto market data (via CUSIP → BARRID mapping)
2. Subsamples to monthly rebalance dates (first trading day of each calendar month)
3. Computes alpha as cross-sectional z-score with guarded division (handles std=0 edge cases)
4. Filters to rows with clean optimizer inputs (no NaN/Inf in alpha, beta, or risk)
5. Outputs `data/signal_final.parquet`

---

## 11. File Tree

```
sub-industry-correlation/
├── .env                                          # DB paths, SLURM config, gamma, email
├── Makefile                                      # ← MODIFIED: added claude-ew-dash, claude-opt-dash
│
├── data/
│   ├── NAICS_00-26.csv                           # CUSIP → NAICS mapping (Compustat)
│   ├── 2022_NAICS_Structure.xlsx                 # Census Bureau NAICS code → English name
│   ├── gics_sub_industry.csv                     # CUSIP → GICS sub-industry mapping
│   ├── pred_beta_adder.py                        # original signal prep — DO NOT USE (has bugs)
│   ├── VS_claude_pred_beta_adder.py              # ← fixed signal prep → signal_final.parquet
│   ├── signal_monthly_overlapping.parquet         # ⚠️ BIASED (look-ahead in Pass 2 training)
│   ├── signal_monthly_nonoverlapping.parquet      # ⚠️ BIASED
│   ├── signal_monthly_overlapping_nolook.parquet  # ✓ CLEAN
│   ├── signal_monthly_nonoverlapping_nolook.parquet # ✓ CLEAN
│   ├── signal_final.parquet                       # MVO input (update to point at _nolook)
│   └── weights/                                   # MVO optimizer output (one parquet per year)
│
├── src/
│   ├── framework/
│   │   ├── run_backtest.py                        # submits SLURM MVO backtest
│   │   ├── opt_dash.py                            # original MVO dashboard — has Sharpe/div bugs
│   │   ├── ew_dash.py                             # original EW dashboard — has Sharpe bug
│   │   ├── claude_opt_dash.py                     # ← fixed MVO dashboard
│   │   └── claude_ew_dash.py                      # ← fixed EW dashboard
│   │
│   └── signal/
│       ├── full_results/                          # GICS eigenvalue analysis outputs
│       ├── full_results_naics/                    # NAICS eigenvalue analysis outputs
│       ├── full_results_naics_EWMA/               # EWMA PCA + signal outputs
│       │   ├── pc_scores_ewma.parquet             # Pass 1 output: daily PC1 scores
│       │   ├── eigenvalue_summary.csv
│       │   ├── eigenvalue_by_naics.csv
│       │   ├── eigenvalue_timeseries.png
│       │   ├── eigenvalue_by_naics.png
│       │   ├── classification_rolling.png
│       │   ├── classification_rolling_weekly.png
│       │   ├── signal_distribution.png
│       │   └── signal_distribution_weekly.png
│       │
│       ├── full_results_monthly_overlapping/       # Monthly overlapping signal diagnostics
│       │   ├── rank_ic_summary.csv
│       │   ├── rank_ic_plots.png
│       │   ├── classification_summary.csv
│       │   └── signal_distribution.png
│       │
│       ├── full_results_monthly_nonoverlapping/    # Monthly non-overlapping diagnostics
│       │   ├── rank_ic_summary.csv
│       │   ├── rank_ic_plots.png
│       │   ├── classification_summary.csv
│       │   └── signal_distribution.png
│       │
│       ├── gics_strats/                            # Early GICS-based strategies
│       │
│       └── naics_strats/
│           ├── pca1_eigenvalue_naics.py            # NAICS eigenvalue analysis (top 1 PC)
│           ├── pca3_eigenvalue_naics.py            # NAICS eigenvalue analysis (top 3 PCs)
│           │
│           └── EWMA_2_pass/
│               ├── EWMA_pass1_extract_pc_scores.py # Pass 1: EWMA PCA → pc_scores_ewma.parquet
│               ├── EWMA_pass1.sh                   # SLURM job for Pass 1
│               ├── EWMA_pass2_lag_regression.py     # Daily lag Pass 2 (early version)
│               ├── EWMA_pass2.sh
│               │
│               ├── weekly_pass_2/
│               │   ├── pass2_weekly_lags.py         # Weekly-averaged lags, daily prediction
│               │   └── EWMA_weekly_lag.sh
│               │
│               ├── monthly_cum_idio_ret/
│               │   ├── EWMA_pass2_monthly_overlapping.py    # ⚠️ BIASED
│               │   ├── EWMA_pass2_monthly_nonoverlapping.py # ⚠️ BIASED
│               │   └── run_monthly_signals.sh
│               │
│               └── nolookahead/                             # ✓ ALL CLEAN FILES
│                   ├── nolook_EWMA_pass2_monthly_overlapping.py
│                   ├── nolook_EWMA_pass2_monthly_nonoverlapping.py
│                   └── run_nolook_overlapping.sh
```

---

## How to Run the Full Pipeline (Clean Version)

```bash
# 1. Pass 1 — Extract EWMA PC1 scores (~13 hours on SLURM)
sbatch src/signal/naics_strats/EWMA_2_pass/EWMA_pass1.sh

# 2. Pass 2 — No-lookahead monthly prediction (~1 hour on SLURM)
sbatch src/signal/naics_strats/EWMA_2_pass/nolookahead/run_nolook_overlapping.sh

# 3. Signal prep — Format for MVO backtester
source .venv/bin/activate
python data/VS_claude_pred_beta_adder.py

# 4. Backtest — MVO optimization via SLURM
make run-backtest

# 5. Analyze results
make claude-opt-dash    # MVO portfolio performance
make claude-ew-dash     # Equal-weight quantile analysis
```
