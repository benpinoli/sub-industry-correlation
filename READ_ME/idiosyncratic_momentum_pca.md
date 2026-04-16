# Factor-Augmented Lag Approach to Idiosyncratic Industry Momentum

---

## 1. Mathematical Formulation

### Setup

Let $\mathcal{S}_g$ be the set of $N_g$ stocks belonging to GICS sub-industry $g$ (column `gsubind`). For a target stock $i \in \mathcal{S}_g$, define:

- $\epsilon_{i,t}$ — the idiosyncratic (specific) return of stock $i$ on day $t$, taken directly from the `specific_return` column.
- $\mathbf{E}_{-i,t} \in \mathbb{R}^{N_g - 1}$ — the vector of idiosyncratic returns of all *peer* stocks $j \in \mathcal{S}_g \setminus \{i\}$ on day $t$.

Over a rolling estimation window of length $W = 252$ trading days ending at day $t$, stack the peer returns into a matrix:

$$
\mathbf{X}_{-i} = \begin{bmatrix} \mathbf{E}_{-i,\, t-W+1}^\top \\ \vdots \\ \mathbf{E}_{-i,\, t}^\top \end{bmatrix} \in \mathbb{R}^{W \times (N_g - 1)}
$$

### PCA Factor Extraction

Standardize each column of $\mathbf{X}_{-i}$ to zero mean and unit variance, yielding $\tilde{\mathbf{X}}_{-i}$.

Compute the eigendecomposition of the sample covariance matrix:

$$
\frac{1}{W-1} \tilde{\mathbf{X}}_{-i}^\top \tilde{\mathbf{X}}_{-i} = \mathbf{V} \boldsymbol{\Lambda} \mathbf{V}^\top
$$

where $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_{N_g-1})$ with $\lambda_1 \geq \lambda_2 \geq \cdots$ and $\mathbf{V} = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_{N_g-1}]$.

Retain the first $M$ components (where $M \ll N_g - 1$) and project to obtain the latent factors:

$$
\mathbf{F} = \tilde{\mathbf{X}}_{-i} \, \mathbf{V}_M \in \mathbb{R}^{W \times M}
$$

where $\mathbf{V}_M = [\mathbf{v}_1 \mid \cdots \mid \mathbf{v}_M]$. The $m$-th column $F_{m,s}$ ($s = t-W+1, \dots, t$) is the $m$-th principal component score on day $s$.

### Predictive Regression

Define the lagged factor matrix up to maximum lag $K = 21$:

$$
\mathbf{Z}_t = \bigl[ F_{1,t-1},\; F_{1,t-2},\; \dots,\; F_{1,t-K},\; F_{2,t-1},\; \dots,\; F_{M,t-K} \bigr] \in \mathbb{R}^{M \cdot K}
$$

The forecasting equation for stock $i$'s next-day idiosyncratic return is:

$$
\epsilon_{i,t+1} = \alpha_i + \sum_{m=1}^{M} \sum_{k=1}^{K} \beta_{m,k}\, F_{m,t-k+1} + \eta_{i,t+1}
$$

where $\eta_{i,t+1}$ is the forecast error. The total number of regressors is $M \times K + 1$. With $M = 3$ and $K = 21$, this is 64 parameters — far smaller than the naive $(N_g - 1) \times K$ which can easily exceed 500.

---

## 2. Methodological Trade-offs

### Why PCA beats the naive approach

| Problem | Naive OLS on raw lags | PCA factor-augmented lags |
|---|---|---|
| **Dimensionality** | $(N-1) \times K$ regressors, often $P \gg T$ | $M \times K$ regressors, $P \ll T$ by construction |
| **Multicollinearity** | Peer residuals are correlated; $\mathbf{X}^\top\mathbf{X}$ is near-singular | PCs are orthogonal by construction; no collinearity |
| **Overfitting** | Fits noise; catastrophic out-of-sample | Regularized implicitly via dimension reduction |
| **Interpretability** | Hundreds of noisy coefficients | $M$ factors with clear variance-explained ranking |

### PCA vs. Bayesian VAR (BVAR)

A BVAR with a Minnesota prior shrinks coefficients toward zero and dampens distant lags, which helps with dimensionality but still estimates $O(N^2 K)$ parameters in the posterior mean. PCA is a *hard* dimension reduction — it collapses the cross-section to $M$ factors *before* any regression, so the downstream model never sees the full $N-1$ space. The trade-off: BVAR preserves stock-level granularity (you can read off pair-specific lead-lag effects), while PCA sacrifices that for a much more stable, lower-variance forecast.

### Choosing $M$ — the number of components

Three complementary approaches:

1. **Scree / explained-variance threshold.** Retain $M$ components explaining, say, 80–90% of the cross-sectional variance. Simple and robust, but the threshold is arbitrary.

2. **Bai–Ng information criteria (IC$_{p1}$, IC$_{p2}$).** Designed for large $N$, large $T$ factor models. Penalizes overfitting by balancing fit against model complexity. Preferred in econometrics.

3. **Out-of-sample cross-validation.** Within each rolling window, hold out the last $h$ days, fit with $M = 1, 2, \dots, M_{\max}$, pick the $M$ minimizing RMSE on the held-out block. Most directly aligned with the forecasting objective but computationally heavier.

In practice, start with the scree plot, validate with Bai–Ng, and use CV if compute budget allows.

---

## 3. Python Implementation (Polars)

### Cell 1 — Imports and configuration

```python
import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────
WINDOW        = 252   # rolling estimation window (trading days)
MAX_LAG       = 21    # K: maximum lag for PC factors
N_COMPONENTS  = 3     # M: number of principal components to retain
SUBIND_COL    = 'gsubind'
RETURN_COL    = 'specific_return'
DATE_COL      = 'date'
ID_COL        = 'cusip'     # stable identifier — tickers change, CUSIPs do not
MIN_PEERS     = 5           # skip sub-industries with fewer peers
```

### Cell 2 — Preprocessing

```python
# Ensure proper types
merged_df = merged_df.with_columns(pl.col(DATE_COL).cast(pl.Date))
merged_df = merged_df.sort([ID_COL, DATE_COL])

# Pivot to wide format: rows = dates, columns = cusips, values = specific_return
returns_wide = (
    merged_df
    .pivot(on=ID_COL, index=DATE_COL, values=RETURN_COL, aggregate_function='first')
    .sort(DATE_COL)
)

# All cusip columns (everything except the date column)
id_cols = [c for c in returns_wide.columns if c != DATE_COL]

# Drop rows where more than 50% of cusips are null
min_non_null = int(len(id_cols) * 0.5)
returns_wide = returns_wide.filter(
    pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in id_cols]) >= min_non_null
)

# Build cusip -> gsubind lookup (most recent mapping per cusip)
cusip_subind = (
    merged_df
    .sort(DATE_COL)
    .group_by(ID_COL)
    .last()
    .select([ID_COL, SUBIND_COL])
)
subind_map: dict[str, int] = dict(
    zip(cusip_subind[ID_COL].to_list(), cusip_subind[SUBIND_COL].to_list())
)

print(f"Universe: {len(id_cols)} cusips, {returns_wide.height} trading days")
print(f"Sub-industries: {len(set(subind_map.values()))}")
```

### Cell 3 — Core engine: PCA + lagged regression for one stock

```python
def build_lagged_matrix(scores: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Given scores of shape (T, M), return a matrix of shape (T, M*max_lag)
    with lags 1..max_lag.  Rows with insufficient history are filled with NaN.
    """
    T, M = scores.shape
    out = np.full((T, M * max_lag), np.nan)
    for lag in range(1, max_lag + 1):
        out[lag:, (lag - 1) * M : lag * M] = scores[:-lag]
    return out


def forecast_stock_pca(
    target_cusip: str,
    peer_cusips: list[str],
    returns_wide: pl.DataFrame,
    window: int = WINDOW,
    max_lag: int = MAX_LAG,
    n_components: int = N_COMPONENTS,
) -> pl.DataFrame:
    """
    Rolling-window PCA forecast for a single target stock (identified by CUSIP).
    Returns a Polars DataFrame: date, y_true, y_pred, n_peers, var_explained.
    """
    dates = returns_wide[DATE_COL].to_numpy()
    target_arr = returns_wide[target_cusip].to_numpy().astype(np.float64)

    # Extract peer matrix once — (T x n_peers), numpy for speed
    peer_arr = returns_wide.select(peer_cusips).to_numpy().astype(np.float64)

    total_lookback = window + max_lag
    results_date, results_yt, results_yp = [], [], []
    results_np, results_ve = [], []

    for t in range(total_lookback, len(dates) - 1):
        # ── Slice the estimation window ────────────────────────
        start = t - total_lookback + 1
        end   = t + 1  # exclusive

        y_block = target_arr[start:end]              # (total_lookback,)
        X_block = peer_arr[start:end]                # (total_lookback, n_peers)

        # Drop peers with any NaN in this window
        valid_mask = ~np.isnan(X_block).any(axis=0)
        X_valid = X_block[:, valid_mask]
        n_peers = X_valid.shape[1]

        if n_peers < max(n_components, MIN_PEERS):
            continue

        # ── Step 1: PCA on peer returns ────────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_valid)

        m = min(n_components, n_peers)
        pca = PCA(n_components=m)
        pc_scores = pca.fit_transform(X_scaled)       # (total_lookback, m)
        var_explained = pca.explained_variance_ratio_.sum()

        # ── Step 2: Build lagged PC features ───────────────────
        lagged = build_lagged_matrix(pc_scores, max_lag)  # (total_lookback, m*max_lag)

        # Find rows where both y and all lags are non-NaN
        valid_rows = ~np.isnan(lagged).any(axis=1) & ~np.isnan(y_block)
        idx = np.where(valid_rows)[0]

        if len(idx) < m * max_lag + 10:
            continue

        # ── Step 3: Fit regression on estimation window ────────
        train_idx = idx[idx < (end - start - 1)]
        last_idx  = idx[-1]

        if len(train_idx) < m * max_lag + 10 or last_idx != (end - start - 1):
            continue

        y_train = y_block[train_idx]
        X_train = lagged[train_idx]

        reg = LinearRegression()
        reg.fit(X_train, y_train)

        # ── Step 4: Predict t+1 ───────────────────────────────
        X_last = lagged[[last_idx]]
        y_pred = reg.predict(X_last)[0]

        y_true = target_arr[t + 1]
        if np.isnan(y_true):
            continue

        results_date.append(dates[t + 1])
        results_yt.append(y_true)
        results_yp.append(y_pred)
        results_np.append(n_peers)
        results_ve.append(var_explained)

    return pl.DataFrame({
        'date':          results_date,
        'y_true':        results_yt,
        'y_pred':        results_yp,
        'n_peers':       results_np,
        'var_explained': results_ve,
    })
```

### Cell 4 — Run for a target stock (example)

```python
# ── Pick a target ──────────────────────────────────────────
TARGET_CUSIP = id_cols[0]  # <-- replace with your cusip of interest
target_subind = subind_map[TARGET_CUSIP]

# Find peers in the same GICS sub-industry
peer_cusips = [
    c for c in id_cols
    if c != TARGET_CUSIP and subind_map.get(c) == target_subind
]

print(f"Target: {TARGET_CUSIP}  |  Sub-industry: {target_subind}  |  Peers: {len(peer_cusips)}")

# ── Run forecast ───────────────────────────────────────────
forecast_df = forecast_stock_pca(TARGET_CUSIP, peer_cusips, returns_wide)
print(f"Forecast days generated: {forecast_df.height}")
forecast_df.head(10)
```

### Cell 5 — Evaluation metrics

```python
if forecast_df.height > 0:
    yt = forecast_df['y_true'].to_numpy()
    yp = forecast_df['y_pred'].to_numpy()

    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    mae  = np.mean(np.abs(yt - yp))
    hit_rate = np.mean(np.sign(yt) == np.sign(yp))
    ic, _ = spearmanr(yt, yp)
    avg_ve = forecast_df['var_explained'].mean()

    print(f"RMSE:              {rmse:.6f}")
    print(f"MAE:               {mae:.6f}")
    print(f"Hit rate (sign):   {hit_rate:.3f}")
    print(f"Rank IC (Spearman):{ic:.4f}")
    print(f"Avg var explained: {avg_ve:.3f}")
```

### Cell 6 — Run across all sub-industries (batch mode)

```python
all_forecasts = []

# Group cusips by sub-industry
subind_groups: dict[int, list[str]] = defaultdict(list)
for c in id_cols:
    si = subind_map.get(c)
    if si is not None:
        subind_groups[si].append(c)

for subind, available in subind_groups.items():
    if len(available) < MIN_PEERS + 1:
        continue

    for target in available:
        peers = [c for c in available if c != target]
        try:
            df = forecast_stock_pca(target, peers, returns_wide)
            if df.height > 0:
                df = df.with_columns([
                    pl.lit(target).alias('cusip'),
                    pl.lit(subind).alias('gsubind'),
                ])
                all_forecasts.append(df)
        except Exception as e:
            print(f"  ⚠ {target}: {e}")

    print(f"✓ {subind}: {len(available)} stocks done")

all_forecasts_df = pl.concat(all_forecasts)
print(f"\nTotal forecast rows: {all_forecasts_df.height}")
print(f"CUSIPs covered:      {all_forecasts_df['cusip'].n_unique()}")
```

### Cell 7 — Aggregate IC by sub-industry

```python
ic_records = []
for subind in all_forecasts_df['gsubind'].unique().to_list():
    subset = all_forecasts_df.filter(pl.col('gsubind') == subind)
    if subset.height < 30:
        continue
    ic_val, _ = spearmanr(subset['y_true'].to_numpy(), subset['y_pred'].to_numpy())
    ic_records.append({'gsubind': subind, 'rank_ic': ic_val, 'n_obs': subset.height})

ic_table = pl.DataFrame(ic_records).sort('rank_ic', descending=True)
print("Top 10 sub-industries by Rank IC:")
print(ic_table.head(10))

# Pooled IC
pooled_ic, _ = spearmanr(
    all_forecasts_df['y_true'].to_numpy(),
    all_forecasts_df['y_pred'].to_numpy(),
)
print(f"\nPooled IC: {pooled_ic:.4f}")
```

### Cell 8 — Scree plot helper (choose M)

```python
import matplotlib.pyplot as plt

def plot_scree(target_cusip, peer_cusips, returns_wide, window=WINDOW):
    """Plot explained variance ratio to help choose M."""
    X_peers = returns_wide.tail(window).select(peer_cusips).to_numpy().astype(np.float64)

    # Drop columns with any NaN
    valid_mask = ~np.isnan(X_peers).any(axis=0)
    X_valid = X_peers[:, valid_mask]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_valid)

    pca_full = PCA().fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.bar(range(1, len(cumvar) + 1), pca_full.explained_variance_ratio_,
           alpha=0.5, label='Individual')
    ax.step(range(1, len(cumvar) + 1), cumvar, where='mid',
            color='red', label='Cumulative')
    ax.axhline(0.8, ls='--', color='grey', lw=0.8, label='80% threshold')
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Variance Explained')
    ax.set_title(f'Scree Plot — {target_cusip} peers ({X_valid.shape[1]} stocks)')
    ax.legend()
    plt.tight_layout()
    plt.show()

plot_scree(TARGET_CUSIP, peer_cusips, returns_wide)
```
