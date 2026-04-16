# %% [markdown]
# # Signal Research Notebook
# 
# Interactive research and development notebook for signal creation and testing.
# 
# ## Workflow
# 1. **Load & Explore Data** - Load market data and inspect distributions
# 2. **Develop Signal** - Build and test signal logic interactively
# 3. **Implement** - Copy validated logic to `create_signal.py`
# 4. **Execute** - Run `uv run create-signal` to generate `data/signal.parquet`
# 5. **Validate Signal** - use `uv run ew-dash` to view signal characteristics.
# 6. **Backtest** - Use `uv run backtest` for slurm backtest on super computer
# 7. **Performance** - Use `uv run opt-dash` for in depth analysis of mvo backtested signal.
# 
# ## Tips
# - Use cells to isolate different aspects of your signal
# - Modify parameters directly in cells to test variations
# - Check signal statistics regularly to catch issues early
# - Document your assumptions and findings as you develop

# %% [markdown]
# ## Setup

# %%
from dotenv import load_dotenv
load_dotenv() # This MUST come before the sf_quant imports

import polars as pl
import numpy as np
import datetime as dt
import sf_quant.data as sfd
import sf_quant.research as sfr
import polars_ols

# %% [markdown]
# ## 1. Load & Explore Data
# 
# Load your market data and inspect key characteristics before developing the signal.

# %%
def load_data() -> pl.DataFrame:
    """
    Load and prepare market data for signal development.
    
    Returns:
        pl.DataFrame: Market data with required columns
    """

    start = dt.date(2000,1,1)
    end = dt.date(2025,1,1)

    columns = [
        'ticker',
        'date',
        'barrid',
        'cusip',
        'price',
        'return',
        'specific_return',
        'specific_risk'
        # 'volatility'
    ]

    df = sfd.load_assets( 
        start = start,
        end = end,
        columns = columns,
        in_universe=True,

    ).filter(
        pl.col('price')
        .shift(1)
        .over('barrid')
        .gt(5)
    )

    return df

    # TODO: Load data from source (API, file, database)
    
    # TODO: Filter data as needed (date range, symbols, quality checks)

    # TODO: Inlcude Betas for backtester

    # EXAMPLE 
    
    # start = dt.date(1996, 1, 1)
    # end = dt.date(2024, 12, 31)

    # columns = [
    #     'date',
    #     'barrid',
    #     'ticker',
    #     'price',
    #     'return',
    #     'specific_risk',
    #     'predicted_beta'
    # ]

    # return sfd.load_assets(
    #     start=start,
    #     end=end,
    #     in_universe=True,
    #     columns=columns
    # ).filter(
    #     (pl.col('price')
    #     .shift(1)
    #     .gt(5))
    # )
    

df = load_data()
df

# %%
ind_df = pl.read_csv('/home/bpinoli/sub-industry-correlation/data/gics_sub_industry.csv')
ind_df

# %%
ind_df = ind_df.with_columns(
    pl.col('datadate').str.to_date('%Y-%m-%d')
)

merged_df = df.join_asof(
    other=ind_df,
    left_on='date',
    right_on='datadate',
    by='cusip',
    strategy='backward' #we need to do this in order to take the most recent before that date
                    )

merged_df

# %% [markdown]
# ## 2. Signal Development
# 
# Build and test your signal logic. Modify parameters and logic here to find optimal configurations.

# %%
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

# %%
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

# # Drop rows where more than 50% of cusips are null
# min_non_null = int(len(id_cols) * 0.5)
# returns_wide = returns_wide.filter(
#     pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in id_cols]) >= min_non_null
# )


# REPLACE WITH:
min_non_null = 20  # just need *some* stocks reporting on each date
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

# %%
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
    target_ticker: str,
    peer_tickers: list[str],
    returns_wide: pl.DataFrame,
    window: int = WINDOW,
    max_lag: int = MAX_LAG,
    n_components: int = N_COMPONENTS,
) -> pl.DataFrame:
    """
    Rolling-window PCA forecast for a single target stock.
    Returns a Polars DataFrame: date, y_true, y_pred, n_peers, var_explained.
    """
    dates = returns_wide[DATE_COL].to_numpy()
    target_arr = returns_wide[target_ticker].to_numpy().astype(np.float64)

    # Extract peer matrix once — (T x n_peers), numpy for speed
    peer_arr = returns_wide.select(peer_tickers).to_numpy().astype(np.float64)

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

# %%
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

# %%
# ── Diagnostics ────────────────────────────────────────────
total_lookback = WINDOW + MAX_LAG  # 273

# 1. How many trading days total?
print(f"Total trading days in returns_wide: {returns_wide.height}")
print(f"Need at least {total_lookback + 2} to produce any forecasts")

# 2. Does the target have data?
target_vals = returns_wide[TARGET_CUSIP].to_numpy()
non_null = (~np.isnan(target_vals.astype(float))).sum()
print(f"\nTarget non-null days: {non_null} / {len(target_vals)}")

# 3. How many peers survive the NaN filter in a typical window?
test_block = returns_wide.select(peer_cusips).tail(total_lookback).to_numpy().astype(np.float64)
valid_peers = (~np.isnan(test_block).any(axis=0)).sum()
print(f"Peers with zero NaNs in last {total_lookback}-day window: {valid_peers} / {len(peer_cusips)}")

# 4. What does peer coverage look like per-column?
null_frac = np.isnan(test_block).mean(axis=0)
print(f"Median null fraction across peers: {np.median(null_frac):.2%}")
print(f"Peers with <10% nulls: {(null_frac < 0.10).sum()}")

# %%
# ── Pick a target ──────────────────────────────────────────
TARGET_CUSIP = '602496101'  # <-- replace with your cusip of interest
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

# %%
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

# %%
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

# %%
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

# %%
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

# %%
def create_signal(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create signal based on market data.
    
    Args:
        df: Market data DataFrame
        
    Returns:
        pl.DataFrame: DataFrame with signal column added
    """
    # TODO: Implement signal calculation, include alpha logic.

    #here we are going to make matrix of assets in sub industries
    
    

    return df

signal = create_signal(df)
signal.head()

# %%
sfd.get_assets_columns()

# %%


# %% [markdown]
# ## 3. Signal Analysis
# 
# Examine signal statistics and distributions to understand its characteristics.

# %% [markdown]
# ### Statistics

# %%
# sfr.get_signal_stats(signal)

# %% [markdown]
# ### Distribution

# %%
# sfr.get_signal_distribution(signal)

# %% [markdown]
# ## 4. Validation Checks ?
# 
# Verify signal quality and identify any issues before implementation.

# %% [markdown]
# ## 5. Next Steps
# 
# When satisfied with your signal:
# 
# 1. **Copy** your data loading and signal calculation logic to `create_signal.py`
# 2. **Run** `uv run create-signal` to save the signal to `data/signal.parquet`
# 3. **Open** `uv run ew-dash` to analyze the signal before backtesting


