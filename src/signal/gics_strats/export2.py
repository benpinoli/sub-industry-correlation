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
    strategy='backward'
)

merged_df

# %% [markdown]
# ## 2. Signal Development
# 
# **Factor-Augmented Lag Model for Idiosyncratic Industry Momentum**
# 
# For each stock $i$ in GICS sub-industry $g$:
# 1. Pivot peer idiosyncratic returns to a wide matrix
# 2. Extract $M$ principal components from the $(N-1)$ peers in a rolling 252-day window
# 3. Build lagged features from PCs (lags $1..K$)
# 4. Regress stock $i$'s idiosyncratic return on the $M \times K$ lagged PC scores (distributed lag model)
# 5. The predicted $\hat{\epsilon}_{i,t+1}$ becomes the trading signal

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
ID_COL        = 'cusip'
MIN_PEERS     = 5

# %%
# ── Helper functions ───────────────────────────────────────

def _build_lagged_matrix(scores: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Given scores of shape (T, M), return a matrix of shape (T, M*max_lag)
    with lags 1..max_lag. Rows with insufficient history are filled with NaN.
    """
    T, M = scores.shape
    out = np.full((T, M * max_lag), np.nan)
    for lag in range(1, max_lag + 1):
        out[lag:, (lag - 1) * M : lag * M] = scores[:-lag]
    return out
def _forecast_stock_pca(
    target_cusip: str,
    peer_cusips: list[str],
    returns_wide: pl.DataFrame,
    dates: np.ndarray,
    window: int = WINDOW,
    max_lag: int = MAX_LAG,
    n_components: int = N_COMPONENTS,
    refit_freq: int = 21,
) -> pl.DataFrame:
    """
    Rolling-window PCA distributed-lag forecast.
    Refits PCA + regression every refit_freq days, reuses model in between.
    """
    target_arr = returns_wide[target_cusip].to_numpy().astype(np.float64)
    peer_arr = returns_wide.select(peer_cusips).to_numpy().astype(np.float64)

    total_lookback = window + max_lag
    results_date, results_signal = [], []

    cached_scaler = None
    cached_pca = None
    cached_valid_mask = None
    cached_beta = None  # regression coefficients
    last_refit_t = -refit_freq

    for t in range(total_lookback, len(dates) - 1):
        start = t - total_lookback + 1
        end   = t + 1

        y_block = target_arr[start:end]
        X_block = peer_arr[start:end]

        # ── Refit PCA + regression every refit_freq days ──────
        if t - last_refit_t >= refit_freq:
            valid_mask = ~np.isnan(X_block).any(axis=0)
            X_valid = X_block[:, valid_mask]
            n_peers = X_valid.shape[1]

            if n_peers < max(n_components, MIN_PEERS):
                cached_pca = None
                last_refit_t = t
                continue

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_valid)

            m = min(n_components, n_peers)
            pca = PCA(n_components=m)
            pc_scores = pca.fit_transform(X_scaled)

            lagged = _build_lagged_matrix(pc_scores, max_lag)
            valid_rows = ~np.isnan(lagged).any(axis=1) & ~np.isnan(y_block)
            idx = np.where(valid_rows)[0]

            if len(idx) < m * max_lag + 10:
                cached_pca = None
                last_refit_t = t
                continue

            train_idx = idx[idx < (end - start - 1)]
            if len(train_idx) < m * max_lag + 10:
                cached_pca = None
                last_refit_t = t
                continue

            # OLS via numpy — orders of magnitude faster than sklearn
            X_train = np.column_stack([np.ones(len(train_idx)), lagged[train_idx]])
            beta, _, _, _ = np.linalg.lstsq(X_train, y_block[train_idx], rcond=None)

            cached_scaler = scaler
            cached_pca = pca
            cached_valid_mask = valid_mask
            cached_beta = beta
            last_refit_t = t

        if cached_pca is None:
            continue

        # ── Project current window using cached PCA ───────────
        X_valid = X_block[:, cached_valid_mask]
        if X_valid.shape[1] < n_components:
            continue

        col_means = np.nanmean(X_valid, axis=0)
        X_filled = np.where(np.isnan(X_valid), col_means, X_valid)
        X_scaled = cached_scaler.transform(X_filled)
        pc_scores = cached_pca.transform(X_scaled)

        lagged = _build_lagged_matrix(pc_scores, max_lag)
        last_row = lagged[-1]

        if np.isnan(last_row).any() or np.isnan(target_arr[t + 1]):
            continue

        # Predict using cached beta
        y_pred = cached_beta[0] + last_row @ cached_beta[1:]

        results_date.append(dates[t + 1].item())
        results_signal.append(y_pred)

    return pl.DataFrame({
        'date':   results_date,
        'cusip':  [target_cusip] * len(results_date),
        'signal': results_signal,
    })

# %%
def create_signal(df: pl.DataFrame) -> pl.DataFrame:
    """
    Create idiosyncratic industry momentum signal.

    For each stock, runs PCA on same-sub-industry peers' idiosyncratic returns,
    then fits a distributed lag regression on the principal component scores
    to forecast next-day idiosyncratic return. That forecast is the signal.

    Args:
        df: merged DataFrame with columns including cusip, date,
            specific_return, and gsubind.

    Returns:
        pl.DataFrame: Original DataFrame with 'signal' column added.
    """
    # ── 1. Preprocessing ───────────────────────────────────────
    df = df.with_columns(pl.col(DATE_COL).cast(pl.Date))
    df = df.sort([ID_COL, DATE_COL])

    # Pivot to wide: rows = dates, columns = cusips
    returns_wide = (
        df
        .pivot(on=ID_COL, index=DATE_COL, values=RETURN_COL, aggregate_function='first')
        .sort(DATE_COL)
    )

    id_cols = [c for c in returns_wide.columns if c != DATE_COL]

    # Drop dates with almost no data (keep lenient — per-window NaN handling does the rest)
    min_non_null = 20
    returns_wide = returns_wide.filter(
        pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in id_cols]) >= min_non_null
    )

    dates = returns_wide[DATE_COL].to_numpy()

    print(f"Universe: {len(id_cols)} cusips, {returns_wide.height} trading days")

    # ── 2. Build cusip -> gsubind lookup ────────────────────────
    cusip_subind = (
        df.sort(DATE_COL)
        .group_by(ID_COL)
        .last()
        .select([ID_COL, SUBIND_COL])
    )
    subind_map: dict[str, int] = dict(
        zip(cusip_subind[ID_COL].to_list(), cusip_subind[SUBIND_COL].to_list())
    )

    # ── 3. Group cusips by sub-industry ─────────────────────────
    subind_groups: dict[int, list[str]] = defaultdict(list)
    for c in id_cols:
        si = subind_map.get(c)
        if si is not None:
            subind_groups[si].append(c)

    print(f"Sub-industries with >= {MIN_PEERS + 1} stocks: "
          f"{sum(1 for v in subind_groups.values() if len(v) >= MIN_PEERS + 1)}")

    # ── 4. Run PCA distributed lag forecast for every stock ─────
    all_signals = []
    n_subinds = len(subind_groups)

    for i, (subind, available) in enumerate(subind_groups.items()):
        if len(available) < MIN_PEERS + 1:
            continue

        for target in available:
            peers = [c for c in available if c != target]
            try:
                sig = _forecast_stock_pca(target, peers, returns_wide, dates)
                if sig.height > 0:
                    all_signals.append(sig)
            except Exception as e:
                print(f"  ⚠ {target}: {e}")

        if (i + 1) % 10 == 0 or (i + 1) == n_subinds:
            print(f"  [{i+1}/{n_subinds}] sub-industries processed")

    signals_df = pl.concat(all_signals)
    print(f"\nSignal generated: {signals_df.height} rows, "
          f"{signals_df['cusip'].n_unique()} cusips")

    # ── 5. Join signal back to original DataFrame ──────────────
    result = df.join(
        signals_df,
        on=[DATE_COL, ID_COL],
        how='left',
    )

    return result

# %%
signal = create_signal(merged_df)
signal.head()

# %%
# we are going to do this on only 5 sub industries first.

# # Test on just 5 sub-industries
# test_subinds = merged_df[SUBIND_COL].unique().head(4).to_list()
# test_df = merged_df.filter(pl.col(SUBIND_COL).is_in(test_subinds))
# signal = create_signal(test_df)

# %% [markdown]
# ## 3. Signal Analysis
# 
# Examine signal statistics and distributions to understand its characteristics.

# %% [markdown]
# ### Statistics

# %%
sfr.get_signal_stats(signal)

# %% [markdown]
# ### Distribution

# %%
sfr.get_signal_distribution(signal)

# %% [markdown]
# ## 4. Validation Checks

# %%
# Quick sanity checks on the signal
print(f"Total rows:        {signal.height}")
print(f"Rows with signal:  {signal.filter(pl.col('signal').is_not_null()).height}")
print(f"Signal coverage:   {signal.filter(pl.col('signal').is_not_null()).height / signal.height:.1%}")
print(f"Signal mean:       {signal['signal'].mean():.6f}")
print(f"Signal std:        {signal['signal'].std():.6f}")
print(f"Date range:        {signal[DATE_COL].min()} to {signal[DATE_COL].max()}")

# %% [markdown]
# ## 5. Next Steps
# 
# When satisfied with your signal:
# 
# 1. **Copy** your data loading and signal calculation logic to `create_signal.py`
# 2. **Run** `uv run create-signal` to save the signal to `data/signal.parquet`
# 3. **Open** `uv run ew-dash` to analyze the signal before backtesting
# 4. **Backtest** with `uv run backtest` on the supercomputer


