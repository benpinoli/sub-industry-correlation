import os
import numpy as np
import polars as pl
import datetime as dt
from dotenv import load_dotenv
from collections import defaultdict
import warnings
import time

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────
WINDOW        = 252
MAX_LAG       = 21
N_COMPONENTS  = 1
SUBIND_COL    = 'gsubind'
RETURN_COL    = 'specific_return'
DATE_COL      = 'date'
ID_COL        = 'cusip'
MIN_PEERS     = 5

RESULTS_DIR   = '/home/bpinoli/sub-industry-correlation/src/signal/full_results'


# ── Helper functions ───────────────────────────────────────

def _build_lagged_matrix(scores: np.ndarray, max_lag: int) -> np.ndarray:
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
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Rolling-window PCA distributed-lag forecast.
    Refits PCA + regression every day. Uses top 1 PC by default.

    Returns:
        (signal_df, eigenvalue_df)
    """
    target_arr = returns_wide[target_cusip].to_numpy().astype(np.float64)
    peer_arr = returns_wide.select(peer_cusips).to_numpy().astype(np.float64)

    total_lookback = window + max_lag
    results_date, results_signal = [], []
    eig_date, eig_val, eig_var_explained, eig_n_peers = [], [], [], []

    for t in range(total_lookback, len(dates) - 1):
        start = t - total_lookback + 1
        end   = t + 1

        y_block = target_arr[start:end]
        X_block = peer_arr[start:end]

        # Drop peers with any NaN in this window
        valid_mask = ~np.isnan(X_block).any(axis=0)
        X_valid = X_block[:, valid_mask]
        n_peers = X_valid.shape[1]

        if n_peers < max(n_components, MIN_PEERS):
            continue

        # PCA on peer returns
        X_mean = X_valid.mean(axis=0)
        X_std = X_valid.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_scaled = (X_valid - X_mean) / X_std

        U, S, Vt = np.linalg.svd(X_scaled, full_matrices=False)
        pc_scores = U[:, :n_components] * S[:n_components]

        # Track eigenvalue: S^2/(n-1) is the eigenvalue of the cov matrix
        # Variance explained: S_1^2 / sum(S^2)
        total_var = (S ** 2).sum()
        top_eigenvalue = S[0] ** 2 / (len(y_block) - 1)
        top_var_explained = S[0] ** 2 / total_var if total_var > 0 else 0.0

        eig_date.append(dates[t].item())
        eig_val.append(top_eigenvalue)
        eig_var_explained.append(top_var_explained)
        eig_n_peers.append(n_peers)

        # Build lagged features
        lagged = _build_lagged_matrix(pc_scores, max_lag)
        last_row = lagged[-1]

        if np.isnan(last_row).any() or np.isnan(y_block).any():
            continue

        valid_rows = ~np.isnan(lagged).any(axis=1) & ~np.isnan(y_block)
        idx = np.where(valid_rows)[0]
        train_idx = idx[idx < (end - start - 1)]

        if len(train_idx) < max_lag + 10:
            continue

        # OLS via numpy
        X_train = np.column_stack([np.ones(len(train_idx)), lagged[train_idx]])
        beta, _, _, _ = np.linalg.lstsq(X_train, y_block[train_idx], rcond=None)

        # Predict t+1
        y_pred = beta[0] + last_row @ beta[1:]

        if np.isnan(target_arr[t + 1]):
            continue

        results_date.append(dates[t + 1].item())
        results_signal.append(y_pred)

    signal_df = pl.DataFrame({
        'date':   results_date,
        'cusip':  [target_cusip] * len(results_date),
        'signal': results_signal,
    })

    eigenvalue_df = pl.DataFrame({
        'date':            eig_date,
        'cusip':           [target_cusip] * len(eig_date),
        'top_eigenvalue':  eig_val,
        'var_explained':   eig_var_explained,
        'n_peers':         eig_n_peers,
    })

    return signal_df, eigenvalue_df


# ── Data loading ───────────────────────────────────────────

def load_data() -> pl.DataFrame:
    import sf_quant.data as sfd

    start = dt.date(2000, 1, 1)
    end = dt.date(2025, 1, 1)

    columns = [
        'ticker', 'date', 'barrid', 'cusip',
        'price', 'return', 'specific_return', 'specific_risk'
    ]

    df = sfd.load_assets(
        start=start,
        end=end,
        columns=columns,
        in_universe=True,
    ).filter(
        pl.col('price').shift(1).over('barrid').gt(5)
    )

    return df


def merge_industry_data(df: pl.DataFrame) -> pl.DataFrame:
    ind_df = pl.read_csv('/home/bpinoli/sub-industry-correlation/data/gics_sub_industry.csv')
    ind_df = ind_df.with_columns(
        pl.col('datadate').str.to_date('%Y-%m-%d')
    )

    merged_df = df.join_asof(
        other=ind_df,
        left_on='date',
        right_on='datadate',
        by='cusip',
        strategy='backward',
    )

    return merged_df


# ── Graphs and tables ─────────────────────────────────────

def save_eigenvalue_analysis(all_eigenvalues_df: pl.DataFrame, results_dir: str):
    """Save eigenvalue summary stats and time series plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\nGenerating eigenvalue analysis...")

    # ── 1. Overall summary table ──────────────────────────────
    summary = all_eigenvalues_df.select([
        pl.col('top_eigenvalue').mean().alias('mean'),
        pl.col('top_eigenvalue').median().alias('median'),
        pl.col('top_eigenvalue').std().alias('std'),
        pl.col('top_eigenvalue').min().alias('min'),
        pl.col('top_eigenvalue').max().alias('max'),
        pl.col('top_eigenvalue').quantile(0.25).alias('p25'),
        pl.col('top_eigenvalue').quantile(0.75).alias('p75'),
        pl.col('var_explained').mean().alias('mean_var_explained'),
        pl.col('var_explained').median().alias('median_var_explained'),
        pl.col('var_explained').std().alias('std_var_explained'),
        pl.col('var_explained').min().alias('min_var_explained'),
        pl.col('var_explained').max().alias('max_var_explained'),
        pl.col('n_peers').mean().alias('mean_peers'),
        pl.len().alias('n_observations'),
    ])

    summary_path = os.path.join(results_dir, 'eigenvalue_summary.csv')
    summary.write_csv(summary_path)
    print(f"  Saved eigenvalue summary to {summary_path}")
    print(summary)

    # ── 2. Per-sub-industry summary ───────────────────────────
    per_subind = (
        all_eigenvalues_df
        .group_by('gsubind')
        .agg([
            pl.col('top_eigenvalue').mean().alias('mean_eigenvalue'),
            pl.col('top_eigenvalue').median().alias('median_eigenvalue'),
            pl.col('var_explained').mean().alias('mean_var_explained'),
            pl.col('var_explained').median().alias('median_var_explained'),
            pl.col('n_peers').mean().alias('mean_peers'),
            pl.len().alias('n_obs'),
        ])
        .sort('mean_var_explained', descending=True)
    )

    per_subind_path = os.path.join(results_dir, 'eigenvalue_by_subindustry.csv')
    per_subind.write_csv(per_subind_path)
    print(f"  Saved per-sub-industry eigenvalue table to {per_subind_path}")
    print("\n  Top 10 sub-industries by mean variance explained:")
    print(per_subind.head(10))

    # ── 3. Time series: daily average top eigenvalue ──────────
    daily_avg = (
        all_eigenvalues_df
        .group_by('date')
        .agg([
            pl.col('top_eigenvalue').mean().alias('avg_eigenvalue'),
            pl.col('var_explained').mean().alias('avg_var_explained'),
        ])
        .sort('date')
    )

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    dates_plot = daily_avg['date'].to_list()
    axes[0].plot(dates_plot, daily_avg['avg_eigenvalue'].to_numpy(),
                 linewidth=0.5, alpha=0.8, color='steelblue')
    axes[0].set_ylabel('Avg Top Eigenvalue')
    axes[0].set_title('Daily Average Top PCA Eigenvalue (Across All Stocks)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(dates_plot, daily_avg['avg_var_explained'].to_numpy(),
                 linewidth=0.5, alpha=0.8, color='darkorange')
    axes[1].set_ylabel('Avg Variance Explained')
    axes[1].set_xlabel('Date')
    axes[1].set_title('Daily Average Variance Explained by Top PC')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(results_dir, 'eigenvalue_timeseries.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved eigenvalue time series plot to {path}")

    # ── 4. Per-sub-industry time series (top 9 by count) ─────
    top_subinds = (
        all_eigenvalues_df
        .group_by('gsubind')
        .agg(pl.len().alias('n'))
        .sort('n', descending=True)
        .head(9)['gsubind']
        .to_list()
    )

    fig, axes = plt.subplots(3, 3, figsize=(18, 12), sharex=True)
    axes_flat = axes.flatten()

    for idx, subind in enumerate(top_subinds):
        ax = axes_flat[idx]
        sub_data = (
            all_eigenvalues_df
            .filter(pl.col('gsubind') == subind)
            .group_by('date')
            .agg(pl.col('var_explained').mean().alias('avg_var_explained'))
            .sort('date')
        )
        ax.plot(sub_data['date'].to_list(), sub_data['avg_var_explained'].to_numpy(),
                linewidth=0.4, alpha=0.8)
        ax.set_title(f'SubInd {subind}', fontsize=9)
        ax.set_ylabel('Var Expl.', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=7)

    plt.suptitle('Variance Explained by Top PC — Per Sub-Industry (Top 9 by Count)', fontsize=12)
    plt.tight_layout()
    path = os.path.join(results_dir, 'eigenvalue_by_subindustry.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved per-sub-industry eigenvalue plot to {path}")


def save_classification_metrics(result: pl.DataFrame, results_dir: str):
    """Save precision/recall/F1 graph and summary table."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score, f1_score

    print("\nGenerating classification metrics...")

    clf_df = (
        result
        .filter(
            pl.col('signal').is_not_null()
            & pl.col('specific_return').is_not_null()
        )
        .with_columns([
            (pl.col('signal') > 0).cast(pl.Int32).alias('pred_up'),
            (pl.col('specific_return') > 0).cast(pl.Int32).alias('actual_up'),
        ])
    )

    # ── Daily cross-sectional metrics ─────────────────────────
    daily_metrics = (
        clf_df
        .group_by('date')
        .agg([
            ((pl.col('pred_up') == 1) & (pl.col('actual_up') == 1)).sum().alias('tp'),
            ((pl.col('pred_up') == 1) & (pl.col('actual_up') == 0)).sum().alias('fp'),
            ((pl.col('pred_up') == 0) & (pl.col('actual_up') == 1)).sum().alias('fn'),
        ])
        .sort('date')
    )

    dates_arr = daily_metrics['date'].to_list()
    tp = daily_metrics['tp'].to_numpy().astype(float)
    fp = daily_metrics['fp'].to_numpy().astype(float)
    fn = daily_metrics['fn'].to_numpy().astype(float)

    roll = 21
    roll_dates, roll_prec, roll_rec, roll_f1 = [], [], [], []

    for i in range(roll, len(dates_arr)):
        tp_sum = tp[i - roll:i].sum()
        fp_sum = fp[i - roll:i].sum()
        fn_sum = fn[i - roll:i].sum()

        prec = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0
        rec  = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        roll_dates.append(dates_arr[i])
        roll_prec.append(prec)
        roll_rec.append(rec)
        roll_f1.append(f1)

    # ── Rolling plot ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(roll_dates, roll_prec, label='Precision', alpha=0.7, linewidth=0.8)
    ax.plot(roll_dates, roll_rec, label='Recall', alpha=0.7, linewidth=0.8)
    ax.plot(roll_dates, roll_f1, label='F1 Score', alpha=0.9, linewidth=1.2, color='black')
    ax.axhline(0.5, ls='--', color='grey', lw=0.8, label='Random baseline')
    ax.set_xlabel('Date')
    ax.set_ylabel('Score')
    ax.set_title('Rolling 21-Day Cross-Sectional Classification Metrics')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(results_dir, 'classification_rolling.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved rolling classification plot to {path}")

    # ── Overall table 2010–2024 ───────────────────────────────
    subset = clf_df.filter(
        (pl.col('date') >= pl.date(2010, 1, 1))
        & (pl.col('date') <= pl.date(2024, 12, 31))
    )

    p_all = subset['pred_up'].to_numpy()
    a_all = subset['actual_up'].to_numpy()

    summary = pl.DataFrame({
        'Metric':    ['Precision', 'Recall', 'F1 Score', 'Accuracy', 'N'],
        'Value':     [
            precision_score(a_all, p_all, zero_division=0),
            recall_score(a_all, p_all, zero_division=0),
            f1_score(a_all, p_all, zero_division=0),
            (p_all == a_all).mean(),
            float(len(a_all)),
        ],
    })

    summary_path = os.path.join(results_dir, 'classification_summary_2010_2024.csv')
    summary.write_csv(summary_path)
    print(f"  Saved classification summary to {summary_path}")
    print(summary)


def save_signal_diagnostics(result: pl.DataFrame, results_dir: str):
    """Save signal coverage and distribution diagnostics."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\nGenerating signal diagnostics...")

    sig_filled = result.filter(pl.col('signal').is_not_null())

    diagnostics = pl.DataFrame({
        'Metric': [
            'Total rows', 'Rows with signal', 'Signal coverage',
            'Signal mean', 'Signal std', 'Signal min', 'Signal max',
            'Signal p25', 'Signal p50', 'Signal p75',
            'Date min', 'Date max',
        ],
        'Value': [
            str(result.height),
            str(sig_filled.height),
            f"{sig_filled.height / result.height:.1%}",
            f"{result['signal'].mean():.6f}",
            f"{result['signal'].std():.6f}",
            f"{result['signal'].min():.6f}",
            f"{result['signal'].max():.6f}",
            f"{result['signal'].quantile(0.25):.6f}",
            f"{result['signal'].quantile(0.50):.6f}",
            f"{result['signal'].quantile(0.75):.6f}",
            str(result[DATE_COL].min()),
            str(result[DATE_COL].max()),
        ],
    })

    diag_path = os.path.join(results_dir, 'signal_diagnostics.csv')
    diagnostics.write_csv(diag_path)
    print(f"  Saved signal diagnostics to {diag_path}")
    print(diagnostics)

    # ── Signal distribution histogram ─────────────────────────
    sig_vals = sig_filled['signal'].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sig_vals, bins=200, alpha=0.7, color='steelblue', edgecolor='none')
    ax.axvline(0, ls='--', color='red', lw=1)
    ax.set_xlabel('Signal Value (Predicted Idio Return)')
    ax.set_ylabel('Frequency')
    ax.set_title('Signal Distribution')
    ax.set_xlim(np.percentile(sig_vals, 1), np.percentile(sig_vals, 99))
    plt.tight_layout()
    path = os.path.join(results_dir, 'signal_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved signal distribution plot to {path}")


# ── Main signal creation ───────────────────────────────────

def create_signal():
    """
    Loads data, creates idiosyncratic industry momentum signal,
    saves signal parquet and all analysis outputs to full_results/.
    """
    load_dotenv()
    project_root = os.getcwd()
    output_path = os.getenv("SIGNAL_PATH", "data/signal.parquet")
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────
    print("Loading market data...")
    df = load_data()
    print(f"  Loaded {df.height} rows in {time.time() - t0:.0f}s")

    print("Merging industry classifications...")
    merged_df = merge_industry_data(df)

    # ── 2. Preprocessing ──────────────────────────────────────
    print("Preprocessing...")
    merged_df = merged_df.with_columns(pl.col(DATE_COL).cast(pl.Date))
    merged_df = merged_df.sort([ID_COL, DATE_COL])

    returns_wide = (
        merged_df
        .pivot(on=ID_COL, index=DATE_COL, values=RETURN_COL, aggregate_function='first')
        .sort(DATE_COL)
    )

    id_cols = [c for c in returns_wide.columns if c != DATE_COL]

    min_non_null = 20
    returns_wide = returns_wide.filter(
        pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in id_cols]) >= min_non_null
    )

    dates = returns_wide[DATE_COL].to_numpy()
    print(f"  Universe: {len(id_cols)} cusips, {returns_wide.height} trading days")

    # ── 3. Build cusip -> gsubind lookup ──────────────────────
    cusip_subind = (
        merged_df.sort(DATE_COL)
        .group_by(ID_COL)
        .last()
        .select([ID_COL, SUBIND_COL])
    )
    subind_map: dict[str, int] = dict(
        zip(cusip_subind[ID_COL].to_list(), cusip_subind[SUBIND_COL].to_list())
    )

    # ── 4. Group cusips by sub-industry ───────────────────────
    subind_groups: dict[int, list[str]] = defaultdict(list)
    for c in id_cols:
        si = subind_map.get(c)
        if si is not None:
            subind_groups[si].append(c)

    eligible = [(si, v) for si, v in subind_groups.items() if len(v) >= MIN_PEERS + 1]
    total_stocks = sum(len(v) for _, v in eligible)
    print(f"  Sub-industries with >= {MIN_PEERS + 1} stocks: {len(eligible)}")
    print(f"  Total stocks to process: {total_stocks}")

    # ── 5. Run PCA distributed lag forecast ───────────────────
    print(f"\nStarting signal generation at {dt.datetime.now().strftime('%H:%M:%S')}...")
    all_signals = []
    all_eigenvalues = []
    stocks_done = 0

    for i, (subind, available) in enumerate(eligible):
        for target in available:
            peers = [c for c in available if c != target]
            try:
                sig_df, eig_df = _forecast_stock_pca(target, peers, returns_wide, dates)
                if sig_df.height > 0:
                    all_signals.append(sig_df)
                if eig_df.height > 0:
                    eig_df = eig_df.with_columns(pl.lit(subind).alias('gsubind'))
                    all_eigenvalues.append(eig_df)
            except Exception as e:
                print(f"  ⚠ {target}: {e}")
            stocks_done += 1

        elapsed = time.time() - t0
        pct = stocks_done / total_stocks * 100
        eta = elapsed / stocks_done * (total_stocks - stocks_done) if stocks_done > 0 else 0
        print(f"  [{i+1}/{len(eligible)}] {subind} done "
              f"({stocks_done}/{total_stocks} stocks, {pct:.1f}%, "
              f"elapsed {elapsed/60:.0f}m, ETA {eta/60:.0f}m)")

    signals_df = pl.concat(all_signals)
    print(f"\nSignal generated: {signals_df.height} rows, "
          f"{signals_df['cusip'].n_unique()} cusips")

    # ── 6. Join signal back and save ──────────────────────────
    result = merged_df.join(
        signals_df,
        on=[DATE_COL, ID_COL],
        how='left',
    )

    result.write_parquet(output_path)
    print(f"Saved signal to {output_path}")

    # ── 7. Eigenvalue analysis ────────────────────────────────
    all_eigenvalues_df = pl.concat(all_eigenvalues)
    eig_parquet_path = os.path.join(RESULTS_DIR, 'eigenvalues.parquet')
    all_eigenvalues_df.write_parquet(eig_parquet_path)
    print(f"Saved eigenvalue data to {eig_parquet_path}")

    save_eigenvalue_analysis(all_eigenvalues_df, RESULTS_DIR)

    # ── 8. Classification metrics ─────────────────────────────
    save_classification_metrics(result, RESULTS_DIR)

    # ── 9. Signal diagnostics ─────────────────────────────────
    save_signal_diagnostics(result, RESULTS_DIR)

    # ── Done ──────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"  Signal:       {output_path}")
    print(f"  Results:      {RESULTS_DIR}/")
    print(f"  Total time:   {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    create_signal()
