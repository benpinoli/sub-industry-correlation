"""
Pass 2: Load precomputed EWMA PC1 scores from Pass 1, run distributed lag
regressions for each stock with a 252-day rolling window, save signal parquet.

Because the EWMA eigenvector is stable day-to-day, lagging PC scores
across days is valid.

Expects: full_results_naics/pc_scores_ewma.parquet
Saves:   data/signal.parquet + classification/diagnostic outputs
"""
import os
import numpy as np
import polars as pl
import datetime as dt
from dotenv import load_dotenv
import warnings
import time

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────
MAX_LAG       = 21
REG_WINDOW    = 252   # rolling OLS window
DATE_COL      = 'date'
ID_COL        = 'cusip'
RETURN_COL    = 'specific_return'
INDUSTRY_COL  = 'naics'

PC_SCORES_PATH = '/home/bpinoli/sub-industry-correlation/src/signal/full_results_naics_EWMA/pc_scores_ewma.parquet'
RESULTS_DIR   = '/home/bpinoli/sub-industry-correlation/src/signal/full_results_naics_EWMA'

# ── Distributed lag regression on precomputed PC scores ───

def _run_lag_regression(
    cusip_scores: pl.DataFrame,
    max_lag: int = MAX_LAG,
    reg_window: int = REG_WINDOW,
) -> pl.DataFrame:
    """
    Given a single stock's time series of (date, pc1, y), build lagged
    PC1 features and run rolling 252-day OLS to predict next-day idio return.
    """
    cusip_scores = cusip_scores.sort('date')
    dates = cusip_scores['date'].to_list()
    pc1 = cusip_scores['pc1'].to_numpy().astype(np.float64)
    y = cusip_scores['y'].to_numpy().astype(np.float64)

    n = len(dates)
    if n < max_lag + 30:
        return pl.DataFrame({'date': [], 'cusip': [], 'signal': []})

    # Build lagged PC1 matrix: column k = pc1 shifted by k+1 days
    lagged = np.full((n, max_lag), np.nan)
    for lag in range(1, max_lag + 1):
        lagged[lag:, lag - 1] = pc1[:-lag]

    cusip_val = cusip_scores['cusip'][0]
    results_date, results_signal = [], []

    min_train = max_lag + 10

    for t in range(min_train, n - 1):
        lag_row_t = lagged[t]
        if np.isnan(lag_row_t).any():
            continue

        # Rolling 252-day window for training
        window_start = max(0, t - reg_window)
        train_mask = ~np.isnan(lagged[window_start:t]).any(axis=1) & ~np.isnan(y[window_start:t])
        train_idx = np.where(train_mask)[0] + window_start

        if len(train_idx) < min_train:
            continue

        X_train = np.column_stack([np.ones(len(train_idx)), lagged[train_idx]])
        y_train = y[train_idx]

        beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

        # Predict t+1 using lags known at time t
        y_pred = beta[0] + lag_row_t @ beta[1:]

        if np.isnan(y[t + 1]):
            continue

        results_date.append(dates[t + 1])
        results_signal.append(y_pred)

    return pl.DataFrame({
        'date':   results_date,
        'cusip':  [cusip_val] * len(results_date),
        'signal': results_signal,
    })


# ── Graphs and tables ─────────────────────────────────────

def save_classification_metrics(result: pl.DataFrame, results_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_score, recall_score, f1_score

    print("\nGenerating classification metrics...")

    clf_df = (
        result
        .filter(
            pl.col('signal').is_not_null()
            & pl.col(RETURN_COL).is_not_null()
        )
        .with_columns([
            (pl.col('signal') > 0).cast(pl.Int32).alias('pred_up'),
            (pl.col(RETURN_COL) > 0).cast(pl.Int32).alias('actual_up'),
        ])
    )

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

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(roll_dates, roll_prec, label='Precision', alpha=0.7, linewidth=0.8)
    ax.plot(roll_dates, roll_rec, label='Recall', alpha=0.7, linewidth=0.8)
    ax.plot(roll_dates, roll_f1, label='F1 Score', alpha=0.9, linewidth=1.2, color='black')
    ax.axhline(0.5, ls='--', color='grey', lw=0.8, label='Random baseline')
    ax.set_xlabel('Date')
    ax.set_ylabel('Score')
    ax.set_title('Rolling 21-Day Cross-Sectional Classification Metrics (NAICS, EWMA PCA)')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(results_dir, 'classification_rolling.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved rolling classification plot to {path}")

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

    sig_vals = sig_filled['signal'].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(sig_vals, bins=200, alpha=0.7, color='steelblue', edgecolor='none')
    ax.axvline(0, ls='--', color='red', lw=1)
    ax.set_xlabel('Signal Value (Predicted Idio Return)')
    ax.set_ylabel('Frequency')
    ax.set_title('Signal Distribution (NAICS, EWMA PCA)')
    ax.set_xlim(np.percentile(sig_vals, 1), np.percentile(sig_vals, 99))
    plt.tight_layout()
    path = os.path.join(results_dir, 'signal_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved signal distribution plot to {path}")


# ── Main ───────────────────────────────────────────────────

def main():
    load_dotenv()
    project_root = os.getcwd()
    output_path = os.getenv("SIGNAL_PATH", "data/signal.parquet")
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    t0 = time.time()

    # ── 1. Load precomputed EWMA PC scores ────────────────────
    print(f"Loading EWMA PC scores from {PC_SCORES_PATH}...")
    pc_df = pl.read_parquet(PC_SCORES_PATH)
    print(f"  {pc_df.height} rows, {pc_df['cusip'].n_unique()} cusips")

    # ── 2. Run lag regressions per stock ──────────────────────
    cusips = pc_df['cusip'].unique().to_list()
    total = len(cusips)
    print(f"\nRunning distributed lag regression for {total} stocks...")
    print(f"  Lag length: {MAX_LAG} days, OLS window: {REG_WINDOW} days")

    all_signals = []
    for i, cusip in enumerate(cusips):
        cusip_data = pc_df.filter(pl.col('cusip') == cusip)
        try:
            sig_df = _run_lag_regression(cusip_data)
            if sig_df.height > 0:
                all_signals.append(sig_df)
        except Exception as e:
            print(f"  ⚠ {cusip}: {e}")

        if (i + 1) % 500 == 0 or (i + 1) == total:
            elapsed = time.time() - t0
            pct = (i + 1) / total * 100
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"  [{i+1}/{total}] {pct:.1f}%, "
                  f"elapsed {elapsed/60:.0f}m, ETA {eta/60:.0f}m")

    signals_df = pl.concat(all_signals)
    print(f"\nSignal generated: {signals_df.height} rows, "
          f"{signals_df['cusip'].n_unique()} cusips")

    # ── 3. Load original data and join signal ─────────────────
    print("\nLoading original market data for join...")
    import sf_quant.data as sfd

    df = sfd.load_assets(
        start=dt.date(2000, 1, 1),
        end=dt.date(2025, 1, 1),
        columns=['ticker', 'date', 'barrid', 'cusip', 'price', 'return',
                 'specific_return', 'specific_risk'],
        in_universe=True,
    ).filter(
        pl.col('price').shift(1).over('barrid').gt(5)
    )

    # Merge NAICS
    naics_df = pl.read_csv('/home/bpinoli/sub-industry-correlation/data/NAICS_00-26.csv')
    naics_df = naics_df.with_columns(pl.col('datadate').str.to_date('%Y-%m-%d'))
    naics_df = naics_df.select(['cusip', 'datadate', 'naics'])

    merged_df = df.join_asof(
        other=naics_df, left_on='date', right_on='datadate',
        by='cusip', strategy='backward',
    ).filter(pl.col(INDUSTRY_COL).is_not_null())

    # Join signal
    result = merged_df.join(
        signals_df,
        on=[DATE_COL, ID_COL],
        how='left',
    )

    # ── 4. Save signal parquet ────────────────────────────────
    result.write_parquet(output_path)
    print(f"Saved signal to {output_path}")

    # ── 5. Classification metrics ─────────────────────────────
    save_classification_metrics(result, RESULTS_DIR)

    # ── 6. Signal diagnostics ─────────────────────────────────
    save_signal_diagnostics(result, RESULTS_DIR)

    # ── Done ──────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PASS 2 COMPLETE — Signal Generation (NAICS, EWMA PCA, Distributed Lag)")
    print(f"  Signal:       {output_path}")
    print(f"  Results:      {RESULTS_DIR}/")
    print(f"  Total time:   {total_time/60:.0f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
