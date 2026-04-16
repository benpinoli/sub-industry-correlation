"""
Pass 2 (Monthly Prediction, Overlapping Targets):
Predict next-month (21-day) cumulative idiosyncratic return using
4 weekly-averaged PC1 lags. Train on all daily observations.

Overlapping targets = more training data but serially correlated residuals.

Expects: full_results_naics_EWMA/pc_scores_ewma.parquet
Saves:   signal + diagnostics to full_results_monthly_overlapping/
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
N_WEEKLY_LAGS  = 4
DAYS_PER_WEEK  = 5
FORWARD_DAYS   = 21    # predict cumulative return over next 21 days
REG_WINDOW     = 252
DATE_COL       = 'date'
ID_COL         = 'cusip'
RETURN_COL     = 'specific_return'
INDUSTRY_COL   = 'naics'

PC_SCORES_PATH = '/home/bpinoli/sub-industry-correlation/src/signal/naics_strats/EWMA_2_pass/full_results_naics_EWMA/pc_scores_ewma.parquet'
RESULTS_DIR    = '/home/bpinoli/sub-industry-correlation/src/signal/full_results_monthly_overlapping'


def _run_monthly_overlapping(
    cusip_scores: pl.DataFrame,
    n_weekly_lags: int = N_WEEKLY_LAGS,
    days_per_week: int = DAYS_PER_WEEK,
    forward_days: int = FORWARD_DAYS,
    reg_window: int = REG_WINDOW,
) -> pl.DataFrame:
    cusip_scores = cusip_scores.sort('date')
    dates = cusip_scores['date'].to_list()
    pc1 = cusip_scores['pc1'].to_numpy().astype(np.float64)
    y = cusip_scores['y'].to_numpy().astype(np.float64)

    n = len(dates)
    total_lookback = n_weekly_lags * days_per_week  # 20

    if n < total_lookback + forward_days + 30:
        return pl.DataFrame({'date': [], 'cusip': [], 'signal': []})

    # ── Build forward cumulative return ───────────────────────
    y_monthly = np.full(n, np.nan)
    for t in range(n - forward_days):
        block = y[t + 1:t + 1 + forward_days]
        if not np.isnan(block).any():
            y_monthly[t] = block.sum()

    # ── Build weekly-averaged lag matrix ──────────────────────
    lagged_weekly = np.full((n, n_weekly_lags), np.nan)
    for t in range(total_lookback, n):
        for wk in range(n_weekly_lags):
            start_lag = wk * days_per_week + 1
            end_lag = (wk + 1) * days_per_week
            idx_start = t - end_lag
            idx_end = t - start_lag + 1
            if idx_start < 0:
                break
            week_slice = pc1[idx_start:idx_end]
            if np.isnan(week_slice).any():
                break
            lagged_weekly[t, wk] = np.mean(week_slice)

    cusip_val = cusip_scores['cusip'][0]
    results_date, results_signal = [], []
    min_train = n_weekly_lags + 10

    for t in range(total_lookback, n - forward_days):
        lag_row_t = lagged_weekly[t]
        if np.isnan(lag_row_t).any():
            continue

        # Rolling window training on overlapping targets
        window_start = max(0, t - reg_window)
        valid_mask = (~np.isnan(lagged_weekly[window_start:t]).any(axis=1)
                      & ~np.isnan(y_monthly[window_start:t]))
        train_idx = np.where(valid_mask)[0] + window_start

        if len(train_idx) < min_train:
            continue

        X_train = np.column_stack([np.ones(len(train_idx)), lagged_weekly[train_idx]])
        y_train = y_monthly[train_idx]

        beta, _, _, _ = np.linalg.lstsq(X_train, y_train, rcond=None)

        y_pred = beta[0] + lag_row_t @ beta[1:]

        if np.isnan(y_monthly[t]):
            continue

        results_date.append(dates[t])
        results_signal.append(y_pred)

    return pl.DataFrame({
        'date':   results_date,
        'cusip':  [cusip_val] * len(results_date),
        'signal': results_signal,
    })


# ── Diagnostics ───────────────────────────────────────────

def save_diagnostics(signals_df: pl.DataFrame, pc_df: pl.DataFrame, results_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr

    print("\nGenerating diagnostics...")

    # Build monthly forward return for evaluation
    pc_sorted = pc_df.sort([ID_COL, DATE_COL])

    # Join signal with y_monthly
    # Recompute y_monthly from pc_df
    eval_frames = []
    for cusip in signals_df['cusip'].unique().to_list():
        cdf = pc_sorted.filter(pl.col('cusip') == cusip).sort('date')
        y_arr = cdf['y'].to_numpy().astype(np.float64)
        dates_arr = cdf['date'].to_list()
        n = len(y_arr)
        for t in range(n - FORWARD_DAYS):
            block = y_arr[t + 1:t + 1 + FORWARD_DAYS]
            if not np.isnan(block).any():
                eval_frames.append({
                    'date': dates_arr[t],
                    'cusip': cusip,
                    'y_monthly': block.sum(),
                })

    eval_df = pl.DataFrame(eval_frames)
    merged = signals_df.join(eval_df, on=['date', 'cusip'], how='inner')

    # ── Daily cross-sectional rank IC ─────────────────────────
    daily_ic_records = []
    for date in merged['date'].unique().sort().to_list():
        day = merged.filter(pl.col('date') == date)
        if day.height < 10:
            continue
        ic, _ = spearmanr(day['signal'].to_numpy(), day['y_monthly'].to_numpy())
        if not np.isnan(ic):
            daily_ic_records.append({'date': date, 'rank_ic': ic})

    daily_ic = pl.DataFrame(daily_ic_records).sort('date')
    ic_arr = daily_ic['rank_ic'].to_numpy()
    ic_dates = daily_ic['date'].to_list()

    print(f"  Days with valid IC: {len(ic_arr)}")
    print(f"  Mean daily IC:      {ic_arr.mean():.4f}")
    print(f"  Median daily IC:    {np.median(ic_arr):.4f}")
    print(f"  Std daily IC:       {ic_arr.std():.4f}")
    print(f"  IC > 0 fraction:    {(ic_arr > 0).mean():.3f}")
    print(f"  IC IR (mean/std):   {ic_arr.mean() / ic_arr.std():.4f}")

    # Save IC summary
    ic_summary = pl.DataFrame({
        'Metric': ['Mean IC', 'Median IC', 'Std IC', 'IC > 0 %', 'IC IR', 'N days'],
        'Value': [
            f"{ic_arr.mean():.6f}",
            f"{np.median(ic_arr):.6f}",
            f"{ic_arr.std():.6f}",
            f"{(ic_arr > 0).mean():.3f}",
            f"{ic_arr.mean() / ic_arr.std():.6f}",
            str(len(ic_arr)),
        ],
    })
    ic_summary.write_csv(os.path.join(results_dir, 'rank_ic_summary.csv'))

    # ── Cumulative IC plot ────────────────────────────────────
    cumul_ic = np.cumsum(ic_arr)
    roll = 21
    roll_ic = np.convolve(ic_arr, np.ones(roll)/roll, mode='valid')
    roll_dates = ic_dates[roll-1:]

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axes[0].bar(ic_dates, ic_arr, width=1, alpha=0.3, color='steelblue')
    axes[0].axhline(0, color='black', lw=0.5)
    axes[0].axhline(ic_arr.mean(), color='red', ls='--', lw=1, label=f'Mean = {ic_arr.mean():.4f}')
    axes[0].set_ylabel('Daily Rank IC')
    axes[0].set_title('Daily Cross-Sectional Rank IC — Monthly Overlapping')
    axes[0].legend()

    axes[1].plot(roll_dates, roll_ic, linewidth=0.8, color='steelblue')
    axes[1].axhline(0, color='black', lw=0.5)
    axes[1].set_ylabel('Rolling 21-Day IC')

    axes[2].plot(ic_dates, cumul_ic, linewidth=1, color='darkblue')
    axes[2].axhline(0, color='black', lw=0.5)
    axes[2].set_ylabel('Cumulative IC')
    axes[2].set_xlabel('Date')

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'rank_ic_plots.png'), dpi=150)
    plt.close()
    print(f"  Saved IC plots")

    # ── Classification metrics ────────────────────────────────
    from sklearn.metrics import precision_score, recall_score, f1_score

    pred_up = (merged['signal'].to_numpy() > 0).astype(int)
    actual_up = (merged['y_monthly'].to_numpy() > 0).astype(int)

    clf_summary = pl.DataFrame({
        'Metric': ['Precision', 'Recall', 'F1 Score', 'Accuracy', 'N'],
        'Value': [
            precision_score(actual_up, pred_up, zero_division=0),
            recall_score(actual_up, pred_up, zero_division=0),
            f1_score(actual_up, pred_up, zero_division=0),
            (pred_up == actual_up).mean(),
            float(len(pred_up)),
        ],
    })
    clf_summary.write_csv(os.path.join(results_dir, 'classification_summary.csv'))
    print(f"  Saved classification summary")
    print(clf_summary)

    # ── Signal distribution ───────────────────────────────────
    sig_vals = signals_df['signal'].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5))
    p1, p99 = np.percentile(sig_vals, [1, 99])
    clipped = sig_vals[(sig_vals >= p1) & (sig_vals <= p99)]
    ax.hist(clipped, bins=200, alpha=0.7, color='steelblue', edgecolor='none')
    ax.axvline(0, ls='--', color='red', lw=1)
    ax.set_xlabel('Signal Value (Predicted Monthly Idio Return)')
    ax.set_ylabel('Frequency')
    ax.set_title('Signal Distribution — Monthly Overlapping')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'signal_distribution.png'), dpi=150)
    plt.close()
    print(f"  Saved signal distribution")


# ── Main ───────────────────────────────────────────────────

def main():
    load_dotenv()
    project_root = os.getcwd()
    output_path = os.path.join(project_root, 'data/signal_monthly_overlapping.parquet')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t0 = time.time()

    print(f"Loading EWMA PC scores from {PC_SCORES_PATH}...")
    pc_df = pl.read_parquet(PC_SCORES_PATH)
    print(f"  {pc_df.height} rows, {pc_df['cusip'].n_unique()} cusips")

    cusips = pc_df['cusip'].unique().to_list()
    total = len(cusips)
    print(f"\nRunning monthly overlapping lag regression for {total} stocks...")
    print(f"  {N_WEEKLY_LAGS} weekly lags, {FORWARD_DAYS}-day forward return")
    print(f"  {N_WEEKLY_LAGS + 1} parameters, OLS window: {REG_WINDOW} days")

    all_signals = []
    for i, cusip in enumerate(cusips):
        cusip_data = pc_df.filter(pl.col('cusip') == cusip)
        try:
            sig_df = _run_monthly_overlapping(cusip_data)
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

    signals_df.write_parquet(output_path)
    print(f"Saved signal to {output_path}")

    save_diagnostics(signals_df, pc_df, RESULTS_DIR)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE — Monthly Overlapping Signal")
    print(f"  Signal:   {output_path}")
    print(f"  Results:  {RESULTS_DIR}/")
    print(f"  Time:     {total_time/60:.0f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
