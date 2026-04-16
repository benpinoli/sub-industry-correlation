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
INDUSTRY_COL  = 'naics'
RETURN_COL    = 'specific_return'
DATE_COL      = 'date'
ID_COL        = 'cusip'
MIN_PEERS     = 2     # minimum peers (3 stocks total = target + 2 peers)

RESULTS_DIR   = '/home/bpinoli/sub-industry-correlation/src/signal/full_results_naics'
NAICS_NAMES_PATH = '/home/bpinoli/sub-industry-correlation/data/2022_NAICS_Structure.xlsx'


# ── Load NAICS code -> name lookup ─────────────────────────

def load_naics_names() -> dict[str, str]:
    import openpyxl

    wb = openpyxl.load_workbook(NAICS_NAMES_PATH, read_only=True)
    ws = wb.active

    naics_names = {}
    for row in ws.iter_rows(min_row=1, values_only=True):
        code = row[1]
        title = row[2]
        if code is None or title is None:
            continue
        code_str = str(code).strip()
        title_str = str(title).strip().rstrip('T').rstrip('*').strip()
        if len(code_str) == 6 and code_str.isdigit():
            naics_names[code_str] = title_str

    wb.close()
    print(f"  Loaded {len(naics_names)} NAICS 6-digit industry names")
    return naics_names


def get_naics_label(code, naics_names: dict[str, str], max_len: int = 40) -> str:
    code_str = str(code)
    name = naics_names.get(code_str, "Unknown")
    if len(name) > max_len:
        name = name[:max_len - 1] + "…"
    return f"{code_str} - {name}"


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
) -> pl.DataFrame:
    """
    Rolling-window PCA distributed-lag forecast.
    Refits PCA + regression every day. Uses top 1 PC by default.
    """
    target_arr = returns_wide[target_cusip].to_numpy().astype(np.float64)
    peer_arr = returns_wide.select(peer_cusips).to_numpy().astype(np.float64)

    total_lookback = window + max_lag
    results_date, results_signal = [], []

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

    return pl.DataFrame({
        'date':   results_date,
        'cusip':  [target_cusip] * len(results_date),
        'signal': results_signal,
    })


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
    naics_df = pl.read_csv('/home/bpinoli/sub-industry-correlation/data/NAICS_00-26.csv')
    naics_df = naics_df.with_columns(
        pl.col('datadate').str.to_date('%Y-%m-%d')
    )

    naics_df = naics_df.select(['cusip', 'datadate', 'naics'])

    merged_df = df.join_asof(
        other=naics_df,
        left_on='date',
        right_on='datadate',
        by='cusip',
        strategy='backward',
    )

    return merged_df


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
            & pl.col('specific_return').is_not_null()
        )
        .with_columns([
            (pl.col('signal') > 0).cast(pl.Int32).alias('pred_up'),
            (pl.col('specific_return') > 0).cast(pl.Int32).alias('actual_up'),
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
    ax.set_title('Rolling 21-Day Cross-Sectional Classification Metrics (NAICS)')
    ax.legend()
    plt.tight_layout()
    path = os.path.join(results_dir, 'classification_rolling.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved rolling classification plot to {path}")

    # Overall table 2010–2024
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
    ax.set_title('Signal Distribution (NAICS)')
    ax.set_xlim(np.percentile(sig_vals, 1), np.percentile(sig_vals, 99))
    plt.tight_layout()
    path = os.path.join(results_dir, 'signal_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved signal distribution plot to {path}")


# ── Main ───────────────────────────────────────────────────

def create_signal():
    load_dotenv()
    project_root = os.getcwd()
    output_path = os.getenv("SIGNAL_PATH", "data/signal.parquet")
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t0 = time.time()

    # ── 0. Load NAICS names ───────────────────────────────────
    print("Loading NAICS industry names...")
    naics_names = load_naics_names()

    # ── 1. Load data ──────────────────────────────────────────
    print("Loading market data...")
    df = load_data()
    print(f"  Loaded {df.height} rows in {time.time() - t0:.0f}s")

    print("Merging NAICS classifications...")
    merged_df = merge_industry_data(df)

    before = merged_df.height
    merged_df = merged_df.filter(pl.col(INDUSTRY_COL).is_not_null())
    print(f"  Dropped {before - merged_df.height} rows with no NAICS match")

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

    # ── 3. Build cusip -> naics lookup ────────────────────────
    cusip_naics = (
        merged_df.sort(DATE_COL)
        .group_by(ID_COL)
        .last()
        .select([ID_COL, INDUSTRY_COL])
    )
    industry_map: dict[str, str] = dict(
        zip(cusip_naics[ID_COL].to_list(), cusip_naics[INDUSTRY_COL].to_list())
    )

    # ── 4. Group cusips by NAICS ──────────────────────────────
    industry_groups: dict[str, list[str]] = defaultdict(list)
    for c in id_cols:
        ind = industry_map.get(c)
        if ind is not None:
            industry_groups[ind].append(c)

    eligible = [(ind, v) for ind, v in industry_groups.items() if len(v) >= MIN_PEERS + 1]
    total_stocks = sum(len(v) for _, v in eligible)
    print(f"  NAICS codes with >= {MIN_PEERS + 1} stocks: {len(eligible)}")
    print(f"  Total stocks to process: {total_stocks}")

    # ── 5. Run PCA distributed lag forecast ───────────────────
    print(f"\nStarting signal generation at {dt.datetime.now().strftime('%H:%M:%S')}...")
    all_signals = []
    stocks_done = 0

    for i, (ind, available) in enumerate(eligible):
        for target in available:
            peers = [c for c in available if c != target]
            try:
                sig_df = _forecast_stock_pca(target, peers, returns_wide, dates)
                if sig_df.height > 0:
                    all_signals.append(sig_df)
            except Exception as e:
                print(f"  ⚠ {target}: {e}")
            stocks_done += 1

        elapsed = time.time() - t0
        pct = stocks_done / total_stocks * 100
        eta = elapsed / stocks_done * (total_stocks - stocks_done) if stocks_done > 0 else 0
        ind_label = get_naics_label(ind, naics_names, max_len=30)
        print(f"  [{i+1}/{len(eligible)}] {ind_label} done "
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

    # ── 7. Classification metrics ─────────────────────────────
    save_classification_metrics(result, RESULTS_DIR)

    # ── 8. Signal diagnostics ─────────────────────────────────
    save_signal_diagnostics(result, RESULTS_DIR)

    # ── Done ──────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE — Signal Generation (NAICS, PC1, Distributed Lag)")
    print(f"  Signal:       {output_path}")
    print(f"  Results:      {RESULTS_DIR}/")
    print(f"  Total time:   {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    create_signal()
