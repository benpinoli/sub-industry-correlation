"""
Pass 1: Extract daily PC1 scores using EWMA covariance.

The exponentially weighted covariance matrix ensures the eigenvector
evolves smoothly day-to-day, making it valid to lag PC scores across
days in Pass 2.

Half-life = 63 trading days (~3 months).

Saves: full_results_naics/pc_scores_ewma.parquet
Columns: date, cusip, pc1, y, n_peers, naics
"""
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
HALFLIFE      = 63    # EWMA half-life in trading days (~3 months)
INDUSTRY_COL  = 'naics'
RETURN_COL    = 'specific_return'
DATE_COL      = 'date'
ID_COL        = 'cusip'
MIN_PEERS     = 2

RESULTS_DIR   = '/home/bpinoli/sub-industry-correlation/src/signal/full_results_naics_EWMA'
NAICS_NAMES_PATH = '/home/bpinoli/sub-industry-correlation/data/2022_NAICS_Structure.xlsx'


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


def get_naics_label(code, naics_names, max_len=40):
    code_str = str(code)
    name = naics_names.get(code_str, "Unknown")
    if len(name) > max_len:
        name = name[:max_len - 1] + "…"
    return f"{code_str} - {name}"


def _extract_pc_scores_ewma(
    target_cusip: str,
    peer_cusips: list[str],
    returns_wide: pl.DataFrame,
    dates: np.ndarray,
    window: int = WINDOW,
    halflife: int = HALFLIFE,
) -> pl.DataFrame:
    """
    For each day, compute EWMA covariance of peer returns, eigendecompose,
    and project to get PC1 score. EWMA ensures eigenvector stability across days.
    """
    target_arr = returns_wide[target_cusip].to_numpy().astype(np.float64)
    peer_arr = returns_wide.select(peer_cusips).to_numpy().astype(np.float64)

    # Precompute exponential weights (newest = highest weight)
    decay = np.exp(-np.log(2) / halflife)
    raw_weights = decay ** np.arange(window - 1, -1, -1)  # length = window
    raw_weights /= raw_weights.sum()

    out_date, out_pc1, out_y, out_n_peers = [], [], [], []

    # Track previous eigenvector to fix sign flips
    prev_vec = None

    for t in range(window, len(dates)):
        start = t - window
        end = t

        X_block = peer_arr[start:end]  # (window, n_peers)

        # Drop peers with any NaN in this window
        valid_mask = ~np.isnan(X_block).any(axis=0)
        X_valid = X_block[:, valid_mask]
        n_peers = X_valid.shape[1]

        if n_peers < MIN_PEERS:
            prev_vec = None  # reset sign tracking
            continue

        # ── EWMA covariance ───────────────────────────────────
        # Weighted mean
        w = raw_weights  # already sums to 1
        w_mean = (w[:, None] * X_valid).sum(axis=0)
        X_centered = X_valid - w_mean

        # Weighted covariance: C = X_centered.T @ diag(w) @ X_centered
        cov_ewma = (X_centered * w[:, None]).T @ X_centered

        # Eigendecompose (eigh returns ascending order)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_ewma)
        # Flip to descending
        idx = np.argsort(eigenvalues)[::-1]
        top_vec = eigenvectors[:, idx[0]]

        # Fix sign flips: ensure continuity with previous eigenvector
        if prev_vec is not None and len(prev_vec) == len(top_vec):
            if np.dot(top_vec, prev_vec) < 0:
                top_vec = -top_vec
        prev_vec = top_vec.copy()

        # Project latest day's centered data onto top eigenvector
        pc1_score = X_centered[-1] @ top_vec

        y_val = target_arr[t] if t < len(target_arr) else np.nan

        out_date.append(dates[t].item())
        out_pc1.append(pc1_score)
        out_y.append(y_val)
        out_n_peers.append(n_peers)

    return pl.DataFrame({
        'date':     out_date,
        'cusip':    [target_cusip] * len(out_date),
        'pc1':      out_pc1,
        'y':        out_y,
        'n_peers':  out_n_peers,
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
        start=start, end=end, columns=columns, in_universe=True,
    ).filter(
        pl.col('price').shift(1).over('barrid').gt(5)
    )

    return df


def merge_industry_data(df: pl.DataFrame) -> pl.DataFrame:
    naics_df = pl.read_csv('/home/bpinoli/sub-industry-correlation/data/NAICS_00-26.csv')
    naics_df = naics_df.with_columns(pl.col('datadate').str.to_date('%Y-%m-%d'))
    naics_df = naics_df.select(['cusip', 'datadate', 'naics'])

    merged_df = df.join_asof(
        other=naics_df, left_on='date', right_on='datadate',
        by='cusip', strategy='backward',
    )
    return merged_df


# ── Main ───────────────────────────────────────────────────

def main():
    load_dotenv()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    t0 = time.time()

    print("Loading NAICS industry names...")
    naics_names = load_naics_names()

    print("Loading market data...")
    df = load_data()
    print(f"  Loaded {df.height} rows in {time.time() - t0:.0f}s")

    print("Merging NAICS classifications...")
    merged_df = merge_industry_data(df)
    before = merged_df.height
    merged_df = merged_df.filter(pl.col(INDUSTRY_COL).is_not_null())
    print(f"  Dropped {before - merged_df.height} rows with no NAICS match")

    print("Preprocessing...")
    merged_df = merged_df.with_columns(pl.col(DATE_COL).cast(pl.Date))
    merged_df = merged_df.sort([ID_COL, DATE_COL])

    returns_wide = (
        merged_df
        .pivot(on=ID_COL, index=DATE_COL, values=RETURN_COL, aggregate_function='first')
        .sort(DATE_COL)
    )

    id_cols = [c for c in returns_wide.columns if c != DATE_COL]

    returns_wide = returns_wide.filter(
        pl.sum_horizontal([pl.col(c).is_not_null().cast(pl.Int32) for c in id_cols]) >= 20
    )

    dates = returns_wide[DATE_COL].to_numpy()
    print(f"  Universe: {len(id_cols)} cusips, {returns_wide.height} trading days")

    # Build cusip -> naics lookup
    cusip_naics = (
        merged_df.sort(DATE_COL).group_by(ID_COL).last()
        .select([ID_COL, INDUSTRY_COL])
    )
    industry_map = dict(
        zip(cusip_naics[ID_COL].to_list(), cusip_naics[INDUSTRY_COL].to_list())
    )

    # Group by NAICS
    industry_groups: dict[str, list[str]] = defaultdict(list)
    for c in id_cols:
        ind = industry_map.get(c)
        if ind is not None:
            industry_groups[ind].append(c)

    eligible = [(ind, v) for ind, v in industry_groups.items() if len(v) >= MIN_PEERS + 1]
    total_stocks = sum(len(v) for _, v in eligible)
    print(f"  NAICS codes with >= {MIN_PEERS + 1} stocks: {len(eligible)}")
    print(f"  Total stocks to process: {total_stocks}")
    print(f"  EWMA half-life: {HALFLIFE} trading days")

    # ── Extract PC scores ─────────────────────────────────────
    print(f"\nStarting EWMA PC score extraction at {dt.datetime.now().strftime('%H:%M:%S')}...")
    all_scores = []
    stocks_done = 0

    for i, (ind, available) in enumerate(eligible):
        for target in available:
            peers = [c for c in available if c != target]
            try:
                scores_df = _extract_pc_scores_ewma(target, peers, returns_wide, dates)
                if scores_df.height > 0:
                    scores_df = scores_df.with_columns(pl.lit(ind).alias('naics'))
                    all_scores.append(scores_df)
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

    all_scores_df = pl.concat(all_scores)
    output_path = os.path.join(RESULTS_DIR, 'pc_scores_ewma.parquet')
    all_scores_df.write_parquet(output_path)

    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"PASS 1 COMPLETE — EWMA PC Score Extraction")
    print(f"  Output:       {output_path}")
    print(f"  Rows:         {all_scores_df.height}")
    print(f"  CUSIPs:       {all_scores_df['cusip'].n_unique()}")
    print(f"  Half-life:    {HALFLIFE} days")
    print(f"  Total time:   {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
