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
N_COMPONENTS  = 3     # Track top 3 eigenvalues
INDUSTRY_COL  = 'naics'
RETURN_COL    = 'specific_return'
DATE_COL      = 'date'
ID_COL        = 'cusip'
MIN_PEERS     = 2     # minimum peers needed (3 stocks total = target + 2 peers)

RESULTS_DIR   = '/home/bpinoli/sub-industry-correlation/src/signal/full_results_naics'
NAICS_NAMES_PATH = '/home/bpinoli/sub-industry-correlation/data/2022_NAICS_Structure.xlsx'


# ── Load NAICS code -> name lookup ─────────────────────────

def load_naics_names() -> dict[str, str]:
    """Load 6-digit NAICS codes and their English titles from Census xlsx."""
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
    """Return 'CODE - Name' label, truncated if needed."""
    code_str = str(code)
    name = naics_names.get(code_str, "Unknown")
    if len(name) > max_len:
        name = name[:max_len - 1] + "…"
    return f"{code_str} - {name}"


# ── Core function: PCA eigenvalue extraction (top 3) ──────# changed to top 1 only 

def _extract_eigenvalues(
    target_cusip: str,
    peer_cusips: list[str],
    returns_wide: pl.DataFrame,
    dates: np.ndarray,
    window: int = WINDOW,
    n_components: int = N_COMPONENTS,
) -> pl.DataFrame:
    peer_arr = returns_wide.select(peer_cusips).to_numpy().astype(np.float64)

    eig_date = []
    eig_vals = {i: [] for i in range(n_components)}
    eig_var_expl = {i: [] for i in range(n_components)}
    eig_cumul_var = []
    eig_n_peers = []

    for t in range(window, len(dates)):
        start = t - window
        end   = t

        X_block = peer_arr[start:end]

        valid_mask = ~np.isnan(X_block).any(axis=0)
        X_valid = X_block[:, valid_mask]
        n_peers = X_valid.shape[1]

        if n_peers < MIN_PEERS:
            continue

        X_mean = X_valid.mean(axis=0)
        X_std = X_valid.std(axis=0)
        X_std[X_std == 0] = 1.0
        X_scaled = (X_valid - X_mean) / X_std

        S = np.linalg.svd(X_scaled, compute_uv=False)

        total_var = (S ** 2).sum()
        m = min(n_components, len(S))

        eig_date.append(dates[t].item())
        eig_n_peers.append(n_peers)

        cumul = 0.0
        for i in range(n_components):
            if i < m and total_var > 0:
                eigenvalue = S[i] ** 2 / (window - 1)
                var_expl = S[i] ** 2 / total_var
            else:
                eigenvalue = 0.0
                var_expl = 0.0
            eig_vals[i].append(eigenvalue)
            eig_var_expl[i].append(var_expl)
            cumul += var_expl

        eig_cumul_var.append(cumul)

    data = {
        'date':                eig_date,
        'cusip':               [target_cusip] * len(eig_date),
        'n_peers':             eig_n_peers,
        'cumul_var_explained': eig_cumul_var,
    }
    for i in range(n_components):
        data[f'eigenvalue_{i+1}'] = eig_vals[i]
        data[f'var_explained_{i+1}'] = eig_var_expl[i]

    return pl.DataFrame(data)


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

def save_eigenvalue_analysis(all_eig_df: pl.DataFrame, naics_names: dict, results_dir: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("\nGenerating eigenvalue analysis...")

    # ── 1. Overall summary table ──────────────────────────────
    agg_exprs = [pl.len().alias('n_observations'), pl.col('n_peers').mean().alias('mean_peers')]
    for i in range(1, N_COMPONENTS + 1):
        col = f'eigenvalue_{i}'
        ve_col = f'var_explained_{i}'
        agg_exprs.extend([
            pl.col(col).mean().alias(f'mean_eigenvalue_{i}'),
            pl.col(col).median().alias(f'median_eigenvalue_{i}'),
            pl.col(ve_col).mean().alias(f'mean_var_explained_{i}'),
            pl.col(ve_col).median().alias(f'median_var_explained_{i}'),
        ])
    agg_exprs.extend([
        pl.col('cumul_var_explained').mean().alias('mean_cumul_var_explained'),
        pl.col('cumul_var_explained').median().alias('median_cumul_var_explained'),
    ])

    summary = all_eig_df.select(agg_exprs)

    summary_path = os.path.join(results_dir, 'eigenvalue_summary.csv')
    summary.write_csv(summary_path)
    print(f"  Saved eigenvalue summary to {summary_path}")
    print(summary)

    # ── 2. Per-NAICS summary ─────────────────────────────────
    per_naics_aggs = [
        pl.col('n_peers').mean().alias('mean_peers'),
        pl.len().alias('n_obs'),
    ]
    for i in range(1, N_COMPONENTS + 1):
        per_naics_aggs.extend([
            pl.col(f'eigenvalue_{i}').mean().alias(f'mean_eigenvalue_{i}'),
            pl.col(f'var_explained_{i}').mean().alias(f'mean_var_explained_{i}'),
        ])
    per_naics_aggs.append(
        pl.col('cumul_var_explained').mean().alias('mean_cumul_var_explained')
    )

    per_naics = (
        all_eig_df
        .group_by('naics')
        .agg(per_naics_aggs)
        .sort('mean_cumul_var_explained', descending=True)
    )

    # Add English names
    per_naics = per_naics.with_columns(
        pl.col('naics').map_elements(
            lambda x: naics_names.get(str(x), "Unknown"), return_dtype=pl.Utf8
        ).alias('naics_name')
    )

    per_naics_path = os.path.join(results_dir, 'eigenvalue_by_naics.csv')
    per_naics.write_csv(per_naics_path)
    print(f"  Saved per-NAICS eigenvalue table to {per_naics_path}")
    print("\n  Top 15 NAICS codes by mean cumulative variance explained (top 3 PCs):")
    print(per_naics.select(['naics', 'naics_name', 'mean_cumul_var_explained',
                            'mean_var_explained_1', 'mean_var_explained_2',
                            'mean_var_explained_3', 'mean_peers', 'n_obs']).head(15))

    # ── 3. Time series: daily average eigenvalues ─────────────
    daily_avg = (
        all_eig_df
        .group_by('date')
        .agg([
            pl.col('eigenvalue_1').mean().alias('avg_eigenvalue_1'),
            pl.col('eigenvalue_2').mean().alias('avg_eigenvalue_2'),
            pl.col('eigenvalue_3').mean().alias('avg_eigenvalue_3'),
            pl.col('var_explained_1').mean().alias('avg_var_explained_1'),
            pl.col('var_explained_2').mean().alias('avg_var_explained_2'),
            pl.col('var_explained_3').mean().alias('avg_var_explained_3'),
            pl.col('cumul_var_explained').mean().alias('avg_cumul_var_explained'),
        ])
        .sort('date')
    )

    dates_plot = daily_avg['date'].to_list()

    # Plot 1: Eigenvalue magnitudes
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates_plot, daily_avg['avg_eigenvalue_1'].to_numpy(),
            linewidth=0.6, alpha=0.8, label='PC1', color='steelblue')
    ax.plot(dates_plot, daily_avg['avg_eigenvalue_2'].to_numpy(),
            linewidth=0.6, alpha=0.8, label='PC2', color='darkorange')
    ax.plot(dates_plot, daily_avg['avg_eigenvalue_3'].to_numpy(),
            linewidth=0.6, alpha=0.8, label='PC3', color='forestgreen')
    ax.set_ylabel('Avg Eigenvalue')
    ax.set_xlabel('Date')
    ax.set_title('Daily Average Top 3 PCA Eigenvalues — NAICS Peer Groups')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, 'eigenvalue_timeseries.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved eigenvalue time series plot to {path}")

    # Plot 2: Variance explained (stacked area)
    fig, ax = plt.subplots(figsize=(14, 5))
    ve1 = daily_avg['avg_var_explained_1'].to_numpy()
    ve2 = daily_avg['avg_var_explained_2'].to_numpy()
    ve3 = daily_avg['avg_var_explained_3'].to_numpy()
    ax.fill_between(dates_plot, 0, ve1, alpha=0.6, label='PC1', color='steelblue')
    ax.fill_between(dates_plot, ve1, ve1 + ve2, alpha=0.6, label='PC2', color='darkorange')
    ax.fill_between(dates_plot, ve1 + ve2, ve1 + ve2 + ve3, alpha=0.6, label='PC3', color='forestgreen')
    ax.set_ylabel('Variance Explained')
    ax.set_xlabel('Date')
    ax.set_title('Daily Average Variance Explained by Top 3 PCs — NAICS Peer Groups')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(results_dir, 'variance_explained_stacked.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved stacked variance explained plot to {path}")

    # ── 4. Per-NAICS time series (top 9 by cumul var explained)
    top_naics = per_naics.head(9)['naics'].to_list()

    fig, axes = plt.subplots(3, 3, figsize=(20, 14), sharex=True)
    axes_flat = axes.flatten()

    for idx, naics_code in enumerate(top_naics):
        ax = axes_flat[idx]
        sub_data = (
            all_eig_df
            .filter(pl.col('naics') == naics_code)
            .group_by('date')
            .agg([
                pl.col('var_explained_1').mean().alias('ve1'),
                pl.col('var_explained_2').mean().alias('ve2'),
                pl.col('var_explained_3').mean().alias('ve3'),
            ])
            .sort('date')
        )
        d = sub_data['date'].to_list()
        v1 = sub_data['ve1'].to_numpy()
        v2 = sub_data['ve2'].to_numpy()
        v3 = sub_data['ve3'].to_numpy()

        ax.fill_between(d, 0, v1, alpha=0.6, color='steelblue')
        ax.fill_between(d, v1, v1 + v2, alpha=0.6, color='darkorange')
        ax.fill_between(d, v1 + v2, v1 + v2 + v3, alpha=0.6, color='forestgreen')

        label = get_naics_label(naics_code, naics_names, max_len=35)
        ax.set_title(label, fontsize=8)
        ax.set_ylabel('Var Expl.', fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=6)

    plt.suptitle('Variance Explained by Top 3 PCs — Top 9 NAICS Codes', fontsize=12)
    plt.tight_layout()
    path = os.path.join(results_dir, 'eigenvalue_by_naics.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved per-NAICS eigenvalue plot to {path}")


# ── Main ───────────────────────────────────────────────────

def create_signal():
    load_dotenv()
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

    # ── 5. Run PCA eigenvalue extraction ──────────────────────
    print(f"\nStarting PCA eigenvalue extraction at {dt.datetime.now().strftime('%H:%M:%S')}...")
    all_eigenvalues = []
    stocks_done = 0
    for i, (ind, available) in enumerate(eligible):
        # Adaptive: 3 components if >= 6 stocks, else 1
        n_comps = 3 if len(available) >= 6 else 1
        # n_comps = 1 #this is a change from 3 to 1 that we need
        for target in available:
            peers = [c for c in available if c != target]
            try:
                eig_df = _extract_eigenvalues(target, peers, returns_wide, dates,
                                            n_components=n_comps)
                if eig_df.height > 0:
                    eig_df = eig_df.with_columns(pl.lit(ind).alias('naics'))
                    all_eigenvalues.append(eig_df)
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

    # ── 6. Save eigenvalue data ───────────────────────────────
    all_eig_df = pl.concat(all_eigenvalues)
    eig_parquet_path = os.path.join(RESULTS_DIR, 'eigenvalues_naics.parquet')
    all_eig_df.write_parquet(eig_parquet_path)
    print(f"\nSaved eigenvalue data to {eig_parquet_path}")
    print(f"  {all_eig_df.height} rows, {all_eig_df['cusip'].n_unique()} cusips")

    # ── 7. Graphs and tables ──────────────────────────────────
    save_eigenvalue_analysis(all_eig_df, naics_names, RESULTS_DIR)

    # ── Done ──────────────────────────────────────────────────
    total_time = time.time() - t0
    print(f"\n{'='*60}")
    print(f"COMPLETE — PCA Eigenvalue Analysis (NAICS, Top 3 PCs)")
    print(f"  Eigenvalues:  {eig_parquet_path}")
    print(f"  Results:      {RESULTS_DIR}/")
    print(f"  Total time:   {total_time/3600:.1f} hours ({total_time/60:.0f} minutes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    create_signal()
