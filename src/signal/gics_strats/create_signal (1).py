from dotenv import load_dotenv
load_dotenv() # This MUST come before the sf_quant imports

import os
import polars as pl
import numpy as np
import datetime as dt
import sf_quant.data as sfd
import sf_quant.research as sfr
import polars_ols

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

def create_signal():
    """
    Loads data, creates a simple signal, and saves it to parquet.
    """
    # Load environment variables from .env file
    load_dotenv()
    project_root = os.getcwd()
    output_path = os.getenv("SIGNAL_PATH", "data/signal.parquet")
    if not os.path.isabs(output_path):
        output_path = os.path.join(project_root, output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)


    # TODO: Load Data
    df = load_data()

    # TODO: Add your signal logic here (remember alpha logic)

    # TODO: Save to data/signal.parquet

    # df.write_parquet(output_path)

if __name__ == "__main__":
    create_signal()
