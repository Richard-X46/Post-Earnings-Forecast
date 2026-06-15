"""
Downloads VIX and SPX daily data via yfinance, writes to data/index_data.parquet.
"""

import sys
from pathlib import Path

import polars as pl
import yfinance as yf

START = "2013-12-01"   
OUT_PATH = Path("src/ingestion/data/index_data.parquet")

TICKERS = {
    "^VIX": "vix",
    "^GSPC": "spx",
}


def fetch_ticker(ticker: str, name: str) -> pl.DataFrame:
    """Download one index from Yahoo and return date + close as Polars."""
    df = yf.download(ticker, start=START, auto_adjust=True, progress=False)

    # Fail loudly here rather than letting an empty frame flow downstream
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    df = df.reset_index()
    # yfinance quirk: newer versions return MultiIndex (tuple) columns,
    # e.g. ("Close", "^VIX"). Handle both shapes so it works on any version.
    df.columns = [
        (c[0] if isinstance(c, tuple) else c).lower() for c in df.columns
    ]

    return pl.from_pandas(df).select(
        pl.col("date").cast(pl.Date),
        pl.col("close").alias(f"{name}_close"),
    )


def main(force: bool = False):
    if OUT_PATH.exists() and not force:
        print(f"{OUT_PATH} already exists. Use --force to re-download.")
        return

    frames = [fetch_ticker(t, n) for t, n in TICKERS.items()]

    # Inner join: keep only dates where both indices traded
    out = frames[0].join(frames[1], on="date", how="inner").sort("date")

    # --- Verification log: confirm coverage before writing ---
    print(f"Rows:       {out.height}")
    print(f"Date range: {out['date'].min()} → {out['date'].max()}")
    print(f"Nulls:      {out.null_count().to_dicts()[0]}")

    # Sanity checks — crash with a clear message if the download looks wrong
    assert out["date"].min().year <= 2013, "Data does not start in 2014!"
    assert out.height > 2900, "Suspiciously few rows — check the download."

    OUT_PATH.parent.mkdir(exist_ok=True)   # create data/ if missing
    out.write_parquet(OUT_PATH)
    print(f"Written to {OUT_PATH}")


# Runs only when executed directly
if __name__ == "__main__":
    force = "--force" in sys.argv

    main(force)