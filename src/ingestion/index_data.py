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
    df = yf.download(ticker, start=START, auto_adjust=True, progress=False)

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    df = df.reset_index()
    df.columns = [
        (c[0] if isinstance(c, tuple) else c).lower() for c in df.columns
    ]

    return pl.from_pandas(df).select(
        pl.col("date").cast(pl.Date),
        pl.col("close").alias(f"{name}_close"),
    )


def smoke_test():
    print("=== Smoke test: index_data ===")
    test_ticker, test_name = next(iter(TICKERS.items()))
    try:
        df = fetch_ticker(test_ticker, test_name)
        if df.is_empty():
            raise RuntimeError("Fetched empty DataFrame")
        print(f"SUCCESS: Fetched {df.height} rows for {test_ticker} ({test_name})")
    except Exception as e:
        print(f"FAIL: Could not fetch {test_ticker} — {e}")


def main(force: bool = False):
    if OUT_PATH.exists() and not force:
        print(f"{OUT_PATH} already exists. Use --force to re-download.")
        return

    frames = [fetch_ticker(t, n) for t, n in TICKERS.items()]

    out = frames[0].join(frames[1], on="date", how="inner").sort("date")

    print(f"Rows:       {out.height}")
    print(f"Date range: {out['date'].min()} → {out['date'].max()}")
    print(f"Nulls:      {out.null_count().to_dicts()[0]}")

    assert out["date"].min().year <= 2013, "Data does not start in 2014!"
    assert out.height > 2900, "Suspiciously few rows — check the download."

    OUT_PATH.parent.mkdir(exist_ok=True)
    out.write_parquet(OUT_PATH)
    print(f"Written to {OUT_PATH}")


if __name__ == "__main__":
    smoke_test()

    force = "--force" in sys.argv
    main(force)
