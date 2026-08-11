# yfinance backfill for all syms in the S&P 500 list for OHLCV data.
#  This is to backfill missing data for the 3 symbols that were missing from the initial backfill - BRK-B, BF-B, AZO.
#  The missing data was due to yfinance ticker symbol issues with the hyphen in the symbol names.

import yfinance as yf
import duckdb
import polars as pl
import os
from dotenv import load_dotenv
from deltalake import DeltaTable
from concurrent.futures import ThreadPoolExecutor


load_dotenv()

bucket = os.getenv("S3_BUCKET")
DELTA_PATH = f"s3://{bucket}/post-earnings-forecast/ohlcv_delta/"

storage_options = {
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_REGION": "ca-central-1",
}


def load_snp500_symbols():
    source = f"s3://{bucket}/post-earnings-forecast/snp500/*.csv"
    return pl.read_csv(source, storage_options={
        "key": os.getenv("S3_ACCESS_KEY"),
        "secret": os.getenv("S3_SECRET_KEY"),
        "token": os.getenv("AWS_SESSION_TOKEN"),
        "expand": True,
    })


def fetch_ohlcv(symbol: str) -> pl.DataFrame | None:
    hist = yf.Ticker(symbol).history(period="max")
    if hist.empty:
        return None
    return (
        pl.from_pandas(hist.reset_index())
        .select(["Date", "Open", "High", "Low", "Close", "Volume"])
        .rename({"Date": "date", "Open": "open", "High": "high",
                 "Low": "low", "Close": "close", "Volume": "volume"})
        .with_columns(pl.lit(symbol).alias("symbol"))
        .with_columns(pl.col("date").dt.date())
    )


def write_ohlcv(df: pl.DataFrame):
    DeltaTable(DELTA_PATH, storage_options=storage_options).merge(
        source=df.to_arrow(),
        predicate="target.symbol = source.symbol AND target.date = source.date",
        source_alias="source",
        target_alias="target"
    ).when_matched_update_all() \
     .when_not_matched_insert_all() \
     .execute()


def get_existing_symbols():
    df = pl.read_delta(DELTA_PATH, storage_options=storage_options)
    return df.select("symbol").unique().to_series().to_list()


def find_pending_symbols(all_symbols, existing_symbols):
    return [sym for sym in all_symbols if sym not in existing_symbols]


def backfill_ohlcv(symbols):
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(fetch_ohlcv, symbols))
    all_data = [df for df in results if df is not None]
    if all_data:
        write_ohlcv(pl.concat(all_data))
    return all_data


def smoke_test():
    print("=== Smoke test: fetch_ohlcv ===")
    test_sym = "IBM"
    df = fetch_ohlcv(test_sym)
    if df is not None and not df.is_empty():
        print(f"SUCCESS: Fetched {df.height} rows for {test_sym}")
        print(df.head(3))
    else:
        print(f"FAILURE: No data returned for {test_sym}")


if __name__ == "__main__":
    smoke_test()

    snp500_df = load_snp500_symbols()
    symbols = snp500_df["Symbol"].to_list()

    existing = get_existing_symbols()
    pending = find_pending_symbols(symbols, existing)

    print(f"Total symbols: {len(symbols)}")
    print(f"Existing symbols: {len(existing)}")
    print(f"Pending symbols: {len(pending)}")

    fixed_pending_syms = ["BRK-B", "BF-B"]
    backfill_ohlcv(fixed_pending_syms)

    con = duckdb.connect()
    con.execute("INSTALL aws; LOAD aws;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("CALL load_aws_credentials();")

    query = f"""
        SELECT *
        FROM delta_scan('{DELTA_PATH}')
    """
    con.execute(query).df()

    df = pl.read_delta(DELTA_PATH, storage_options=storage_options)
    df.head()
    df.filter(pl.col("symbol") == "AAPL").sort("date", descending=False).head()
    df.estimated_size() / 1e9
    df.filter(
        pl.col("date") >= pl.lit("2010-01-01").str.to_date()
    ).to_pandas()

    duplicates_df = (
        df.select(["symbol", "date"])
        .group_by(["symbol", "date"])
        .len(name="count")
        .filter(pl.col("count") > 1)
    )