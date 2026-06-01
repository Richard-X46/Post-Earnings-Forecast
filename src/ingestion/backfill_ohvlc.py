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


# path = "src/ingestion/data/snp500_*.csv"
source = f"s3://{os.getenv('S3_BUCKET')}/post-earnings-forecast/snp500/*.csv"
df = pl.read_csv(source, storage_options={
    "key": os.getenv("S3_ACCESS_KEY"),
    "secret": os.getenv("S3_SECRET_KEY"),
    "token": os.getenv("AWS_SESSION_TOKEN"),
        "expand": True,})



# new delta table 

# create a duckdb connection
con = duckdb.connect()
con.execute("INSTALL aws; LOAD aws;")
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("CALL load_aws_credentials();") 



bucket = os.getenv('S3_BUCKET')


# writing to s3 delta lake table name - 

symbols = df["Symbol"].to_list()


DELTA_PATH = f"s3://{bucket}/post-earnings-forecast/ohlcv_delta/"

storage_options = {
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_REGION": "ca-central-1",
}

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


# fetch_ohlcv("AAPL")


def write_ohlcv(df: pl.DataFrame):
    DeltaTable(DELTA_PATH, storage_options=storage_options).merge(
        source=df.to_arrow(),
        predicate="target.symbol = source.symbol AND target.date = source.date",
        source_alias="source",
        target_alias="target"
    ).when_matched_update_all() \
     .when_not_matched_insert_all() \
     .execute()



# scan ohlcv table to check for duplicates

df = pl.read_delta(DELTA_PATH, storage_options=storage_options)

existing_syms = df.select("symbol").unique().to_series().to_list()

pending_syms = [sym for sym in symbols if sym not in existing_syms]

print(f"Total symbols: {len(symbols)}")
print(f"Existing symbols: {len(existing_syms)}")
print(f"Pending symbols: {len(pending_syms)}")

# pending symbols

fixed_pending_syms = ['BRK-B', 'BF-B']


# fetch all symbols in parallel - network bound so threads help
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(fetch_ohlcv, fixed_pending_syms))


all_data = [df for df in results if df is not None]
len(all_data)
len(results)

# write_deltalake(
#     DELTA_PATH,
#     pl.concat(all_data).to_arrow(),
#     partition_by=["symbol"],
#     mode="overwrite", # this nukes everything
#     storage_options=storage_options
# )

DeltaTable(DELTA_PATH, storage_options=storage_options).merge(
        source=pl.concat(all_data).to_arrow(),
        predicate="target.symbol = source.symbol AND target.date = source.date",
        source_alias="source",
        target_alias="target"
    ).when_matched_update_all() \
     .when_not_matched_insert_all() \
     .execute()



# -------

# duckdb scan on delta table to get data
query = f"""
    SELECT *
    FROM delta_scan('{DELTA_PATH}')
"""

con.execute(query).df()



storage_options = {
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_REGION": "ca-central-1",
}
bucket = os.getenv('S3_BUCKET')
DELTA_PATH = f"s3://{bucket}/post-earnings-forecast/ohlcv_delta/"
df = pl.read_delta(DELTA_PATH, storage_options=storage_options)
df.head()


# filter df for AAPL sort by date
df.filter(pl.col("symbol") == "AAPL").sort("date", descending=False).head()

# size of the dataframe in GB
df.estimated_size() / 1e9

# filter on data for after 2010 only
df.filter(
    pl.col("date") >= pl.lit("2010-01-01").str.to_date()
).to_pandas()


# checking if duplicates exist for symbol and date combination
duplicates_df = (
    df.select(["symbol", "date"])
    .group_by(["symbol", "date"])
    .len(name="count")
    .filter(pl.col("count") > 1)
)