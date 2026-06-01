# compaction of existing hive earnings to delta table for earnings for 
# downstream joins with ohlcv and news sentiment data

import polars as pl
import os
from ingestion.transcript_news import  tor_get,test_key
from dotenv import load_dotenv
from deltalake import DeltaTable

load_dotenv()


storage_options = {
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_REGION": "ca-central-1",
}


# --- func to get earnings data for a symbol from alpha vantage and return as a dataframe

def get_av_earnings(sym, apikey=test_key):
    """
    Fetch earnings data for a given symbol from Alpha Vantage and return as a DataFrame.

    Parameters:
    sym (str): The stock symbol.
    apikey (str): The Alpha Vantage API key.

    Returns:
    pl.DataFrame: A DataFrame containing the earnings data.
    """
    url = f"https://www.alphavantage.co/query?function=EARNINGS&symbol={sym}&apikey={apikey}"
    print(f"Fetching AV earnings for {sym} from {url}")
    res = tor_get(url)
    try:
        df = pl.DataFrame(res["quarterlyEarnings"])
        df = df.with_columns([
            pl.lit(sym).alias("symbol"),
            pl.col("fiscalDateEnding").str.to_date().alias("fiscalDateEnding"),
            pl.col("reportedDate").str.to_date().alias("reportedDate"),
        ])

        # av_quarter = year of fiscalDateEnding + quarter number based on month
        df = df.with_columns([
            pl.col("fiscalDateEnding").dt.month().alias("fiscal_month"),
            pl.col("fiscalDateEnding").dt.year().alias("fiscal_year"),
            (pl.col("fiscalDateEnding").dt.year().cast(pl.Utf8) + pl.lit("Q") +
            pl.col("fiscalDateEnding").dt.quarter().cast(pl.Utf8)).alias("av_quarter"),
            pl.col("reportedDate").dt.strftime("%Y%m%dT0000").alias("time_from"),
        ])
        
        # quarter end months tell you which quarter it is
        # Jan/Apr/Jul/Oct = Q4/Q1/Q2/Q3 depending on fiscal calendar
        # but AV transcript API uses YYYY + Q1-Q4 based on fiscal calendar order
        # safest: just use fiscalDateEnding as the key to match transcripts
        
        df = df.with_columns([
            (pl.col("fiscalDateEnding").dt.year().cast(pl.Utf8) + pl.lit("Q") +
            pl.col("fiscalDateEnding").dt.quarter().cast(pl.Utf8)).alias("av_quarter"),
            pl.col("reportedDate").dt.strftime("%Y%m%dT0000").alias("time_from"),
        ])

        df = df.filter(pl.col("reportedDate") >= pl.date(2014, 1, 1))
        return df
    except Exception as e:
        logging.error(f"Error processing AV earnings data for {sym}: {e}")
        pass



# snp500 symbols from csv in s3

path = "src/ingestion/data/snp500_*.csv"
source = f"s3://{os.getenv('S3_BUCKET')}/post-earnings-forecast/snp500/*.csv"
df = pl.read_csv(source, storage_options={
    "key": os.getenv("S3_ACCESS_KEY"),
    "secret": os.getenv("S3_SECRET_KEY"),
    "token": os.getenv("AWS_SESSION_TOKEN"),
        "expand": True,})



#-----/// compaction of existing hive earnings to delta table for earnings 

# old hive table
hive_earnings_df = pl.scan_parquet(f"s3://{os.getenv('S3_BUCKET')}/post-earnings-forecast/earnings/*/*.parquet",
         storage_options=storage_options).collect()
hive_earnings_df.columns



#  delta table for earnings / check table_setup.py for schema definition and creation of delta table
DELTA_EARNINGS = f"s3://{os.getenv('S3_BUCKET')}/post-earnings-forecast/earnings_delta/"



# writing to a new s3 delta lake table name - earnings_delta
DeltaTable(DELTA_EARNINGS, storage_options=storage_options).merge(
        source=hive_earnings_df.to_arrow(),
        predicate="target.symbol = source.symbol AND target.av_quarter = source.av_quarter",
        source_alias="source",
        target_alias="target"
    ).when_matched_update_all() \
     .when_not_matched_insert_all() \
     .execute()


# scanning the delta table to validate compaction
delta_earnings_df = pl.scan_delta(DELTA_EARNINGS, storage_options=storage_options).collect()
delta_earnings_df.head()


delta_earnings_df['symbol'].unique()    




# ---// dealing with missing syms


syms = df["Symbol"].to_list()

earnings_existing_syms = delta_earnings_df["symbol"].unique().to_list()

missing_syms = set(syms) - set(earnings_existing_syms)

missing_manual = ['AZO', 'BF-B', 'BRK-B'] # replace . with - for BRK-B and BF-B

# checking if missing manual part of delta_earnings_df
delta_earnings_df.filter(pl.col('symbol').is_in(missing_manual))['symbol'].unique()


# ---- backfill the missing symbols for earnings delta table


for i,sym in enumerate(missing_manual):
    print(f"Processing {sym} ({i+1}/{len(missing_manual)})")
    # get the data from alpha vantage
    av_earnings_df = get_av_earnings(sym)
    if av_earnings_df is not None and not av_earnings_df.is_empty():
        # write to delta lake
        DeltaTable(DELTA_EARNINGS, storage_options=storage_options).merge(
            source=av_earnings_df.to_arrow(),
            predicate="target.symbol = source.symbol AND target.av_quarter = source.av_quarter",
            source_alias="source",
            target_alias="target"
        ).when_matched_update_all() \
         .when_not_matched_insert_all() \
         .execute()
        print(f"Backfilled earnings data for {sym} from Alpha Vantage")
    else:
        print(f"No earnings data found for {sym} from Alpha Vantage")




# 