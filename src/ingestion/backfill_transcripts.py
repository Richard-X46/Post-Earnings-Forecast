# Ingestion backfill for transcripts on s&p500 stocks into s3
# check for existing partition before AV calls to avoid unnecessary API calls and circuit rotations

import yfinance as yf
import pandas as pd
from ingestion.transcript_news import  get_earnings_call_transcript, get_news_sentiment,keygen,tor_get
import duckdb
import datetime
import importlib
import sys
import secrets
from dotenv import load_dotenv
import os
import polars as pl
import base64
import s3fs
import fsspec
import logging
import time
import boto3


# config and setup
load_dotenv()

logging.basicConfig(level=logging.INFO)
fsspec.config.conf["open_expand"] = True  

#duckdb config for encrypted parquet with s3fs
con = duckdb.connect(database=':memory:')
con.sql(f"PRAGMA add_parquet_key('main_key', {os.getenv('DUCKDB_KEY')});")

# av config
test_key = os.getenv("AV_KEYS")

# s3 config
s3 = s3fs.S3FileSystem(
    key=os.getenv("S3_ACCESS_KEY"),
    secret=os.getenv("S3_SECRET_KEY"),
    token=os.getenv("AWS_SESSION_TOKEN"),
)

# s3 setup using boto3
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.getenv("S3_ACCESS_KEY"),
    aws_secret_access_key=os.getenv("S3_SECRET_KEY"),
    aws_session_token=os.getenv("AWS_SESSION_TOKEN"),  # optional but important if using temp creds
    region_name='ca-central-1'
)

keygen()
test_key = keygen()

# --- /// s3 transcript partition check function

def transcript_output_prefix(symbol, quarter):
    bucket = os.getenv("S3_BUCKET")
    return f"{bucket}/post-earnings-forecast/transcripts/symbol={symbol}/av_quarter={quarter}/"

def check_transcript_partition_exists(symbol, quarter):
    prefix = transcript_output_prefix(symbol, quarter)
    try:
        return len(s3.ls(prefix)) > 0
    except FileNotFoundError:
        return False

# --- /// earnings partition check function

def earnings_output_prefix(symbol):
    bucket = os.getenv("S3_BUCKET")
    return f"{bucket}/post-earnings-forecast/earnings/symbol={symbol}/"

def check_earnings_partition_exists(symbol):
    prefix = earnings_output_prefix(symbol)
    try:
        return len(s3.ls(prefix)) > 0
    except FileNotFoundError:
        return False
    return exists

# ---- /// fetching earnings table given a symbol from AV 

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


# ---- /// get transcripts for each av_quarter in av_earnings_df for each symbol

def fetch_transcripts_for_symbol(sym, earnings_df: pl.DataFrame) -> pl.DataFrame:
    """
    For a given symbol, check each av_quarter in the earnings_df and fetch the transcript if it doesn't exist in s3.
    
    """


    sym_df = earnings_df.filter(pl.col("symbol") == sym)

    updates = []
    for row in sym_df.select(["symbol", "av_quarter"]).iter_rows(named=True):
        quarter = row["av_quarter"]
        time.sleep(1)  # brief sleep to avoid hitting rate limits too quickly
        # checking if transcript partition exists for this symbol
        if not check_transcript_partition_exists(sym, quarter):
            logging.info(f"Transcript for {sym} {quarter} not found, fetching...")
            transcript_data = get_earnings_call_transcript(sym, quarter)

            # if transcript data is missing or empty from AV call, log and skip 
            if not transcript_data or "transcript" not in transcript_data:
                logging.warning(f"No transcript found for {sym} {quarter}, skipping...")


                continue
            else:
                logging.info(f"Transcript fetched for {sym} {quarter}, appending to list")
                updates.append({
                    "symbol": sym,
                    "av_quarter": quarter,
                    "transcript": transcript_data["transcript"],
                })
            
        else:
            logging.info(f"Transcript partition already exists for {sym} {quarter}, skipping...")
            continue

        
    # combine updates into a dataframe and return
    if updates:
        updates_df = pl.DataFrame(updates)
        return earnings_df.join(updates_df, on=["symbol", "av_quarter"], how="left")
    else:
        logging.info(f"No new transcripts fetched for {sym}, returning original earnings_df")
        return earnings_df


# function to handle earnings write

def write_earnings_to_s3(symbol, earnings_df):
    if not check_earnings_partition_exists(symbol):
        logging.info(f"Earnings partition not found for {symbol}, writing to s3...")
        earnings_df.write_parquet(
            f"s3://{os.getenv('S3_BUCKET')}/post-earnings-forecast/earnings/",
            partition_by=["symbol"],
            storage_options={
                "key": os.getenv("AWS_ACCESS_KEY_ID"),
                "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
                                    },
                                    
                                    )
    else:
        logging.info(f"Earnings partition already exists for {symbol}, skipping write.")



def write_transcripts_to_s3(symbol, transcripts_df):
    
    transcripts_df.write_parquet(
        f"s3://{os.getenv('S3_BUCKET')}/post-earnings-forecast/transcripts/",
        partition_by=["symbol", "av_quarter"],
        storage_options={
            "key": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret": os.getenv("AWS_SECRET_ACCESS_KEY"),
                                },
                                )
    start_quarter_date = transcripts_df.select(pl.col("reportedDate").min()).item()
    end_quarter_date = transcripts_df.select(pl.col("reportedDate").max()).item()
    total_records = transcripts_df.height
    logging.info(f"{total_records} Transcripts written to s3 for {symbol}from {start_quarter_date} to {end_quarter_date}")


# compaction of transcripts 
def compact_transcripts(symbol):
    pass



# checking symbols that exist on s3 for transcripts

def get_syms_that_have_transcripts():
    bucket = os.getenv("S3_BUCKET")
    paginator = s3_client.get_paginator("list_objects_v2")
    symbols = []

    for page in paginator.paginate(
        Bucket=bucket,
        Prefix="post-earnings-forecast/transcripts/",
        Delimiter="/"
    ):
        symbols += [
            p["Prefix"].split("symbol=")[1].rstrip("/")
            for p in page.get("CommonPrefixes", [])
            if "symbol=" in p["Prefix"]
        ]
    return symbols

if __name__ == "__main__":


    # read snp500 data from csv
    path = "src/ingestion/data/snp500_*.csv"
    source = f"s3://{os.getenv('S3_BUCKET')}/post-earnings-forecast/snp500/*.csv"
    df = pl.read_csv(source, storage_options={
        "key": os.getenv("S3_ACCESS_KEY"),
        "secret": os.getenv("S3_SECRET_KEY"),
        "token": os.getenv("AWS_SESSION_TOKEN"),
            "expand": True,})

    # 
    
 
    #total symbols in snp500 list
    total_symbols = df.select(pl.col("Symbol")).unique().to_series().to_list()


    # symbols that already have transcripts on s3 to avoid unnecessary AV calls and circuit rotations
    symbols_with_transcripts = get_syms_that_have_transcripts()


    my_frame = pd.read_clipboard(header=None)

    #my_list = my_frame['symbols_to_backfill'].tolist()
    my_list = my_frame.values.flatten().tolist()

    # symbols to backfill transcripts

    #symbols_to_backfill = [sym for sym in total_symbols if sym not in symbols_with_transcripts]
    
    symbols_to_backfill = [sym for sym in my_list if sym not in symbols_with_transcripts]
    # symbols_to_backfill= symbols_to_backfill[:10] # limit to 10 for testing, remove this line for full backfill




    # temp_df = pd.DataFrame({"symbols_to_backfill": symbols_to_backfill})
    # temp_df.to_clipboard(index=False)

    for idx, sym in enumerate(symbols_to_backfill):
        print(f"Processing symbol: {sym} ({idx+1}/{len(symbols_to_backfill)})")

        av_earnings_df = get_av_earnings(sym, apikey=test_key) # API call to AV
        if av_earnings_df is None or av_earnings_df.height == 0:
            logging.warning(f"No earnings data found for {sym}, skipping...")
            continue
        write_earnings_to_s3(sym, av_earnings_df)

        # earnings table needs to be pulled from s3 then fetch transcripts for each sym

        transcripts_df = fetch_transcripts_for_symbol(sym, av_earnings_df) # API calls to AV with tor circuit rotation for transcripts
        write_transcripts_to_s3(sym, transcripts_df)






    # # fetch earnings data for the test symbol
    # av_earnings_df = get_av_earnings(test_sym, apikey=test_key) # API call to AV
    # write_earnings_to_s3(test_sym, av_earnings_df)

    # test_earnings = av_earnings_df.head()

    # # fetch transcripts for the test symbol and merge with earnings df
    # transcripts_df = fetch_transcripts_for_symbol(test_sym, test_earnings) # API calls to AV with tor circuit rotation for transcripts

    # # write transcripts to s3
    # write_transcripts_to_s3(test_sym, transcripts_df)

