import polars as pl
import os
from dotenv import load_dotenv


load_dotenv()

storage_options = {
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_REGION": "ca-central-1",
}


# ---/// backup of s3 data for the following tables
# 1. earnings data from alpha vantage
# 2. ohlcv data from yfinance for all symbols in the S&P 500 list
# 3. transcripts data from the earnings call transcript API , temp added for now as it is incomplete
# backup path 
BACKUP_PATH = "src/ingestion/data/backup/"


#EARNINGS BACKUP
DELTA_EARNINGS = f"s3://{os.getenv('S3_BUCKET')}/post-earnings-forecast/earnings_delta/"

earnigns_df = pl.scan_delta(DELTA_EARNINGS, storage_options=storage_options).collect()
earnigns_df.write_parquet(f"{BACKUP_PATH}earnings_delta_backup.parquet")

# OHLCV BACKUP
DELTA_OHLCV = f"s3://{os.getenv('S3_BUCKET')}/post-earnings-forecast/ohlcv_delta/"

ohlcv_df = pl.scan_delta(DELTA_OHLCV, storage_options=storage_options).collect()
ohlcv_df.write_parquet(f"{BACKUP_PATH}ohlcv_delta_backup.parquet")

#TRANSCRIPT BACKUP
DELTA_TRANSCRIPT = f"s3://{os.getenv('S3_BUCKET')}/post-earnings-forecast/transcripts_delta/"
transcript_df = pl.scan_delta(DELTA_TRANSCRIPT, storage_options=storage_options).collect()
transcript_df.write_parquet(f"{BACKUP_PATH}temp_transcripts.parquet") # temp for now till validation is done on all syms





### S3 Delta vacuum for cleanup
# Removes files not referenced in the transaction log. Useful for
# cleaning up old files after compaction or backfill and for
# reducing storage costs by deleting unreferenced files.


DELTA_TABLES =[DELTA_EARNINGS, DELTA_OHLCV, DELTA_TRANSCRIPT]

for delta_table in DELTA_TABLES:
    dt = DeltaTable(delta_table, storage_options=storage_options)
    #checking for files that would be deleted with a dry run
    dead_files = dt.vacuum(retention_hours=0, dry_run=True,enforce_retention_duration=False)
    print(f"Files to be deleted from {delta_table}: {len(dead_files)}")

    # actually delete
    dt.vacuum(retention_hours=0, dry_run=False,enforce_retention_duration=False)


dt.update()
dead_files = dt.vacuum(retention_hours=0, dry_run=True, enforce_retention_duration=False)
print(len(dead_files))

dt.load_as_version(dt.version())


