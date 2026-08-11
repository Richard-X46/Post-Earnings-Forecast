"""
Backup of S3 delta data for the following tables:
1. earnings data from alpha vantage
2. ohlcv data from yfinance for all symbols in the S&P 500 list
3. transcripts data from the earnings call transcript API (temp, incomplete)

Also performs S3 delta vacuum for cleanup.
"""

import os

import polars as pl
from deltalake import DeltaTable
from dotenv import load_dotenv

BACKUP_PATH = "src/ingestion/data/backup/"


def _get_storage_options():
    return {
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
        "AWS_REGION": "ca-central-1",
    }


def _delta_path(table: str) -> str:
    bucket = os.getenv("S3_BUCKET")
    return f"s3://{bucket}/post-earnings-forecast/{table}_delta/"


def backup_earnings(storage_options: dict) -> pl.DataFrame:
    path = _delta_path("earnings")
    df = pl.scan_delta(path, storage_options=storage_options).collect()
    out = f"{BACKUP_PATH}earnings_delta_backup.parquet"
    df.write_parquet(out)
    print(f"Earnings backup written: {out}  ({df.height} rows)")
    return df


def backup_ohlcv(storage_options: dict) -> pl.DataFrame:
    path = _delta_path("ohlcv")
    df = pl.scan_delta(path, storage_options=storage_options).collect()
    out = f"{BACKUP_PATH}ohlcv_delta_backup.parquet"
    df.write_parquet(out)
    print(f"OHLCV backup written:   {out}  ({df.height} rows)")
    return df


def backup_transcripts(storage_options: dict) -> pl.DataFrame:
    path = _delta_path("transcripts")
    df = pl.scan_delta(path, storage_options=storage_options).collect()
    out = f"{BACKUP_PATH}temp_transcripts.parquet"
    df.write_parquet(out)
    print(f"Transcript backup written: {out}  ({df.height} rows)")
    return df


def vacuum_delta_tables(storage_options: dict):
    tables = ["earnings", "ohlcv", "transcripts"]
    for table in tables:
        path = _delta_path(table)
        dt = DeltaTable(path, storage_options=storage_options)
        dead_files = dt.vacuum(retention_hours=0, dry_run=True, enforce_retention_duration=False)
        print(f"Files to be deleted from {table}_delta: {len(dead_files)}")
        dt.vacuum(retention_hours=0, dry_run=False, enforce_retention_duration=False)


def smoke_test(storage_options: dict):
    print("=== Smoke test: backup ===")
    path = _delta_path("earnings")
    try:
        df = pl.scan_delta(path, storage_options=storage_options).limit(1).collect()
        if df.is_empty():
            raise RuntimeError("Earnings delta table returned no rows")
        print(f"SUCCESS: Connected to earnings delta, {df.columns} columns available")
    except Exception as e:
        print(f"FAIL: Could not read earnings delta — {e}")


def main():
    load_dotenv()
    storage_options = _get_storage_options()

    os.makedirs(BACKUP_PATH, exist_ok=True)

    backup_earnings(storage_options)
    backup_ohlcv(storage_options)
    backup_transcripts(storage_options)

    vacuum_delta_tables(storage_options)


if __name__ == "__main__":
    storage_options = _get_storage_options()
    smoke_test(storage_options)
    main()
