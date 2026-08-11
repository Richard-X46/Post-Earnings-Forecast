"""
Migrates transcripts data from S3 hive-partitioned parquet to Delta Lake format.
Also validates the resulting table with schema and content checks.
"""

import json
import os

import duckdb
import polars as pl
from deltalake import DeltaTable
from deltalake.writer import write_deltalake
from dotenv import load_dotenv

load_dotenv()

BUCKET = os.getenv("S3_BUCKET")
DELTA_PATH = f"s3://{BUCKET}/post-earnings-forecast/transcripts_delta/"
S3_PARQUET_PATH = f"s3://{BUCKET}/post-earnings-forecast/transcripts/*/*/*.parquet"
BACKUP_PATH = "src/ingestion/data/backup/"


def connect_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("INSTALL aws; LOAD aws;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("CALL load_aws_credentials();")
    con.execute("""
        CREATE OR REPLACE SECRET s3_clean_secret (
            TYPE S3,
            PROVIDER credential_chain,
            CHAIN 'env;config',
            REGION 'ca-central-1'
        );
    """)
    return con


def _storage_options():
    return {
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }


def migrate_transcripts(con: duckdb.DuckDBPyConnection) -> pl.DataFrame:
    reader = con.execute(f"""
        SELECT *
        FROM read_parquet(
            '{S3_PARQUET_PATH}',
            hive_partitioning=true,
            union_by_name=true
        )
        ORDER BY symbol
    """).fetch_record_batch(rows_per_batch=50_000)

    write_deltalake(
        DELTA_PATH,
        reader,
        partition_by=["symbol"],
        mode="overwrite",
        schema_mode="overwrite",
        storage_options=_storage_options(),
    )
    print(f"Migration complete: written to {DELTA_PATH}")

    transcript = pl.read_delta(DELTA_PATH, storage_options=_storage_options())
    return transcript


def validate_transcripts(transcript: pl.DataFrame):
    print("Columns:", transcript.columns)
    size_gb = transcript.estimated_size() / 1e9
    print(f"Estimated size: {size_gb:.2f} GB")

    duplicates_df = (
        transcript
        .select(["symbol", "av_quarter"])
        .group_by(["symbol", "av_quarter"])
        .len(name="count")
        .filter(pl.col("count") > 1)
    )
    print(f"Duplicate (symbol, av_quarter) rows: {duplicates_df.height}")

    transcript_with_length = transcript.with_columns(
        pl.col("transcript")
        .list.eval(pl.element().struct.field("content"))
        .list.join(" ")
        .str.len_chars()
        .alias("content_length")
    )

    print("Content length distribution:")
    print(transcript_with_length.select(pl.col("content_length")).describe())

    mean = transcript_with_length.select(pl.col("content_length")).mean()[0, 0]
    std = transcript_with_length.select(pl.col("content_length")).std()[0, 0]
    threshold = mean - (2 * std)

    flagged = transcript_with_length.filter(
        (pl.col("content_length") < threshold)
        | (pl.col("content_length").is_null())
    ).select(["symbol", "av_quarter", "transcript", "content_length"])

    print(f"Flagged rows (content_length < {threshold:.0f}): {flagged.height}")

    return transcript_with_length, flagged


def dump_schema():
    dt = DeltaTable(DELTA_PATH, storage_options=_storage_options())
    print(json.dumps(json.loads(dt.schema().to_json()), indent=2))


def smoke_test():
    print("=== Smoke test: migration ===")
    try:
        con = connect_duckdb()
        result = con.execute("SELECT 1").fetchone()
        if result[0] != 1:
            raise RuntimeError("DuckDB connection test failed")
        print("SUCCESS: DuckDB connected with AWS credentials")
        con.close()
    except Exception as e:
        print(f"FAIL: DuckDB connection — {e}")


def main():
    load_dotenv()
    con = connect_duckdb()

    transcript = migrate_transcripts(con)

    validate_transcripts(transcript)

    os.makedirs(BACKUP_PATH, exist_ok=True)
    transcript.write_parquet(f"{BACKUP_PATH}temp_transcripts.parquet")
    print(f"Local backup written to {BACKUP_PATH}temp_transcripts.parquet")

    dump_schema()

    con.close()


if __name__ == "__main__":
    load_dotenv()
    smoke_test()
    main()
