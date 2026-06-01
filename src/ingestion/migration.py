import duckdb
import polars as pl
import os
from dotenv import load_dotenv
from deltalake.writer import write_deltalake
from deltalake import DeltaTable
import json

load_dotenv()



# duckdb connection setup for reading from s3 and writing to delta lake
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

bucket = os.getenv('S3_BUCKET')

# We use the glob syntax (*/*/*.parquet) to force the engine to look for files directly 
# rather than crawling the folder tree recursively
s3_path = f"s3://{bucket}/post-earnings-forecast/transcripts/*/*/*.parquet"



# storage options for Delta Lake
       
reader = con.execute("""
    SELECT * 
    FROM read_parquet(
        's3://docks-otu-canada-central-data/post-earnings-forecast/transcripts/*/*/*.parquet',
        hive_partitioning=true,
        union_by_name=true
    )
    ORDER BY symbol
""").fetch_record_batch(rows_per_batch=50_000)

write_deltalake(
    "s3://docks-otu-canada-central-data/post-earnings-forecast/transcripts_delta/",
    reader,
    partition_by=["symbol"],
    mode="overwrite",
    schema_mode="overwrite",
    storage_options={
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }

)


# checking the same table now

transcript = pl.read_delta(
    f"s3://{bucket}/post-earnings-forecast/transcripts_delta/",
    storage_options={
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    }
)


transcript.columns
# check how much memoryy the dataframe is using
transcript.estimated_size() / 1e9  # in GB


#---- // local backup for transcript data
path = "src/ingestion/data/backup"
# writing to local parquet file
transcript.write_parquet(f"{path}/temp_transcripts.parquet")


# checking duplicates if they exist for a combination of symbol and av_quarter , this is a polars frame
duplicates_df = (
    transcript
    .select(["symbol", "av_quarter"])
    .group_by(["symbol", "av_quarter"])
    .len(name="count")
    .filter(pl.col("count") > 1)
)



# validation if the transcripts does contain data


# adding a new column transcript length to validate

transcript = transcript.with_columns(
    pl.col("transcript")
    .list.eval(pl.element().struct.field("content"))
    .list.join(" ")
    .str.len_chars()
    .alias("content_length")
)

# distribution of content length
transcript.select(pl.col("content_length")).describe()

# mean of content length
mean = transcript.select(pl.col("content_length")).mean()[0, 0]
std = transcript.select(pl.col("content_length")).std()[0, 0]
threshold = mean - (2 * std)

flagged = transcript.filter(
    (pl.col("content_length") < threshold) | 
    (pl.col("content_length").is_null())
).select(["symbol", "av_quarter","transcript", "content_length"])    

flagged.write_clipboard()  # copy to clipboard for review
flagged['content_length'].value_counts().sort("content_length").write_clipboard()  # copy to clipboard for review

transcript.sort("content_length", descending=False).select(["symbol", "av_quarter", "content_length"]).to_pandas().to_clipboard()


# filter IVZ	2025Q4
transcript.filter((pl.col("symbol") == "IVZ") & (pl.col("av_quarter") == "2025Q4"))['transcript']



mean = 43_969
std  = 12_749
threshold = mean - (2 * std)  # ~18,471 chars

bad = transcript.filter(
    (pl.col("content_length") < threshold) | 
    (pl.col("content_length").is_null())
).select(["symbol", "av_quarter", "transcript_length", "content_length"])\
 .sort("content_length")

print(f"Flagged: {len(bad)} rows")
bad







# ---/// table schema checks

dt = DeltaTable(
    f"s3://{bucket}/post-earnings-forecast/transcripts_delta/",
    storage_options={
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    },
)

print(json.dumps(json.loads(dt.schema().to_json()), indent=2))