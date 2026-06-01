import os
import pyarrow as pa
from deltalake import DeltaTable
from dotenv import load_dotenv

load_dotenv()

bucket = os.getenv("S3_BUCKET")

storage_options = {
    "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID"),
    "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY"),
    "AWS_REGION": "ca-central-1",
}

TABLES = {

    # delta table for ohlcv data
    "ohlcv": {
        "path": f"s3://{bucket}/post-earnings-forecast/ohlcv_delta/",
        "partition_by": ["symbol"],
        "schema": pa.schema([
            pa.field("symbol",  pa.string()),
            pa.field("date",    pa.date32()),
            pa.field("open",    pa.float64()),
            pa.field("high",    pa.float64()),
            pa.field("low",     pa.float64()),
            pa.field("close",   pa.float64()),
            pa.field("volume",  pa.int64()),
        ])
    },

    # delta table for earnings data
    "earnings": {
    "path": f"s3://{bucket}/post-earnings-forecast/earnings_delta/",
    "partition_by": ["symbol"],
    "schema": pa.schema([
        pa.field("fiscalDateEnding",    pa.date32()),
        pa.field("reportedDate",        pa.date32()),
        pa.field("reportedEPS",         pa.string()),
        pa.field("estimatedEPS",        pa.string()),
        pa.field("surprise",            pa.string()),
        pa.field("surprisePercentage",  pa.string()),
        pa.field("reportTime",          pa.string()),
        pa.field("symbol",              pa.string()),
        pa.field("fiscal_month",        pa.int8()),
        pa.field("fiscal_year",         pa.int32()),
        pa.field("av_quarter",          pa.string()),
        pa.field("time_from",           pa.string()),
    ])
}

    # "transcripts": {
    # "path": f"s3://{bucket}/post-earnings-forecast/transcripts_delta/",
    # "partition_by": ["symbol"],

}




# create delta tables if they don't exist

for name, config in TABLES.items():
    DeltaTable.create(
        config["path"],
        schema=config["schema"],
        partition_by=config["partition_by"],
        mode="ignore",
        storage_options=storage_options
    )

